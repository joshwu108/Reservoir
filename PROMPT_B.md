# Autonomous Session — Plan B: reservoir-anchor
# Catastrophic Forgetting Monitor for LLM Fine-tuning

You are an autonomous agent. Work continuously until every task is complete.
Do not stop to ask questions. If you hit a decision not covered below, make
the most reasonable choice, record it in DECISIONS_B.md, and continue.
You have permission to install any packages needed using pip or uv.
Do NOT make any git commits — the user will review and commit manually.
Do not stop until all code is written, all tests pass, and the completion
checklist at the bottom is fully checked off.

---

## What you are building

A Python module called reservoir-anchor that monitors for catastrophic
forgetting during LLM fine-tuning — in real time, before your eval suite
fails.

The problem: when you fine-tune a model on new data it forgets what it
previously knew. Teams currently detect this AFTER training by running
eval suites. By then the compute is wasted and you have to retrain.

This module gives you an early warning during training by:
1. Holding a small AnchorSet of examples representing prior knowledge
2. Running them through the model every N steps to track their loss
3. Alerting when anchor loss rises significantly above baseline
4. Optionally replaying the most-forgotten anchors back into training
   using PER to prioritize the examples being forgotten most severely

The PER connection is direct: priority = how much the model has forgotten
this specific anchor (relative loss increase from baseline). The sum-tree
samples the most-forgotten anchors proportionally for replay, with IS
weight correction to keep the gradient unbiased.

---

## Project location and setup

Project: /Users/joshuawu/Reservoir
Python: 3.13
Package manager: uv
Run tests: cd /Users/joshuawu/Reservoir && uv run pytest tests/ -v
Existing package: reservoir (src/reservoir/) — already has FastPERBuffer,
ExactPERBuffer, C extension (_sumtree), etc.
All 270 existing tests must still pass at the end.

Install packages with:
  uv add <package>                    # adds to pyproject.toml
  uv run pip install <package>        # install directly into venv

Install these before starting:
  uv add --optional anchor "transformers>=4.40" "datasets>=2.18" "accelerate>=0.27"
  uv run pip install transformers datasets accelerate

Do NOT install trl — that is used by the separate Plan A session.

---

## Files to create

DO NOT modify any of these existing files:
  src/reservoir/fast_buffer.py
  src/reservoir/buffer.py
  src/reservoir/c_buffer.py
  src/reservoir/attest.py
  src/reservoir/durable.py
  src/reservoir/draw.py
  src/reservoir/sumtree.py
  src/reservoir/rational.py
  Any existing test files

DO create these new files:
  src/reservoir/anchor_set.py
  src/reservoir/forgetting_monitor.py
  src/reservoir/replay_scheduler.py
  tests/test_anchor_set.py
  tests/test_forgetting_monitor.py
  tests/test_replay_scheduler.py
  DECISIONS_B.md

DO append (do not replace) to:
  src/reservoir/__init__.py — add these lines at the very bottom only:
    from reservoir.anchor_set import AnchorSet
    from reservoir.forgetting_monitor import ForgettingMonitor, ForgettingAlert
    from reservoir.replay_scheduler import ReplayScheduler

---

## Module 1: src/reservoir/anchor_set.py

A labeled collection of examples representing knowledge the model should
not forget during fine-tuning.

```python
@dataclass
class AnchorExample:
    idx: int                  # position in the anchor set (0-indexed)
    data: dict                # the actual example (tokenized or raw)
    tag: str                  # user-supplied group label e.g. "legal-QA"
    baseline_loss: float      # loss at the start of fine-tuning (snapshot)
    current_loss: float       # most recently measured loss
    priority: float           # current forgetting severity (updated by monitor)

class AnchorSet:
    """
    A fixed set of examples representing prior knowledge to preserve.

    Parameters
    ----------
    examples : list[dict]
        Raw examples. Each must be a dict with at least one text field.
    tags : str | list[str]
        Group tag(s) for all examples, or one tag per example.
        Used for grouped forgetting alerts. Default "default".
    n : int | None
        If provided, subsample to n examples using the strategy below.
    strategy : "random" | "priority-stratified"
        How to subsample if n < len(examples).
        "random": random.sample
        "priority-stratified": split examples into quartiles by initial loss
        and sample proportionally from each quartile.
        Default "random" (priority-stratified requires initial losses which
        may not be available at construction time).
    """
```

Public methods:
- __len__() -> int
- __iter__() -> Iterator[AnchorExample]
- __getitem__(idx: int) -> AnchorExample
- groups() -> dict[str, list[AnchorExample]]
    Returns examples grouped by tag.
- snapshot_baseline(losses: dict[int, float]) -> None
    Set baseline_loss for each anchor. Call once at fine-tune start.
    losses is {anchor_idx: loss_value}.
- update_current_losses(losses: dict[int, float]) -> None
    Update current_loss and recompute priority for each anchor.
    priority = max(0, (current_loss - baseline_loss) / (baseline_loss + 1e-8))
    This is relative loss increase: 0 = no forgetting, 1 = loss doubled, etc.
- forgetting_scores() -> dict[str, float]
    Returns {tag: mean_priority} for each group.
    Higher = more forgetting happening in this group.
- most_forgotten(k: int = 10) -> list[AnchorExample]
    Top-k anchors by priority (most forgotten first).

Class method:
- AnchorSet.from_dataset(dataset, n=1000, tags="default", strategy="random")
    Construct from HuggingFace Dataset or list of dicts.

---

## Module 2: src/reservoir/forgetting_monitor.py

HuggingFace TrainerCallback that monitors anchor loss during fine-tuning.

```python
@dataclass
class ForgettingAlert:
    step: int
    tag: str
    forgetting_score: float    # current mean priority for this group
    baseline_score: float      # always 0.0 (baseline is when priority=0)
    threshold: float           # the threshold that was exceeded
    message: str               # human-readable description

class ForgettingMonitor(TrainerCallback):
    """
    HuggingFace TrainerCallback that monitors for catastrophic forgetting.

    Attach to any HuggingFace Trainer to get real-time forgetting alerts.

    Parameters
    ----------
    anchor_sets : list[AnchorSet]
        One or more anchor sets to monitor. Each can have different tags.
    eval_every_n_steps : int
        How often to evaluate anchors. Default 500.
    alert_threshold : float
        Fire alert when a group's mean forgetting score exceeds this.
        0.5 means average anchor loss has risen 50% above baseline.
        Default 0.5.
    auto_replay : bool
        If True, inject most-forgotten anchors back into training.
        Default False (monitoring only — safer default).
    replay_ratio : float
        When auto_replay=True, fraction of each batch to replace with
        anchor replay. Default 0.1 (10% of batch is replayed anchors).
    verbose : bool
        Print forgetting scores to console at each evaluation. Default True.
    log_to_wandb : bool
        If True and wandb is installed, log forgetting scores. Default False.
    """
```

TrainerCallback hooks to implement:

on_train_begin(args, state, control, model, **kwargs):
  - Compute initial losses on all anchors (baseline snapshot)
  - Call anchor_set.snapshot_baseline(losses)
  - Print "Anchor baseline established: {n} anchors across {k} groups"

on_step_end(args, state, control, model, **kwargs):
  - If state.global_step % eval_every_n_steps == 0:
      * Compute current losses on all anchors
      * Call anchor_set.update_current_losses(losses)
      * Compute forgetting_scores()
      * If any group exceeds alert_threshold: create ForgettingAlert,
        append to self.alerts, print warning
      * If verbose: print current scores per group
      * If log_to_wandb: log to wandb if available

on_train_end(args, state, control, **kwargs):
  - Print final forgetting summary per group
  - Print "To generate full report: monitor.get_report()"

Public methods:
- alerts property -> list[ForgettingAlert]  (all fired alerts, chronological)
- get_report() -> ForgettingReport  (see below)
- forgetting_history() -> dict[str, list[tuple[int, float]]]
    Returns {tag: [(step, score), ...]} for plotting forgetting over time.

Computing anchor losses:
  Use model.eval(), torch.no_grad(), and a simple forward pass.
  For each anchor, tokenize if not already tokenized, run forward pass,
  compute cross-entropy loss with reduction="mean".
  Handle both causal LM (labels = input_ids shifted) and
  sequence classification (labels = scalar).
  Auto-detect model type from model.config.model_type or the presence of
  config.num_labels > 1. If uncertain, default to causal LM loss.
  Always restore model.train() after evaluation.

```python
@dataclass
class ForgettingReport:
    total_steps: int
    n_anchors: int
    n_alerts: int
    groups: dict[str, GroupReport]
    alerts: list[ForgettingAlert]

@dataclass
class GroupReport:
    tag: str
    n_anchors: int
    baseline_mean_loss: float
    final_mean_loss: float
    final_forgetting_score: float
    max_forgetting_score: float    # worst point during training
    most_forgotten_examples: list[AnchorExample]  # top 5
```

ForgettingReport methods:
- to_json(path: str) -> None
- to_html(path: str) -> None — plain Python string formatting, no Jinja2
  Must include: per-group forgetting scores, alert timeline, most-forgotten
  examples per group.
- plot(path: str) -> None — use matplotlib if available, skip gracefully if not

---

## Module 3: src/reservoir/replay_scheduler.py

Manages prioritized replay of forgotten anchors back into training.
Only active when ForgettingMonitor(auto_replay=True).

This is where PER is directly applied to catastrophic forgetting:
- Each anchor's priority = its forgetting severity (relative loss increase)
- FastPERBuffer samples proportionally to priority
- IS weights correct for the non-uniform sampling

```python
class ReplayScheduler:
    """
    Schedules prioritized replay of forgotten anchor examples.

    Uses FastPERBuffer to sample anchors proportional to forgetting severity,
    with IS weight correction to keep the gradient unbiased.

    Parameters
    ----------
    anchor_sets : list[AnchorSet]
    replay_ratio : float — fraction of each batch to replace. Default 0.1.
    alpha : float — PER exponent. Default 0.6.
    beta : float — IS correction exponent. Default 0.4.
    """
```

Internals:
- Maintains one FastPERBuffer per AnchorSet (or one combined)
- anchor priority in the PER buffer = anchor.priority from AnchorSet
- When auto_replay fires, call get_replay_batch(batch_size) which:
  1. Computes n_replay = max(1, int(batch_size * replay_ratio))
  2. Samples n_replay anchors from PER buffer proportional to priority
  3. Returns (anchor_examples, is_weights) where is_weights correct for PER bias

Public methods:
- update_priorities(anchor_set: AnchorSet) -> None
    Syncs PER buffer priorities from anchor_set.priorities
- get_replay_batch(batch_size: int) -> tuple[list[AnchorExample], list[float]]
    Returns anchors to replay and their IS weights.
- total_replayed property -> int — cumulative replay steps

Integration with ForgettingMonitor:
ForgettingMonitor creates a ReplayScheduler internally when auto_replay=True.
On each training step where replay fires, the monitor calls
replay_scheduler.get_replay_batch() and does an extra optimizer step
on the replay batch with IS weights applied.

The extra optimizer step (in on_step_end after the normal step):
  model.train()
  replay_examples, is_weights = self.replay_scheduler.get_replay_batch(batch_size)
  if replay_examples:
      inputs = self._collate_anchors(replay_examples)
      outputs = model(**inputs)
      per_example_loss = outputs.loss  # scalar, use reduction="mean"
      # Apply IS weights to de-bias
      weighted_loss = torch.tensor(is_weights).mean() * per_example_loss
      weighted_loss.backward()
      optimizer.step()
      optimizer.zero_grad()

Note: this is a simplified replay step. It uses the mean IS weight as a
scalar multiplier rather than per-example weighting, since most models
return a scalar loss. This is an approximation; document it in DECISIONS_B.md.

---

## Tests

All tests must run WITHOUT a GPU and WITHOUT actual model training.
Use small mock models (random weights, tiny vocab) where models are needed.

### tests/test_anchor_set.py
```python
def test_construction_from_list()
def test_construction_with_tags()
def test_len()
def test_getitem()
def test_groups_by_tag()
def test_snapshot_baseline_stores_losses()
def test_update_current_losses_computes_priority()
    # baseline=0.5, current=1.0 -> priority = (1.0-0.5)/0.5 = 1.0
def test_priority_zero_when_no_forgetting()
    # baseline=0.5, current=0.4 -> priority = 0 (clamped, not negative)
def test_forgetting_scores_by_group()
def test_most_forgotten_sorted_by_priority()
def test_from_dataset_classmethod()
def test_subsampling_with_n()
```

### tests/test_forgetting_monitor.py
```python
# Use unittest.mock for the HF Trainer hooks
# Create a tiny mock model that returns controllable losses

def test_monitor_initializes()
def test_on_train_begin_calls_baseline_snapshot(mock_model)
def test_no_alert_below_threshold(mock_model)
    # Set forgetting_score = 0.1, threshold = 0.5 -> no alert
def test_alert_fires_above_threshold(mock_model)
    # Set forgetting_score = 0.8, threshold = 0.5 -> alert fired
    # assert len(monitor.alerts) == 1
def test_alert_message_contains_tag()
def test_forgetting_history_records_per_step()
def test_get_report_returns_forgetting_report()
def test_verbose_false_suppresses_output(capsys)
def test_auto_replay_false_by_default()
def test_on_train_end_prints_summary(capsys, mock_model)
```

### tests/test_replay_scheduler.py
```python
def test_scheduler_initializes()
def test_update_priorities_syncs_with_anchor_set()
def test_get_replay_batch_returns_correct_count()
    # batch_size=32, ratio=0.1 -> n_replay=3
def test_is_weights_are_floats()
def test_high_priority_anchors_sampled_more()
    # set one anchor to 100x priority of others, sample 1000 times,
    # assert that anchor appears significantly more often
def test_total_replayed_increments()
def test_replay_batch_empty_when_no_anchors_with_priority()
```

---

## DECISIONS_B.md

Create /Users/joshuawu/Reservoir/DECISIONS_B.md immediately when you start.
Format every decision you make that isn't pre-specified above:

```markdown
# Plan B Implementation Decisions

## [Component name] — [Decision description]
**Context:** Why this decision was needed
**Options considered:** A / B / C
**Chosen:** A
**Reason:** One sentence
**Impact:** What this affects
```

Pre-record this known approximation:
```markdown
## ReplayScheduler — IS weight applied as scalar multiplier

**Context:** Most HF models return scalar loss, not per-example losses,
making per-example IS weighting impossible without extra forward passes.
**Options considered:**
  A) Mean IS weight as scalar multiplier (approximation, one forward pass)
  B) Per-example IS weights (accurate, requires custom loss reduction)
**Chosen:** A
**Reason:** Keeps the integration surface simple; per-example requires
  overriding model internals which creates model-specific complexity.
**Impact:** Replay gradient is approximately but not exactly IS-corrected.
  Document as a known limitation.
```

---

## pyproject.toml addition

Add to [project.optional-dependencies]:
```toml
anchor = [
    "transformers>=4.40",
    "datasets>=2.18",
    "accelerate>=0.27",
]
```

---

## Completion checklist

Work until every item is checked:

- [ ] DECISIONS_B.md created (with IS weight approximation pre-recorded)
- [ ] uv add / pip install for required packages completed
- [ ] src/reservoir/anchor_set.py written and importable
- [ ] src/reservoir/forgetting_monitor.py written and importable
- [ ] src/reservoir/replay_scheduler.py written and importable
- [ ] tests/test_anchor_set.py — all tests pass
- [ ] tests/test_forgetting_monitor.py — all tests pass (mocked, no GPU)
- [ ] tests/test_replay_scheduler.py — all tests pass
- [ ] All 270 original tests still pass
- [ ] src/reservoir/__init__.py updated with new exports
- [ ] pyproject.toml updated with [anchor] extra

Run this final verification before declaring done:
```bash
cd /Users/joshuawu/Reservoir
uv run pytest tests/ -v 2>&1 | tail -10
uv run python -c "
from reservoir import AnchorSet, ForgettingMonitor, ForgettingAlert, ReplayScheduler
from reservoir.forgetting_monitor import ForgettingReport, GroupReport
print('Plan B: all imports OK')
"
```

Both must succeed. If they don't, fix the errors and run again.
Do not stop until both succeed.
