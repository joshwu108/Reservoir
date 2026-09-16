# Autonomous Session — Plan A: reservoir-prefcheck
# Preference Noise Detector for RLHF Reward Models

You are an autonomous agent. Work continuously until every task is complete.
Do not stop to ask questions. If you hit a decision not covered below, make
the most reasonable choice, record it in DECISIONS_A.md, and continue.
You have permission to install any packages needed using pip or uv.
Do NOT make any git commits — the user will review and commit manually.
Do not stop until all code is written, all tests pass, and the completion
checklist at the bottom is fully checked off.

---

## What you are building

A Python module called reservoir-prefcheck that wraps HuggingFace TRL's
RewardTrainer to detect mislabeled and inconsistent preference pairs during
reward model training — before they poison the RLHF pipeline.

The key insight: when a preference pair is mislabeled (A chosen when B is
actually better), the reward model's loss on that pair stays high or increases
over training because the model keeps predicting the correct answer and getting
penalized. By tracking loss trajectories per example, you can separate:
- FLIPPED: loss flat or rising (label is wrong — re-annotate or remove)
- AMBIGUOUS: loss oscillates (genuine annotator disagreement — get a second opinion)
- CLEAN: loss decays normally (fine)

This uses PER's core mechanism: each example's loss = its priority. The
trajectory shape over training distinguishes noise types better than any
single-point loss score.

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
  uv add --optional prefcheck "trl>=0.11" "transformers>=4.40" "datasets>=2.18" "accelerate>=0.27"
  uv run pip install trl transformers datasets accelerate

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
  src/reservoir/dataset_buffer.py
  src/reservoir/trajectory.py
  src/reservoir/prefcheck.py
  src/reservoir/report.py
  tests/test_dataset_buffer.py
  tests/test_trajectory.py
  tests/test_prefcheck.py
  tests/test_report.py
  DECISIONS_A.md
  benchmarks/prefcheck/synthetic_noise.py  (if time permits)

DO append (do not replace) to:
  src/reservoir/__init__.py  — add these lines at the very bottom only:
    from reservoir.dataset_buffer import DatasetBuffer
    from reservoir.prefcheck import PreferenceNoiseDetector
    from reservoir.report import PreferenceQualityReport, NoiseLabel

---

## Module 1: src/reservoir/dataset_buffer.py

A prioritized sampler wrapping a HuggingFace Dataset (or any list/dict dataset).
Used to track per-example loss and optionally sample proportional to it.

```python
class DatasetBuffer:
    """
    Prioritized sampler for supervised/RLHF training datasets.

    Tracks per-example loss across training and optionally samples
    proportional to loss (accelerated mode) or uniformly (audit mode).

    Parameters
    ----------
    dataset : any indexable dataset (HuggingFace Dataset, list, etc.)
    alpha : float — PER exponent. Default 0.6.
    beta : float — IS correction exponent. Default 0.4.
    epsilon : float — minimum priority. Default 1e-6.
    mode : "audit" | "accelerated"
        audit: uniform sampling, priorities tracked passively (default, safe)
        accelerated: prioritized sampling active
    priority_cap : float — in accelerated mode, cap priority at this multiple
        of current median to prevent never-converging examples dominating.
        Default 10.0.
    """
```

Internals: uses FastPERBuffer with obs_shape=(1,), action_dim=1 for the
priority tree. The "observation" is just a placeholder — actual data comes
from the dataset by index.

Public methods:
- __len__() -> int
- __getitem__(idx: int) -> dict  — returns dataset[idx] with "__index__": idx added
- update_priority(idx: int, loss: float) -> None  — update this example's priority
- sample_indices(batch_size: int) -> list[int]
    audit mode: returns random.sample(range(len(dataset)), batch_size)
    accelerated mode: uses PER tree, respects priority_cap
- get_is_weights(indices: list[int]) -> list[float]
    audit mode: returns [1.0] * batch_size (uniform = no correction needed)
    accelerated mode: IS weights from PER tree, normalized to [0, 1]
- priorities property -> numpy array of current priorities for all examples
- size property -> int (number of examples)

Priority cap implementation (accelerated mode only):
  cap = priority_cap * median(self.priorities[self.priorities > 0])
  capped_priority = min(raw_loss + epsilon, cap)

Warn via warnings.warn if len(dataset) > 100_000:
  "DatasetBuffer: tracking priorities for {n} examples in memory.
   Consider chunking for very large datasets."

---

## Module 2: src/reservoir/trajectory.py

Records per-example loss history and extracts features for bucketing.

```python
@dataclass
class TrajectoryFeatures:
    example_idx: int
    n_observations: int           # times this example appeared in training
    mean_loss_last_k: float       # mean loss over last window_frac * total_steps
    slope: float                  # linear regression slope (positive = getting worse)
    variance: float               # variance of loss values
    first_correct_step: int | None  # first step where loss < correct_threshold
    loss_history: list[float]     # full loss history, kept for inspection

class TrajectoryLogger:
    """
    Records per-example loss trajectory across training steps.

    Parameters
    ----------
    n_examples : int — number of examples in dataset
    window_frac : float — fraction of total_steps for mean_loss_last_k. Default 0.2.
    correct_threshold : float — loss below this = model correct. Default 0.5.
    warn_threshold : int — warn if n_examples exceeds this. Default 100_000.
    """
```

Public methods:
- log(example_idx: int, step: int, loss: float) -> None
    Records (step, loss) for this example. Appends to internal list.
- finalize(total_steps: int) -> None
    Call after training ends. Computes all TrajectoryFeatures for every
    example that was logged at least once.
- get_features(example_idx: int) -> TrajectoryFeatures | None
    Returns None if this example was never logged.
- get_all_features() -> dict[int, TrajectoryFeatures]
    Returns all computed features. Call after finalize().
- summary_stats() -> dict
    Returns {"n_logged": int, "n_never_seen": int, "mean_observations": float}

Slope computation: use numpy.polyfit(steps, losses, deg=1)[0] where steps
are the actual step numbers (not just indices). If n_observations < 2,
slope = 0.0.

Variance computation: numpy.var(loss_history). If n_observations < 2,
variance = 0.0.

mean_loss_last_k: compute the window cutoff as
  cutoff_step = total_steps * (1 - window_frac)
  use only observations where step >= cutoff_step
  if no observations in window, use the last available observation

first_correct_step: the step number of the first observation where
loss < correct_threshold. None if never.

---

## Module 3: src/reservoir/report.py

Buckets examples by trajectory shape and exports results.

```python
from enum import Enum

class NoiseLabel(Enum):
    FLIPPED = "flipped"
    AMBIGUOUS = "ambiguous"
    CLEAN = "clean"

@dataclass
class ExampleReport:
    example_idx: int
    label: NoiseLabel
    confidence: float          # 0.0 to 1.0, higher = more confident
    features: TrajectoryFeatures

class PreferenceQualityReport:
    """
    Buckets training examples into noise categories based on trajectory.

    Parameters
    ----------
    features : dict[int, TrajectoryFeatures] — from TrajectoryLogger.get_all_features()
    """
```

Bucketing rules — implement EXACTLY as specified, do not change:
  FLIPPED:   slope > 0.01  AND  mean_loss_last_k > median(all mean_loss_last_k)
  AMBIGUOUS: variance > percentile_75(all variances)  AND  -0.01 <= slope <= 0.01
  CLEAN:     everything else

Confidence scoring:
  FLIPPED confidence   = clip((slope - 0.01) / 0.10, 0.0, 1.0)
  AMBIGUOUS confidence = clip((variance - p75_var) / p75_var, 0.0, 1.0)
         where p75_var = 75th percentile of variance across all examples
  CLEAN confidence     = 1.0 - max(flipped_confidence, ambiguous_confidence)

Public properties:
- flipped  -> list[ExampleReport] sorted by confidence descending
- ambiguous -> list[ExampleReport] sorted by confidence descending
- clean    -> list[ExampleReport]

Public methods:
- summary() -> dict with keys:
    n_total, n_flipped, n_ambiguous, n_clean,
    pct_flipped, pct_ambiguous, pct_clean
- to_json(path: str) -> None — writes JSON, integers stored as strings for precision
- to_csv(path: str) -> None — one row per example, columns: idx, label, confidence,
    slope, variance, mean_loss_last_k, n_observations, first_correct_step
- to_html(path: str) -> None — plain Python string formatting ONLY, no Jinja2,
    must include:
    * summary table (bucket counts and percentages)
    * top-50 FLIPPED examples table with all features
    * top-50 AMBIGUOUS examples table with all features
    * timestamp and dataset size at bottom

---

## Module 4: src/reservoir/prefcheck.py

Main user-facing class. Wraps TRL RewardTrainer.

```python
class PreferenceNoiseDetector:
    """
    Wraps TRL's RewardTrainer to detect noisy preference pairs.

    Parameters
    ----------
    model : PreTrainedModel
    tokenizer : PreTrainedTokenizer
    train_dataset : Dataset with "chosen" and "rejected" fields
    mode : "audit" | "accelerated" — default "audit"
    alpha : float — PER exponent for accelerated mode. Default 0.6.
    trajectory_window : float — window fraction for TrajectoryLogger. Default 0.2.
    correct_threshold : float — Default 0.5.
    training_args : RewardConfig | None — if None, use defaults:
        num_train_epochs=3, per_device_train_batch_size=8,
        learning_rate=2e-5, output_dir="./rm_output", report_to="none"
    """

    def train(self) -> None:
        """Run reward model training with trajectory logging."""

    def get_report(self) -> PreferenceQualityReport:
        """Return noise report. Raises RuntimeError if called before train()."""
```

Implementation approach — use a HuggingFace TrainerCallback:

Create an inner class _TrajectoryCallback(TrainerCallback) that:
- Stores a reference to the TrajectoryLogger and DatasetBuffer
- on_step_end: if the trainer just computed loss on a batch, log each
  example's loss via trajectory_logger.log(example_idx, step, loss)
- To get per-example loss during training, override compute_loss in a
  RewardTrainer subclass to additionally call the callback

The cleanest approach for per-example loss tracking:
1. Subclass RewardTrainer as _InstrumentedRewardTrainer
2. Override compute_loss to compute per-example loss (use reduction="none",
   then mean for the returned loss but log individual losses)
3. Get example indices from the batch via batch["__index__"] — this requires
   the DatasetBuffer.__getitem__ adding "__index__" to each item

For audit mode the DataLoader is standard (no PER sampling).
For accelerated mode, pass a custom sampler to the DataLoader.

TRL version guard at top of file:
```python
try:
    import trl
    from trl import RewardTrainer, RewardConfig
except ImportError:
    raise ImportError(
        "reservoir-prefcheck requires trl>=0.11.\n"
        "Install: pip install 'trl>=0.11'"
    )
```

---

## Tests

All tests must run WITHOUT a GPU and WITHOUT actual TRL training.
Mock trl if not installed using pytest.importorskip.

### tests/test_dataset_buffer.py
```python
# Required tests:
def test_audit_mode_len()
def test_audit_mode_getitem_has_index()
def test_audit_mode_update_priority_stores_value()
def test_audit_mode_sample_indices_returns_correct_count()
def test_audit_mode_is_weights_all_ones()
def test_accelerated_mode_high_priority_sampled_more()
    # Set priority[0] = 1000x others, sample 1000 times,
    # assert index 0 appears significantly more often
def test_priority_cap_limits_max_priority()
def test_priorities_property_length()
def test_large_dataset_warns()
    # Use warnings.catch_warnings to assert warning fired for n > 100_000
```

### tests/test_trajectory.py
```python
def test_log_records_entry()
def test_finalize_computes_slope_positive()
    # log increasing losses, assert slope > 0
def test_finalize_computes_slope_negative()
    # log decreasing losses, assert slope < 0
def test_finalize_variance_zero_for_constant()
    # log same loss every step, assert variance == 0
def test_finalize_mean_last_k_uses_window()
    # log 10 steps, window=0.5, assert mean uses only last 5
def test_first_correct_step_found()
    # log loss [0.8, 0.8, 0.3], threshold=0.5, assert first_correct == 2
def test_first_correct_step_none_when_never_correct()
def test_get_features_returns_none_for_unseen()
def test_finalize_required_before_get_all()
    # assert calling get_all_features before finalize raises RuntimeError
def test_n_observations_counts_correctly()
```

### tests/test_report.py
```python
def _make_features(idx, slope, variance, mean_loss, n_obs=10):
    # Helper to build TrajectoryFeatures for testing

def test_flipped_bucket_assigned_correctly()
    # high slope + high mean_loss -> FLIPPED
def test_ambiguous_bucket_assigned_correctly()
    # high variance + near-zero slope -> AMBIGUOUS
def test_clean_bucket_assigned_correctly()
    # low slope + low mean_loss -> CLEAN
def test_flipped_sorted_by_confidence_desc()
def test_summary_percentages_sum_to_100()
def test_to_json_is_valid_json()
def test_to_csv_has_correct_columns()
def test_to_html_contains_html_tag()
def test_to_html_contains_summary()
def test_empty_report_handles_gracefully()
    # no examples logged -> all buckets empty, summary shows 0
```

### tests/test_prefcheck.py
```python
# Mock TRL to avoid GPU requirement
import unittest.mock as mock

def test_prefcheck_raises_if_get_report_before_train()
def test_prefcheck_initializes_in_audit_mode()
def test_prefcheck_initializes_in_accelerated_mode()
def test_prefcheck_train_calls_trainer(mock_trainer)
    # mock the TRL trainer, assert train() was called
def test_prefcheck_get_report_returns_quality_report(mock_trainer)
    # after mocked train(), get_report() returns PreferenceQualityReport
```

---

## DECISIONS_A.md

Create /Users/joshuawu/Reservoir/DECISIONS_A.md immediately when you start.
Format every decision you make that isn't pre-specified above:

```markdown
# Plan A Implementation Decisions

## [Component name] — [Decision description]
**Context:** Why this decision was needed
**Options considered:** A / B / C
**Chosen:** A
**Reason:** One sentence
**Impact:** What this affects
```

---

## pyproject.toml addition

Add to [project.optional-dependencies]:
```toml
prefcheck = [
    "trl>=0.11",
    "transformers>=4.40",
    "datasets>=2.18",
    "accelerate>=0.27",
]
```

---

## Completion checklist

Work until every item is checked:

- [ ] DECISIONS_A.md created (even if empty at start)
- [ ] uv add / pip install for required packages completed
- [ ] src/reservoir/dataset_buffer.py written and importable
- [ ] src/reservoir/trajectory.py written and importable
- [ ] src/reservoir/report.py written and importable
- [ ] src/reservoir/prefcheck.py written and importable
- [ ] tests/test_dataset_buffer.py — all tests pass
- [ ] tests/test_trajectory.py — all tests pass
- [ ] tests/test_report.py — all tests pass
- [ ] tests/test_prefcheck.py — all tests pass (mocked, no GPU)
- [ ] All 270 original tests still pass
- [ ] src/reservoir/__init__.py updated with new exports
- [ ] pyproject.toml updated with [prefcheck] extra

Run this final verification before declaring done:
```bash
cd /Users/joshuawu/Reservoir
uv run pytest tests/ -v 2>&1 | tail -10
uv run python -c "
from reservoir import DatasetBuffer, PreferenceNoiseDetector, PreferenceQualityReport, NoiseLabel
from reservoir.trajectory import TrajectoryLogger, TrajectoryFeatures
print('Plan A: all imports OK')
"
```

Both must succeed. If they don't, fix the errors and run again.
Do not stop until both succeed.
