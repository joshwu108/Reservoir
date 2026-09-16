# Overnight Implementation Prompt — reservoir-prefcheck Week 1-2

Paste this entire file as your first message in a new Claude session with
bypass permissions enabled. The agent should work autonomously through
all tasks. Design choices are pre-decided — do not re-debate them.

---

## Your mission

You are implementing the first two weeks of Plan A (reservoir-prefcheck) for
the Reservoir project. This is a preference noise detection tool that wraps
HuggingFace TRL's RewardTrainer to surface mislabeled or inconsistent
preference pairs during reward model training.

Work autonomously. For any decision not covered below, take the simplest
reasonable option, record it in DECISIONS.md, and move on. Do not ask
clarifying questions. If something is blocked, record it and skip to the
next task.

---

## Project context

Location: /Users/joshuawu/Reservoir
Language: Python 3.13
Package manager: uv (run commands as `uv run pytest ...`, `uv run python ...`)
Build: setuptools with C extension (reservoir._sumtree already built)
Tests: `uv run pytest tests/ -v` — all 270 must still pass at the end

Key existing files:
- src/reservoir/fast_buffer.py — FastPERBuffer (numpy/torch, use this internally)
- src/reservoir/buffer.py — ExactPERBuffer (exact integer, reference only)
- src/reservoir/__init__.py — exports FastPERBuffer, ExactPERBuffer, backend
- pyproject.toml — setuptools build, optional [dev] and [atari] extras

DO NOT modify any existing files except:
- src/reservoir/__init__.py (add new exports at the bottom only)
- pyproject.toml (add trl as optional dependency under new [prefcheck] extra)

---

## What to build

### 1. src/reservoir/dataset_buffer.py

A prioritized sampler wrapping a HuggingFace Dataset.
Used to sample training examples proportional to their current loss.

```python
class DatasetBuffer:
    """Prioritized sampler for supervised/RLHF training datasets.

    Wraps a HuggingFace Dataset and samples examples proportional to
    their current loss. Compatible with PyTorch DataLoader.

    Parameters
    ----------
    dataset : datasets.Dataset or list
        The training dataset. Examples are identified by integer index.
    alpha : float
        PER exponent (0 = uniform, 1 = fully prioritized). Default 0.6.
    beta : float
        IS correction exponent. Default 0.4.
    epsilon : float
        Minimum priority to prevent zero probabilities. Default 1e-6.
    mode : str
        "audit": uniform sampling, priorities tracked passively.
        "accelerated": prioritized sampling active. Default "audit".
    priority_cap : float or None
        In accelerated mode, cap priority at this multiple of median to
        prevent never-converging examples from dominating. Default 10.0.
    """
```

Methods:
- `__len__() -> int`
- `__getitem__(idx) -> dict` — returns dataset[idx] plus {"__index__": idx}
- `update_priority(idx: int, loss: float) -> None`
- `sample_indices(batch_size: int) -> list[int]` — returns indices to sample
- `get_is_weights(indices: list[int]) -> torch.Tensor` — IS correction weights
- `priorities` property -> numpy array of current priorities
- `size` property -> int

Internally uses FastPERBuffer with obs_shape=(1,), action_dim=1.
The "observation" stored is just the index — actual data is in the Dataset.

In audit mode, sample_indices returns random uniform indices.
In accelerated mode, sample_indices uses the PER tree.
Priority tracking (update_priority) works in both modes.

### 2. src/reservoir/trajectory.py

Records per-example loss history and extracts trajectory features.

```python
class TrajectoryLogger:
    """Records per-example loss trajectory across training steps.

    Parameters
    ----------
    n_examples : int
        Number of examples in the dataset.
    window_frac : float
        Fraction of total steps to use for mean_loss_last_k. Default 0.2.
    warn_threshold : int
        Warn if n_examples exceeds this (memory concern). Default 100_000.
    """
```

Methods:
- `log(example_idx: int, step: int, loss: float) -> None`
- `finalize(total_steps: int) -> None` — compute features after training ends
- `get_features(example_idx: int) -> TrajectoryFeatures`
- `get_all_features() -> dict[int, TrajectoryFeatures]`

```python
@dataclass
class TrajectoryFeatures:
    example_idx: int
    n_observations: int          # how many times this example was seen
    mean_loss_last_k: float      # mean loss over last window_frac of steps
    slope: float                 # linear regression slope over all observations
    variance: float              # variance of loss values
    first_correct_step: int | None  # first step where loss < 0.5 (i.e. correct)
    loss_history: list[float]    # full history (kept for inspection)
```

The `first_correct_step` threshold of 0.5 is the binary cross-entropy
midpoint for a reward model (loss < 0.5 means model predicts correct preference).
Expose as a constructor parameter `correct_threshold=0.5`.

### 3. src/reservoir/report.py

Buckets examples into FLIPPED / AMBIGUOUS / CLEAN and exports results.

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
    confidence: float        # 0-1, higher = more confident in the label
    features: TrajectoryFeatures

class PreferenceQualityReport:
    """Buckets training examples by noise type based on trajectory features."""
```

Bucketing rules (pre-decided, implement exactly as specified):
- FLIPPED: slope > 0.01 AND mean_loss_last_k > median(mean_loss_last_k across all)
- AMBIGUOUS: variance > 75th percentile(variance) AND -0.01 <= slope <= 0.01
- CLEAN: everything else

Confidence scoring:
- FLIPPED confidence = clip((slope - 0.01) / 0.1, 0, 1)
- AMBIGUOUS confidence = clip((variance - p75_variance) / p75_variance, 0, 1)
- CLEAN confidence = 1 - max(flipped_confidence, ambiguous_confidence)

Methods:
- `flipped` property -> list[ExampleReport] sorted by confidence desc
- `ambiguous` property -> list[ExampleReport] sorted by confidence desc
- `clean` property -> list[ExampleReport]
- `summary() -> dict` — counts and percentages per bucket
- `to_json(path: str) -> None`
- `to_csv(path: str) -> None`
- `to_html(path: str) -> None` — plain Python string formatting, no Jinja2

HTML report must include:
- Summary table (bucket counts, percentages)
- Top 50 FLIPPED examples with their features
- Top 50 AMBIGUOUS examples with their features
- The command used to generate the report

### 4. src/reservoir/prefcheck.py

The main user-facing class. Wraps TRL RewardTrainer.

```python
class PreferenceNoiseDetector:
    """Detects noisy preference pairs during reward model training.

    Parameters
    ----------
    model : PreTrainedModel
        The reward model to train.
    tokenizer : PreTrainedTokenizer
    train_dataset : Dataset
        HuggingFace Dataset with "chosen" and "rejected" columns.
    mode : str
        "audit" (default) or "accelerated". See DatasetBuffer.
    alpha : float
        PER exponent for accelerated mode. Default 0.6.
    trajectory_window : float
        Window fraction for TrajectoryLogger. Default 0.2.
    training_args : RewardConfig or None
        TRL training args. If None, uses sensible defaults.
    correct_threshold : float
        Loss below this = model predicts correct preference. Default 0.5.
    """

    def train(self) -> None:
        """Run reward model training with trajectory logging."""

    def get_report(self) -> PreferenceQualityReport:
        """Return the noise detection report. Call after train()."""
```

Implementation approach:
1. Build a DatasetBuffer wrapping train_dataset
2. Create a custom RewardTrainer subclass that:
   a. Overrides get_train_dataloader() to use DatasetBuffer in audit mode
      (return a DataLoader with a custom sampler that uses DatasetBuffer)
   b. Adds a TrajectoryLogger as an instance variable
   c. Overrides compute_loss() to additionally call
      trajectory_logger.log(example_idx, step, loss) for each example
   d. Alternatively: use a HF TrainerCallback with on_log() hook

The callback approach is preferred if compute_loss() override is complex.
Use whichever is simpler to implement correctly.

TRL version: require trl>=0.11,<0.14
Add at top of prefcheck.py:
```python
try:
    import trl
    from trl import RewardTrainer, RewardConfig
    _TRL_VERSION = trl.__version__
except ImportError:
    raise ImportError(
        "reservoir-prefcheck requires trl>=0.11. "
        "Install with: pip install 'reservoir[prefcheck]'"
    )
```

---

## pyproject.toml addition

Add to [project.optional-dependencies]:
```toml
prefcheck = [
    "trl>=0.11,<0.14",
    "transformers>=4.40",
    "datasets>=2.18",
    "accelerate>=0.27",
]
```

---

## Tests to write (all must pass without GPU)

### tests/test_dataset_buffer.py
- test_audit_mode_samples_uniformly: uniform sampling in audit mode
- test_accelerated_mode_uses_priorities: high-priority examples sampled more
- test_update_priority_changes_distribution: after update, distribution shifts
- test_is_weights_normalize: IS weights sum to approximately batch_size
- test_getitem_includes_index: __getitem__ adds __index__ key
- test_large_dataset_warns: warn if n > 100k

### tests/test_trajectory.py
- test_log_records_history: logged steps appear in history
- test_finalize_computes_slope: positive slope for rising loss
- test_finalize_computes_variance: variance of constant sequence is 0
- test_finalize_computes_mean_last_k: only last window_frac used
- test_first_correct_step: correct step found when loss drops below threshold
- test_first_correct_step_none_if_never: returns None if never correct
- test_warn_large_dataset: warning raised above threshold

### tests/test_report.py
- test_flipped_bucket: high slope + high loss → FLIPPED
- test_ambiguous_bucket: high variance + near-zero slope → AMBIGUOUS
- test_clean_bucket: low slope + low loss → CLEAN
- test_sorted_by_confidence: flipped list sorted descending
- test_to_json_roundtrip: JSON output is valid JSON
- test_to_csv_has_header: CSV has expected columns
- test_to_html_produces_html: output contains <html> tag
- test_summary_counts: summary percentages sum to 100

### tests/test_prefcheck.py

Mock TRL. Use unittest.mock to avoid requiring trl installed in test env.
Test that:
- PreferenceNoiseDetector initializes without error (mock RewardTrainer)
- train() calls the underlying trainer
- get_report() raises RuntimeError if called before train()
- get_report() returns PreferenceQualityReport after train()

If trl is not installed, skip these tests with pytest.importorskip.

---

## DECISIONS.md

Create /Users/joshuawu/Reservoir/DECISIONS.md to record any choices made
during implementation that were not pre-specified. Format:

```markdown
# Implementation Decisions

## [Component] — [Decision made]
**Options considered:** A, B
**Chosen:** A
**Reason:** [one sentence]
**Date:** [date]
```

---

## Completion criteria

The session is done when:
1. All four new modules exist and are importable
2. All new tests pass
3. All 270 existing tests still pass: `uv run pytest tests/ -v`
4. DECISIONS.md exists with any non-specified choices recorded
5. src/reservoir/__init__.py exports DatasetBuffer and PreferenceNoiseDetector

Run this final check before finishing:
```bash
uv run pytest tests/ -v 2>&1 | tail -5
uv run python -c "
from reservoir import DatasetBuffer, PreferenceNoiseDetector
from reservoir.trajectory import TrajectoryLogger, TrajectoryFeatures
from reservoir.report import PreferenceQualityReport, NoiseLabel
print('All imports OK')
"
```

---

## What NOT to do

- Do not modify fast_buffer.py, c_buffer.py, buffer.py, attest.py, durable.py
- Do not rewrite or reorganize existing tests
- Do not require a GPU or real TRL training to run tests
- Do not add Jinja2 as a dependency
- Do not implement Plan B (reservoir-anchor) — that comes after Plan A ships
- Do not touch the Atari benchmark infrastructure
- Do not commit anything — the user will review and commit
