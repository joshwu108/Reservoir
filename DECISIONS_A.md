# Plan A Implementation Decisions

## DatasetBuffer — Priority storage strategy
**Context:** DatasetBuffer needs per-example priorities for a static dataset (not a circular RL buffer). FastPERBuffer is a circular buffer, so using it directly requires pre-populating it with n dummy entries before actual use.
**Options considered:** A) Use FastPERBuffer with pre-populated dummy entries / B) Implement a standalone priority tree / C) Track priorities with a numpy array only
**Chosen:** A
**Reason:** Spec explicitly says "Internals: uses FastPERBuffer with obs_shape=(1,), action_dim=1 for the priority tree."
**Impact:** DatasetBuffer must pre-populate n dummy transitions on __init__, so memory scales with dataset size.

## DatasetBuffer — IS weight computation
**Context:** sample_indices() and get_is_weights() are separate methods. FastPERBuffer.sample() computes both in one call. Calling sample() twice gives different random indices.
**Options considered:** A) Compute IS weights directly from tree internals / B) Cache last sample() call / C) Implement separate IS weight formula
**Chosen:** A — read directly from _tree leaf nodes and _min_tree for the formula (n * p_i / total)^(-beta) / max_weight
**Reason:** Avoids coupling sample_indices() and get_is_weights() and prevents double-sampling.
**Impact:** DatasetBuffer accesses FastPERBuffer internals (_tree, _min_tree, _tree_capacity).

## DatasetBuffer — Raw priority tracking
**Context:** The priority_cap formula requires median(self.priorities[self.priorities > 0]). Need raw priorities (before alpha exponentiation).
**Options considered:** A) Keep a separate _raw_priorities numpy array / B) Reverse-engineer from tree (tree stores p^alpha, undo via ^(1/alpha))
**Chosen:** A
**Reason:** Simpler, avoids floating-point error from inverse exponentiation.
**Impact:** Both _raw_priorities and the PER tree are kept in sync on every update_priority() call.

## TrajectoryLogger — finalize guard
**Context:** Spec says get_all_features() before finalize() raises RuntimeError.
**Options considered:** A) Use a flag _finalized / B) Return None / C) Raise always if called before finalize
**Chosen:** A — set _finalized = False, flip after finalize(), raise RuntimeError if not finalized in get_all_features()
**Reason:** Clean API boundary; calling before finalize is a programmer error.
**Impact:** Tests must call finalize() before get_all_features().

## PreferenceNoiseDetector — per-example loss tracking
**Context:** TRL RewardTrainer.compute_loss() returns a scalar; need per-example losses.
**Options considered:** A) Subclass RewardTrainer and override compute_loss with reduction="none" / B) Use a TrainerCallback on_step_end / C) Monkey-patch
**Chosen:** A — _InstrumentedRewardTrainer subclasses RewardTrainer, overrides compute_loss to log per-example losses then returns the mean.
**Reason:** Spec explicitly requests this approach.
**Impact:** _InstrumentedRewardTrainer stores a reference to the TrajectoryLogger and DatasetBuffer.

## prefcheck.py — TRL version compatibility
**Context:** TRL API may differ across versions. RewardTrainer was renamed/changed in some releases.
**Options considered:** A) Import RewardConfig from trl / B) Use TrainingArguments fallback / C) Check trl version explicitly
**Chosen:** A — import RewardConfig from trl with fallback to TrainingArguments if RewardConfig not found
**Reason:** TRL>=0.11 has RewardConfig; older versions used TrainingArguments. This gracefully handles both.
**Impact:** prefcheck.py has a try/except for both import paths.

## test_prefcheck.py — Mocking TRL
**Context:** Tests must run without GPU and without actual TRL training.
**Options considered:** A) Mock trl entirely with unittest.mock / B) Use pytest.importorskip / C) Conditional import with skip markers
**Chosen:** A — use unittest.mock.patch to replace RewardTrainer and RewardConfig; tests run with mocked trainer
**Reason:** More control over behavior; tests can assert call counts and return values.
**Impact:** Tests use @pytest.fixture with mock_trainer as a context manager.
