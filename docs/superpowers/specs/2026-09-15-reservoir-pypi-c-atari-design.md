# Design: Reservoir PyPI Library + C Extension + Atari Benchmarks

**Date:** 2026-09-15
**Status:** Approved
**Scope:** Three sequential layers — C sum-tree extension, PyPI-publishable library, full 57-game Atari benchmark suite

---

## 1. Goals

1. Ship `reservoir` to PyPI with a clean install story (`pip install reservoir`)
2. Deliver a C-backed `CFastPERBuffer` that replaces the pure-Python sum-tree bottleneck while keeping the existing `FastPERBuffer` as an untouched fallback
3. Reproduce the Schaul et al. 2016 PER paper's 57-game Atari benchmark using CleanRL DQN

---

## 2. Non-Goals

- Rust/PyO3 extension (decided against; using Python C API)
- Pre-built binary wheels (post-launch follow-up; start with source distribution)
- Distributed or multi-actor replay buffers
- Modifications to the exact `ExactPERBuffer` or attestation system

---

## 3. Architecture

```
Layer 1: C Extension
  src/reservoir/csrc/sumtree.c        -- pure C sum-tree + min-tree structs
  src/reservoir/csrc/sumtreemodule.c  -- Python C API wrapper -> reservoir._sumtree

Layer 2: Buffer Implementations
  src/reservoir/fast_buffer.py        -- UNCHANGED, pure numpy/torch (fallback)
  src/reservoir/c_buffer.py           -- NEW: CFastPERBuffer using reservoir._sumtree

Layer 3: Package + Benchmarks
  src/reservoir/__init__.py           -- auto-selects C or Python buffer on import
  pyproject.toml + setup.py           -- setuptools build, optional [fast] extra
  benchmarks/                         -- CleanRL DQN + 57-game Atari runner
```

---

## 4. Layer 1: C Extension (`reservoir._sumtree`)

### 4.1 Files

| File | Purpose |
|------|---------|
| `src/reservoir/csrc/sumtree.c` | Pure C data structures, no Python headers |
| `src/reservoir/csrc/sumtreemodule.c` | Python C API bindings |

### 4.2 Exposed Python Classes

**`SumTree(capacity: int)`**

| Member | Signature | Notes |
|--------|-----------|-------|
| `update` | `(position: int, priority: int) -> None` | Sets leaf, propagates up |
| `prefix_sum_locate` | `(draw_int: int) -> int` | O(log N) tree walk |
| `get` | `(position: int) -> int` | Returns leaf value |
| `total` | property `-> int` | Root value |
| `capacity` | property `-> int` | Actual capacity (power of 2) |

**`MinTree(capacity: int)`**

| Member | Signature | Notes |
|--------|-----------|-------|
| `update` | `(position: int, priority: int) -> None` | Sets leaf, propagates up |
| `get` | `(position: int) -> int` | Returns leaf value |
| `minimum` | property `-> int` | Root value (min of all leaves) |
| `capacity` | property `-> int` | Actual capacity |

Empty MinTree leaves are initialized to `ULLONG_MAX` as sentinel.

### 4.3 Internal C types

```c
typedef struct {
    uint64_t *tree;   /* flat array, length = 2 * capacity */
    size_t    capacity;
} SumTree;

typedef struct {
    uint64_t *tree;
    size_t    capacity;
} MinTree;
```

Priority values are `uint64_t`. This matches the scaling used in `fast_buffer.py` (float32 priorities scaled to integers).

### 4.4 Correctness

The C sum-tree is validated against the existing pure-Python `ExactSumTree` in tests. Any discrepancy between the two is a test failure.

---

## 5. Layer 2: Buffer Implementations

### 5.1 `CFastPERBuffer` (`src/reservoir/c_buffer.py`)

Identical public API to `FastPERBuffer`. The only internal difference: `self._sum_tree` and `self._min_tree` are `reservoir._sumtree.SumTree` / `MinTree` instances instead of numpy-based Python objects.

**Public API (unchanged from `FastPERBuffer`):**

```python
class CFastPERBuffer:
    def __init__(self, capacity, obs_shape, action_dim=1,
                 alpha=0.6, beta=0.4, epsilon=1e-6, device="cpu")
    @classmethod
    def from_env(cls, env, capacity, alpha=0.6, beta=0.4, device="cpu")
    def insert(self, obs, action, reward, next_obs, done, td_error=None)
    def sample(self, batch_size) -> FastBatch
    def update_priorities(self, indices, td_errors)
    def anneal_beta(self, step, total_steps)
```

Returns the same `FastBatch` dataclass (torch tensors) as `FastPERBuffer`. All numpy array storage for transitions remains in Python.

### 5.2 `fast_buffer.py` — No Changes

This file is not touched. It remains the pure-Python/numpy fallback.

### 5.3 Auto-Selection (`src/reservoir/__init__.py`)

```python
try:
    from reservoir.c_buffer import CFastPERBuffer as FastPERBuffer
    _BACKEND = "c"
except ImportError:
    from reservoir.fast_buffer import FastPERBuffer
    _BACKEND = "python"

# Both always importable by explicit name
from reservoir.c_buffer import CFastPERBuffer      # may raise ImportError if no C ext
from reservoir.fast_buffer import FastPERBuffer as PyFastPERBuffer
from reservoir.buffer import ExactPERBuffer
```

`reservoir.backend` (`"c"` or `"python"`) lets users check which is active.

---

## 6. Layer 3: PyPI Packaging

### 6.1 Build System

Switch from `hatchling` to `setuptools` (hatchling does not support C extensions).

**`pyproject.toml` key changes:**

```toml
[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

[project.optional-dependencies]
fast  = []                                  # triggers C build; no extra pip deps
dev   = ["pytest>=7.0", "hypothesis>=6.0", "pytest-cov>=4.0"]
atari = ["ale-py>=0.9", "gymnasium[atari]>=1.0", "autorom[accept-rom-license]"]
```

Remove `fpdf2` from runtime deps (only needed for the plan PDF generator — move to dev).

**`setup.py`** (minimal):

```python
from setuptools import setup, Extension

setup(ext_modules=[
    Extension(
        "reservoir._sumtree",
        sources=[
            "src/reservoir/csrc/sumtreemodule.c",
            "src/reservoir/csrc/sumtree.c",
        ],
        extra_compile_args=["-O3", "-march=native"],
    )
])
```

### 6.2 Install Experience

```bash
pip install reservoir                    # pure Python, no compiler needed
pip install reservoir[fast]              # builds C extension, unlocks CFastPERBuffer
pip install reservoir[fast,atari]        # C extension + Atari ROM deps
pip install reservoir[fast,atari,dev]    # full development install
```

### 6.3 Version

Bump to `0.2.0` on release.

---

## 7. Atari Benchmarks

### 7.1 File Structure

```
benchmarks/
  atari_dqn.py          -- CleanRL DQN, --buffer-type flag
  run_atari.py          -- CLI: runs N games x M seeds, writes results/
  compare.py            -- loads results/, prints markdown table + saves plot
  configs/
    games.txt           -- all 57 Atari game IDs (NoFrameskip-v4 variants)
    hyperparams.yaml    -- Schaul et al. 2016 Table 1 values
  results/              -- one JSON file per (game, seed, buffer_type) run
```

### 7.2 Hyperparameters (Schaul et al. 2016)

| Parameter | Value |
|-----------|-------|
| alpha | 0.6 |
| beta start | 0.4 |
| beta end | 1.0 (annealed over training) |
| epsilon | 1e-6 |
| capacity | 1,000,000 |
| batch size | 32 |
| learning rate | 1e-4 |
| gamma | 0.99 |
| target update freq | 10,000 steps |
| total steps | 50,000,000 (reducible; paper used 200M) |

### 7.3 Buffer Types

| `--buffer-type` | Implementation |
|-----------------|----------------|
| `c` | `CFastPERBuffer` (C extension) |
| `python` | `PyFastPERBuffer` (pure numpy) |
| `uniform` | SB3 standard `ReplayBuffer` (baseline) |

### 7.4 Runner CLI

```bash
# Single game, one seed
uv run python -m benchmarks.run_atari --buffer-type c --games BreakoutNoFrameskip-v4 --seeds 1

# All 57 games
uv run python -m benchmarks.run_atari --buffer-type c --all-57 --seeds 1 2 3

# Compare results
uv run python -m benchmarks.compare --results-dir benchmarks/results/
```

The runner is **resumable**: it checks `results/` before starting each run and skips completed (game, seed, buffer_type) combinations.

### 7.5 Result Format

Each completed run writes `results/{game}_{buffer_type}_seed{n}.json`:

```json
{
  "game": "BreakoutNoFrameskip-v4",
  "buffer_type": "c",
  "seed": 1,
  "total_steps": 50000000,
  "episode_rewards": [...],
  "episode_steps": [...],
  "final_mean_reward_100ep": 312.4,
  "duration_seconds": 14400
}
```

### 7.6 Comparison Output

`compare.py` produces:
1. A markdown table: rows = games, columns = `uniform` / `python` / `c`, cell = mean reward over last 100 episodes
2. `results/comparison.png` — bar chart, one panel per game

---

## 8. Testing

| Test file | What it covers |
|-----------|---------------|
| `tests/test_c_sumtree.py` | C SumTree/MinTree: update, locate, total, minimum, invariants |
| `tests/test_c_buffer.py` | CFastPERBuffer: insert, sample, update_priorities, IS weights |
| `tests/test_c_vs_python.py` | Property-based: CFastPERBuffer and FastPERBuffer return identical results on same input |
| Existing tests | Unchanged; must all still pass |

---

## 9. Implementation Order

1. C extension (`csrc/` + `sumtreemodule.c`) + tests
2. `CFastPERBuffer` + tests + parity tests vs `FastPERBuffer`
3. `__init__.py` auto-selection + `pyproject.toml` / `setup.py` packaging
4. Smoke test: `pip install -e .[fast]`, run existing test suite
5. Atari benchmark infrastructure (`atari_dqn.py`, `run_atari.py`, `compare.py`)
6. Benchmark runs (iterative: start 2-3 games, then full 57)
7. PyPI publish (`twine upload` / `uv publish`)
