# Design: Reservoir PyPI Library + C Extension + Atari Benchmarks

**Date:** 2026-09-15
**Status:** Approved (rev 3 — all spec-review issues resolved)
**Scope:** Three sequential layers — C sum-tree extension, PyPI-publishable library, full 57-game Atari benchmark suite

---

## 1. Goals

1. Ship `reservoir` to PyPI with a clean install story (`pip install reservoir`)
2. Deliver a C-backed `CFastPERBuffer` that replaces the pure-Python sum-tree bottleneck while keeping the existing `FastPERBuffer` as an untouched fallback
3. Reproduce the Schaul et al. 2016 PER paper's 57-game Atari benchmark using CleanRL DQN

---

## 2. Non-Goals

- Rust/PyO3 extension
- Pre-built binary wheels (post-launch follow-up; start with source distribution only)
- Distributed or multi-actor replay buffers
- Modifications to `ExactPERBuffer`, `ExactSumTree`, or the attestation system
- Exact-integer parity with `ExactPERBuffer` (C tree backs `FastPERBuffer`, which uses float64)

---

## 3. Architecture

```
Layer 1: C Extension
  src/reservoir/csrc/sumtree.c        -- pure C sum-tree + min-tree structs (no Python headers)
  src/reservoir/csrc/sumtreemodule.c  -- Python C API wrapper -> reservoir._sumtree

Layer 2: Buffer Implementations
  src/reservoir/fast_buffer.py        -- UNCHANGED, pure numpy/torch (fallback)
  src/reservoir/c_buffer.py           -- NEW: CFastPERBuffer using reservoir._sumtree

Layer 3: Package + Benchmarks
  src/reservoir/__init__.py           -- updated: auto-selects C or Python buffer, exports public API
  pyproject.toml + setup.py           -- setuptools build, C extension always attempted
  MANIFEST.in                         -- ensures C source files included in sdist
  benchmarks/                         -- adapted CleanRL DQN + 57-game Atari runner
```

---

## 4. Layer 1: C Extension (`reservoir._sumtree`)

### 4.1 Priority Type: `double` (float64)

The C extension backs `CFastPERBuffer`, which is the production equivalent of `FastPERBuffer`. `FastPERBuffer` stores priorities as **float64** — the tree arrays are `np.zeros(..., dtype=np.float64)` and `np.full(..., np.inf, dtype=np.float64)`. Therefore the C tree stores `double`.

**What the tree stores:** priorities already raised to alpha (`p ** alpha`), exactly as `FastPERBuffer._tree_update` does. The C `update()` method receives the pre-exponentiated value, not the raw priority.

### 4.2 Tree Size: Power of Two

`FastPERBuffer` computes `_tree_capacity` as the smallest power of 2 ≥ `capacity` and allocates `2 * _tree_capacity` elements. The C struct must mirror this:

```c
static Py_ssize_t next_power_of_two(Py_ssize_t n) {
    Py_ssize_t p = 1;
    while (p < n) p <<= 1;
    return p;
}
/* tree array size = 2 * next_power_of_two(capacity) */
```

### 4.3 Files

| File | Purpose |
|------|---------|
| `src/reservoir/csrc/sumtree.c` | Pure C data structures, no Python headers. No `PyInit_*`. |
| `src/reservoir/csrc/sumtreemodule.c` | Python C API bindings. Defines `PyInit__sumtree`. |

Both files listed in `Extension.sources`. `sumtree.c` exports only C-internal symbols.

### 4.4 Exposed Python Classes

**`SumTree(capacity: int)`**

| Member | Signature | Notes |
|--------|-----------|-------|
| `update` | `(position: int, value: float) -> None` | Sets leaf (pre-exponentiated), propagates up O(log N) |
| `sample_batch` | `(values: Sequence[float]) -> list[int]` | Batch traversal via C loop; accepts any sequence of floats, returns list of int positions |
| `get` | `(position: int) -> float` | Returns leaf value |
| `total` | property `-> float` | Root value (sum of all leaves) |
| `tree_capacity` | property `-> int` | Internal capacity (next power of 2) |

**`MinTree(capacity: int)`**

| Member | Signature | Notes |
|--------|-----------|-------|
| `update` | `(position: int, value: float) -> None` | Sets leaf (same value as SumTree.update), propagates up |
| `get` | `(position: int) -> float` | Returns leaf value |
| `minimum` | property `-> float` | Root value (min of all leaves) |
| `tree_capacity` | property `-> int` | Internal capacity |

**Key design note on `sample_batch`:** `FastPERBuffer._tree_sample_batch` does a **vectorized** tree walk across all batch values at once. The C `sample_batch` accepts a numpy `ndarray` of float64 draw values and returns a numpy `ndarray` of int64 positions — this is where the C extension provides its main speedup over Python.

**Empty MinTree slots:** sentinel is `INFINITY` (`1.0 / 0.0` in C, i.e., IEEE 754 `+inf`), matching `fast_buffer.py`'s `np.full(..., np.inf)`. `min(inf, x) = x` for all finite x, and an empty-buffer `minimum` returns `inf`.

### 4.5 Internal C structs

```c
typedef struct {
    double    *tree;         /* length = 2 * tree_capacity */
    Py_ssize_t tree_capacity; /* always a power of 2 */
    Py_ssize_t capacity;     /* user-requested capacity */
} SumTree;

typedef struct {
    double    *tree;         /* length = 2 * tree_capacity; init to INFINITY */
    Py_ssize_t tree_capacity;
    Py_ssize_t capacity;
} MinTree;
```

Memory allocated with `PyMem_Malloc` in `tp_new`, freed in `tp_dealloc`.

### 4.6 Error Handling

- `update`: `IndexError` if position out of range; `ValueError` if value is negative, NaN, or Inf
- `sample_batch`: `ValueError` if any draw value < 0 or ≥ total; `RuntimeError` if total == 0; `TypeError` if input is not a float64 ndarray
- All allocation failures: return `NULL` (sets `PyErr_NoMemory`)

---

## 5. Layer 2: Buffer Implementations

### 5.1 `CFastPERBuffer` (`src/reservoir/c_buffer.py`)

Identical public API to `FastPERBuffer`. Internal difference: replaces the `self._tree` / `self._min_tree` numpy arrays and the `_tree_update` / `_tree_sample_batch` methods with `reservoir._sumtree.SumTree` and `MinTree`. All numpy array storage (states, actions, rewards) and torch tensor conversion remain in Python.

**Ground truth from `fast_buffer.py`:**
- Tree stores `priority ** alpha` (pre-exponentiated float64)
- Sampling uses **stratified draws**: divide `[0, total)` into `batch_size` equal segments, draw one `np.random.uniform(0, segment)` offset per segment
- Method is `add()`, not `insert()`
- No `from_env()` classmethod exists in `FastPERBuffer`; `CFastPERBuffer` does not add one either

**Constructor and public API (matches `FastPERBuffer` exactly):**

```python
class CFastPERBuffer:
    def __init__(
        self,
        capacity: int,
        obs_shape: tuple,
        action_dim: int = 1,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 1e-6,
        device: str = "cpu",
    )

    def add(self, state, action, reward, next_state, done, priority=None) -> None
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None
    def sample(self, batch_size: int) -> FastBatch
    def anneal_beta(self, step: int, total_steps: int, beta_end: float = 1.0) -> None

    @property
    def size(self) -> int
    @property
    def total_priority(self) -> float
```

Returns the same `FastBatch` dataclass as `FastPERBuffer`.

**Stratified sampling in `CFastPERBuffer.sample()`:**

```python
total = self._sum_tree.total
segment = total / batch_size
offsets = np.random.uniform(0, segment, size=batch_size)
values = (offsets + segment * np.arange(batch_size)).astype(np.float64)
indices = self._sum_tree.sample_batch(values)  # C extension call
```

### 5.2 `fast_buffer.py` — No Changes

Not touched. Remains the pure-Python/numpy fallback.

### 5.3 Updated `src/reservoir/__init__.py`

```python
"""reservoir: Certified Exact Prioritized Experience Replay."""

__version__ = "0.2.0"

# Exact buffer (pure Python, arbitrary-precision integer arithmetic)
from reservoir.buffer import ExactPERBuffer

# Fast buffer — auto-selects C-backed or pure-Python implementation
try:
    from reservoir.c_buffer import CFastPERBuffer as FastPERBuffer
    _BACKEND = "c"
except (ImportError, OSError):
    # C extension not built, or ABI mismatch (e.g., after Python upgrade) — fall back
    from reservoir.fast_buffer import FastPERBuffer
    _BACKEND = "python"

# Always importable by explicit name
from reservoir.fast_buffer import FastPERBuffer as PyFastPERBuffer
# Note: the line below will raise ImportError/OSError if the C extension is missing;
# callers who explicitly import CFastPERBuffer must handle that themselves.

# Which backend FastPERBuffer resolves to at import time.
# Read-only; reflects import-time selection; do not mutate at runtime.
backend: str = _BACKEND

__all__ = [
    "ExactPERBuffer",
    "FastPERBuffer",
    "PyFastPERBuffer",
    "backend",
    "__version__",
]
```

`except (ImportError, OSError)` — catches both missing extension and ABI/shared-library errors (e.g., after a Python version upgrade). Other exceptions (logic errors, segfaults) propagate unmasked. `CFastPERBuffer` is intentionally **not** in `__all__` — it is only accessible via direct import from `reservoir.c_buffer` on systems where the extension was built.

---

## 6. Layer 3: PyPI Packaging

### 6.1 Build System: hatchling → setuptools

Required changes:
1. `[build-system]` in `pyproject.toml`
2. Remove `[tool.hatch.build.targets.wheel]`, replace with `[tool.setuptools.packages.find]`
3. Add `setup.py` for C extension registration
4. Add `MANIFEST.in` so C source files are included in the sdist

### 6.2 `pyproject.toml` (key sections)

```toml
[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "reservoir"
version = "0.2.0"
requires-python = ">=3.10"
dependencies = [
    "numpy>=2.2.6",
    "torch>=2.0.0",
    "gymnasium>=1.3.0",
]
# stable-baselines3 removed from runtime deps (not imported by any src/reservoir/ module)
# fpdf2 removed from runtime deps (only used for one-off PDF generation, not in src/)

[project.optional-dependencies]
dev   = ["pytest>=7.0", "hypothesis>=6.0", "pytest-cov>=4.0"]
atari = [
    "stable-baselines3>=2.9.0",         # moved here: only needed for uniform baseline
    "ale-py>=0.9",
    "gymnasium[atari]>=1.0",
    "autorom[accept-rom-license]",       # explicit flag required; silently accepts ROM licenses
    "matplotlib>=3.7",                   # for compare.py PNG output
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.uv]
dev-dependencies = [                    # preserved for uv compat; pip uses [project.optional-dependencies]
    "pytest>=7.0",
    "hypothesis>=6.0",
    "pytest-cov>=4.0",
]
```

**Legal note:** `autorom[accept-rom-license]` silently accepts Atari ROM licenses at install time. This must be documented in the README under the `[atari]` install section.

### 6.3 `setup.py`

```python
from setuptools import setup, Extension

setup(
    ext_modules=[
        Extension(
            "reservoir._sumtree",
            sources=[
                "src/reservoir/csrc/sumtreemodule.c",
                "src/reservoir/csrc/sumtree.c",
            ],
            extra_compile_args=["-O3"],
            # -march=native intentionally omitted: produces CPU-specific binaries
            # unsafe for distributed packages. Users who want it can set CFLAGS before building.
        )
    ]
)
```

The C extension is **always registered** in `setup.py`. If the user's environment lacks a C compiler, `pip install reservoir` fails at build time. The `__init__.py` fallback handles the runtime case where the extension was not built into the installed package.

### 6.4 `MANIFEST.in`

```
include src/reservoir/csrc/sumtree.c
include src/reservoir/csrc/sumtreemodule.c
include src/reservoir/csrc/*.h
```

Required so that `python -m build --sdist` includes the C sources, enabling `pip install reservoir` from an sdist tarball on platforms without a pre-built wheel.

### 6.5 Install Experience

```bash
pip install reservoir                    # attempts C build; __init__.py falls back if missing
pip install reservoir[atari]             # + Atari deps + SB3 + matplotlib
pip install reservoir[atari,dev]         # full development install
CFLAGS="-O3 -march=native" pip install reservoir  # opt-in native tuning
```

---

## 7. Atari Benchmarks

### 7.1 CleanRL Integration

CleanRL is **not** a pip-installable library. Its `dqn_atari.py` (Apache-2.0) is **vendored** into `benchmarks/atari_dqn.py` with attribution comment at the top, plus these modifications:
- `--buffer-type` flag: `c` / `python` / `uniform`
- `CFastPERBuffer` / `PyFastPERBuffer` / SB3 `ReplayBuffer` wired in
- JSON result logging added per run

### 7.2 File Structure

```
benchmarks/
  atari_dqn.py          -- vendored + adapted CleanRL DQN, --buffer-type flag
  run_atari.py          -- CLI runner: N games x M seeds, writes results/
  compare.py            -- loads results/, prints markdown table + saves PNG (matplotlib)
  configs/
    games.txt           -- all 57 Atari game IDs (NoFrameskip-v4 variants)
    hyperparams.yaml    -- Schaul et al. 2016 Table 1 values
  results/              -- one JSON file per (game, seed, buffer_type) run
```

### 7.3 Hyperparameters (Schaul et al. 2016)

| Parameter | Value |
|-----------|-------|
| alpha | 0.6 |
| beta start | 0.4 |
| beta end | 1.0 (linearly annealed over training) |
| epsilon | 1e-6 |
| capacity | 1,000,000 |
| batch size | 32 |
| learning rate | 1e-4 |
| gamma | 0.99 |
| target update freq | 10,000 steps |
| total steps | 50,000,000 (paper used 200M; 50M default for iteration) |

### 7.4 Buffer Types

| `--buffer-type` | Implementation |
|-----------------|----------------|
| `c` | `CFastPERBuffer` (C-backed tree) |
| `python` | `PyFastPERBuffer` (pure numpy fallback) |
| `uniform` | SB3 `ReplayBuffer` (no prioritization — baseline) |

### 7.5 Runner CLI

```bash
# Smoke test: 3 games, 1 seed (CI target)
python -m benchmarks.run_atari --buffer-type c \
  --games BreakoutNoFrameskip-v4 PongNoFrameskip-v4 SpaceInvadersNoFrameskip-v4 \
  --seeds 1

# Full 57-game run
python -m benchmarks.run_atari --buffer-type c --all-57 --seeds 1 2 3
```

Runner is **resumable**: skips any `(game, seed, buffer_type)` combination where a result JSON already exists.

### 7.6 Compute Expectations

| Config | Estimate |
|--------|----------|
| 1 game, 50M steps, 1 GPU | ~3–8 hours |
| Full 57 games × 3 seeds × 3 buffer types | ~1,500–4,000 GPU-hours |
| CI smoke test (3 games × 1 seed) | ~10–24 hours |

Full run requires a compute cluster or cloud GPU allocation.

### 7.7 Reproduction Criterion

Rather than comparing against Schaul 2016 Table 1 (which requires matching their network architecture and exact hyperparameters), the acceptance criterion is **self-consistency**:

> `c` and `python` buffer types produce mean final episode reward within **±5%** of each other across all completed game/seed combinations, confirming correctness equivalence of the C and Python implementations.

Comparison against `uniform` is reported as-is (expected: PER ≥ uniform on most games).

### 7.8 Result Format

```json
{
  "game": "BreakoutNoFrameskip-v4",
  "buffer_type": "c",
  "seed": 1,
  "total_steps": 50000000,
  "episode_rewards": [...],
  "final_mean_reward_100ep": 312.4,
  "duration_seconds": 14400
}
```

---

## 8. Testing

| Test file | Covers |
|-----------|--------|
| `tests/test_c_sumtree.py` | `SumTree` / `MinTree`: update, `sample_batch`, `total`, `minimum`, invariants, power-of-two capacity, `INFINITY` sentinel behavior, edge cases (capacity 1, boundary draws, full buffer) |
| `tests/test_c_buffer.py` | `CFastPERBuffer`: `add`, `sample`, `update_priorities`, IS weights, `anneal_beta`, stratified sampling distribution |
| `tests/test_c_vs_python.py` | Property-based (Hypothesis): same sequence of `add` / `update_priorities` / `sample` calls on `CFastPERBuffer` and `FastPERBuffer` with the same `np.random` seed produces identical `FastBatch` outputs. Priority inputs use float32 range, stored as float64 in both trees. |
| Existing 168 tests | Must all continue to pass; `fast_buffer.py` is untouched |

---

## 9. Implementation Order

1. **C extension** — `csrc/sumtree.c` + `csrc/sumtreemodule.c` + `MANIFEST.in` + `tests/test_c_sumtree.py`
2. **`CFastPERBuffer`** — `src/reservoir/c_buffer.py` + `tests/test_c_buffer.py` + `tests/test_c_vs_python.py`
3. **Packaging** — migrate `pyproject.toml` (hatchling → setuptools), add `setup.py`, update `__init__.py`
4. **Build smoke test** — `pip install -e .` compiles extension; `pytest tests/` all 168+ tests pass
5. **Atari benchmark infrastructure** — `benchmarks/atari_dqn.py`, `run_atari.py`, `compare.py`, `configs/`
6. **Benchmark smoke run** — 3 games × 1 seed × 3 buffer types (CI validation)
7. **Full 57-game run** — on adequate compute
8. **PyPI publish** — `python -m build` + `twine upload dist/*`
