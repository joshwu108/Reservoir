# Reservoir PyPI + C Extension + Atari Benchmarks Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `reservoir` to PyPI with a C-backed `CFastPERBuffer` (O(log N) tree ops in C) and a full 57-game Atari benchmark harness to validate correctness and performance.

**Architecture:** A pure-C sum-tree/min-tree extension (`reservoir._sumtree`) replaces the numpy tree arrays inside a new `CFastPERBuffer` class; the existing `FastPERBuffer` is untouched as a fallback. The build backend migrates from hatchling to setuptools so the C extension compiles at install time. A vendored CleanRL DQN script with a `--buffer-type` flag drives the Atari benchmark suite.

**Tech Stack:** Python 3.13, C (Python C API), setuptools>=61, numpy/torch, gymnasium, CleanRL (vendored), stable-baselines3 (uniform baseline), pytest, Hypothesis, matplotlib

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Create | `src/reservoir/csrc/sumtree.h` | C struct definitions and function prototypes |
| Create | `src/reservoir/csrc/sumtree.c` | Pure C sum-tree + min-tree (no Python headers) |
| Create | `src/reservoir/csrc/sumtreemodule.c` | Python C API module (`reservoir._sumtree`) |
| Create | `tests/test_c_sumtree.py` | Unit tests for `SumTree` and `MinTree` Python classes |
| Create | `src/reservoir/c_buffer.py` | `CFastPERBuffer` — C-tree-backed production buffer |
| Create | `tests/test_c_buffer.py` | Unit tests for `CFastPERBuffer` |
| Create | `tests/test_c_vs_python.py` | Hypothesis parity tests: C vs Python buffer |
| Modify | `pyproject.toml` | hatchling→setuptools, dep cleanup, atari extra |
| Create | `setup.py` | C extension registration |
| Create | `MANIFEST.in` | Include C sources in sdist |
| Modify | `src/reservoir/__init__.py` | Public API + C/Python auto-selection |
| Create | `benchmarks/__init__.py` | Empty — makes benchmarks a package |
| Create | `benchmarks/configs/games.txt` | 57 Atari game IDs |
| Create | `benchmarks/configs/hyperparams.yaml` | Schaul 2016 hyperparameters |
| Create | `benchmarks/atari_dqn.py` | Vendored CleanRL DQN + `--buffer-type` flag |
| Create | `benchmarks/run_atari.py` | Resumable CLI runner |
| Create | `benchmarks/compare.py` | Results comparison: markdown table + PNG |
| Create | `benchmarks/results/.gitkeep` | Tracks results directory in git |

---

## Phase 1: C Extension

### Task 1: C header and failing tests

**Files:**
- Create: `src/reservoir/csrc/sumtree.h`
- Create: `tests/test_c_sumtree.py`

- [ ] **Step 1: Create the csrc directory and header file**

```bash
mkdir -p src/reservoir/csrc
```

Create `src/reservoir/csrc/sumtree.h`:

```c
#ifndef RESERVOIR_SUMTREE_H
#define RESERVOIR_SUMTREE_H

#include <stddef.h>

/* Sum-tree: each internal node = sum of its subtree's leaf values.
   Tree array length = 2 * tree_capacity.
   Leaves at indices [tree_capacity-1 .. 2*tree_capacity-2].
   Values are pre-exponentiated priorities (priority ** alpha). */
typedef struct {
    double      *tree;
    Py_ssize_t   tree_capacity;  /* always a power of 2 */
    Py_ssize_t   capacity;       /* user-requested capacity */
} SumTreeData;

/* Min-tree: same layout, tracks minimums.
   Empty slots initialized to INFINITY. */
typedef struct {
    double      *tree;
    Py_ssize_t   tree_capacity;
    Py_ssize_t   capacity;
} MinTreeData;

SumTreeData *sumtree_create(size_t capacity);
void         sumtree_destroy(SumTreeData *t);
void         sumtree_update(SumTreeData *t, size_t pos, double value);
double       sumtree_get(SumTreeData *t, size_t pos);
double       sumtree_total(SumTreeData *t);
size_t       sumtree_locate(SumTreeData *t, double value);

MinTreeData *mintree_create(size_t capacity);
void         mintree_destroy(MinTreeData *t);
void         mintree_update(MinTreeData *t, size_t pos, double value);
double       mintree_get(MinTreeData *t, size_t pos);
double       mintree_minimum(MinTreeData *t);

#endif /* RESERVOIR_SUMTREE_H */
```

- [ ] **Step 2: Write the failing tests**

Create `tests/test_c_sumtree.py`:

```python
"""Tests for reservoir._sumtree C extension (SumTree and MinTree).

All tests import from reservoir._sumtree, which requires the C extension to be
built first. Run `pip install -e .` before running these tests.
"""
import math
import pytest

_sumtree = pytest.importorskip(
    "reservoir._sumtree",
    reason="C extension not built — run `pip install -e .` first",
)
SumTree = _sumtree.SumTree
MinTree = _sumtree.MinTree


# ---------------------------------------------------------------------------
# SumTree: basic construction
# ---------------------------------------------------------------------------

class TestSumTreeConstruction:
    def test_capacity_rounded_to_power_of_two(self):
        t = SumTree(5)
        assert t.tree_capacity == 8

    def test_exact_power_of_two_unchanged(self):
        t = SumTree(8)
        assert t.tree_capacity == 8

    def test_capacity_1(self):
        t = SumTree(1)
        assert t.tree_capacity == 1

    def test_initial_total_is_zero(self):
        t = SumTree(4)
        assert t.total == 0.0

    def test_invalid_capacity(self):
        with pytest.raises((ValueError, OverflowError)):
            SumTree(0)


# ---------------------------------------------------------------------------
# SumTree: update and total
# ---------------------------------------------------------------------------

class TestSumTreeUpdate:
    def test_single_update_sets_total(self):
        t = SumTree(4)
        t.update(0, 3.0)
        assert t.total == pytest.approx(3.0)

    def test_multiple_updates_sum(self):
        t = SumTree(4)
        t.update(0, 1.0)
        t.update(1, 2.0)
        t.update(2, 3.0)
        t.update(3, 4.0)
        assert t.total == pytest.approx(10.0)

    def test_update_overwrites_previous(self):
        t = SumTree(4)
        t.update(0, 5.0)
        t.update(0, 2.0)
        assert t.total == pytest.approx(2.0)

    def test_get_returns_leaf_value(self):
        t = SumTree(4)
        t.update(2, 7.5)
        assert t.get(2) == pytest.approx(7.5)

    def test_update_out_of_range_raises(self):
        t = SumTree(4)
        with pytest.raises(IndexError):
            t.update(4, 1.0)

    def test_update_negative_raises(self):
        t = SumTree(4)
        with pytest.raises(ValueError):
            t.update(0, -1.0)

    def test_update_nan_raises(self):
        t = SumTree(4)
        with pytest.raises(ValueError):
            t.update(0, float("nan"))

    def test_update_inf_raises(self):
        t = SumTree(4)
        with pytest.raises(ValueError):
            t.update(0, float("inf"))

    def test_update_zero_allowed(self):
        t = SumTree(4)
        t.update(0, 0.0)
        assert t.total == 0.0


# ---------------------------------------------------------------------------
# SumTree: sample_batch
# ---------------------------------------------------------------------------

class TestSumTreeSampleBatch:
    def setup_method(self):
        self.t = SumTree(4)
        # priorities [1, 2, 3, 4], total = 10
        for i, p in enumerate([1.0, 2.0, 3.0, 4.0]):
            self.t.update(i, p)

    def test_sample_batch_returns_list(self):
        result = self.t.sample_batch([0.5])
        assert isinstance(result, list)

    def test_sample_batch_correct_length(self):
        result = self.t.sample_batch([1.0, 5.0, 9.5])
        assert len(result) == 3

    def test_sample_batch_boundary_start(self):
        # draw 0.0 → position 0 (priority 1.0 covers [0, 1))
        result = self.t.sample_batch([0.0])
        assert result[0] == 0

    def test_sample_batch_boundary_end(self):
        # draw 9.99 → position 3 (priority 4.0 covers [6, 10))
        result = self.t.sample_batch([9.99])
        assert result[0] == 3

    def test_sample_batch_proportional(self):
        # position 1 has priority 2.0, covers [1.0, 3.0)
        result = self.t.sample_batch([2.0])
        assert result[0] == 1

    def test_sample_batch_empty_tree_raises(self):
        t = SumTree(4)
        with pytest.raises(RuntimeError):
            t.sample_batch([0.5])

    def test_sample_batch_out_of_range_raises(self):
        with pytest.raises(ValueError):
            self.t.sample_batch([10.0])  # total is exactly 10.0

    def test_sample_batch_negative_raises(self):
        with pytest.raises(ValueError):
            self.t.sample_batch([-0.1])

    def test_sample_batch_capacity_1(self):
        t = SumTree(1)
        t.update(0, 5.0)
        assert t.sample_batch([2.5]) == [0]


# ---------------------------------------------------------------------------
# MinTree: basic behavior
# ---------------------------------------------------------------------------

class TestMinTree:
    def test_initial_minimum_is_infinity(self):
        t = MinTree(4)
        assert math.isinf(t.minimum) and t.minimum > 0

    def test_update_single_sets_minimum(self):
        t = MinTree(4)
        t.update(0, 3.0)
        assert t.minimum == pytest.approx(3.0)

    def test_minimum_tracks_smallest(self):
        t = MinTree(4)
        t.update(0, 5.0)
        t.update(1, 2.0)
        t.update(2, 8.0)
        assert t.minimum == pytest.approx(2.0)

    def test_update_replaces_minimum(self):
        t = MinTree(4)
        t.update(0, 2.0)
        t.update(1, 5.0)
        t.update(0, 7.0)  # overwrite min slot
        assert t.minimum == pytest.approx(5.0)

    def test_get_returns_leaf_value(self):
        t = MinTree(4)
        t.update(1, 4.5)
        assert t.get(1) == pytest.approx(4.5)

    def test_tree_capacity_power_of_two(self):
        t = MinTree(6)
        assert t.tree_capacity == 8

    def test_update_out_of_range_raises(self):
        t = MinTree(4)
        with pytest.raises(IndexError):
            t.update(4, 1.0)

    def test_update_negative_raises(self):
        t = MinTree(4)
        with pytest.raises(ValueError):
            t.update(0, -1.0)

    def test_empty_slot_acts_as_infinity_in_min(self):
        # Only slot 1 set — minimum should be that value, not 0
        t = MinTree(4)
        t.update(1, 3.0)
        assert t.minimum == pytest.approx(3.0)
```

- [ ] **Step 3: Run tests — expect ImportError skip**

```bash
uv run pytest tests/test_c_sumtree.py -v
```

Expected: all tests **skipped** with message `C extension not built`.

- [ ] **Step 4: Commit the header and test skeleton**

```bash
git add src/reservoir/csrc/sumtree.h tests/test_c_sumtree.py
git commit -m "test: add C sumtree test skeleton (skipped until extension built)"
```

---

### Task 2: Pure C implementation (`sumtree.c`)

**Files:**
- Create: `src/reservoir/csrc/sumtree.c`

- [ ] **Step 1: Create sumtree.c**

Create `src/reservoir/csrc/sumtree.c`:

```c
#include "sumtree.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>   /* INFINITY, isfinite */

/* ---- helpers ---- */

static size_t next_power_of_two(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

/* ---- SumTree ---- */

SumTreeData *sumtree_create(size_t capacity) {
    SumTreeData *t = (SumTreeData *)malloc(sizeof(SumTreeData));
    if (!t) return NULL;
    t->capacity      = capacity;
    t->tree_capacity = next_power_of_two(capacity);
    t->tree = (double *)calloc(2 * t->tree_capacity, sizeof(double));
    if (!t->tree) { free(t); return NULL; }
    return t;
}

void sumtree_destroy(SumTreeData *t) {
    if (t) { free(t->tree); free(t); }
}

void sumtree_update(SumTreeData *t, size_t pos, double value) {
    size_t idx = t->tree_capacity - 1 + pos;
    t->tree[idx] = value;
    while (idx > 0) {
        idx = (idx - 1) >> 1;  /* parent */
        t->tree[idx] = t->tree[2*idx+1] + t->tree[2*idx+2];
    }
}

double sumtree_get(SumTreeData *t, size_t pos) {
    return t->tree[t->tree_capacity - 1 + pos];
}

double sumtree_total(SumTreeData *t) {
    return t->tree[0];
}

size_t sumtree_locate(SumTreeData *t, double value) {
    size_t idx = 0;
    while (idx < t->tree_capacity - 1) {
        size_t left = 2 * idx + 1;
        if (value < t->tree[left]) {
            idx = left;
        } else {
            value -= t->tree[left];
            idx    = left + 1;
        }
    }
    return idx - (t->tree_capacity - 1);
}

/* ---- MinTree ---- */

MinTreeData *mintree_create(size_t capacity) {
    MinTreeData *t = (MinTreeData *)malloc(sizeof(MinTreeData));
    if (!t) return NULL;
    t->capacity      = capacity;
    t->tree_capacity = next_power_of_two(capacity);
    size_t n = 2 * t->tree_capacity;
    t->tree = (double *)malloc(n * sizeof(double));
    if (!t->tree) { free(t); return NULL; }
    for (size_t i = 0; i < n; i++) t->tree[i] = INFINITY;
    return t;
}

void mintree_destroy(MinTreeData *t) {
    if (t) { free(t->tree); free(t); }
}

void mintree_update(MinTreeData *t, size_t pos, double value) {
    size_t idx = t->tree_capacity - 1 + pos;
    t->tree[idx] = value;
    while (idx > 0) {
        idx = (idx - 1) >> 1;
        double l = t->tree[2*idx+1], r = t->tree[2*idx+2];
        t->tree[idx] = (l < r) ? l : r;
    }
}

double mintree_get(MinTreeData *t, size_t pos) {
    return t->tree[t->tree_capacity - 1 + pos];
}

double mintree_minimum(MinTreeData *t) {
    return t->tree[0];
}
```

- [ ] **Step 2: Commit the C implementation**

```bash
git add src/reservoir/csrc/sumtree.c
git commit -m "feat: add pure C sum-tree and min-tree implementation"
```

---

### Task 3: Python C API module (`sumtreemodule.c`)

**Files:**
- Create: `src/reservoir/csrc/sumtreemodule.c`

- [ ] **Step 1: Create sumtreemodule.c**

Create `src/reservoir/csrc/sumtreemodule.c`:

```c
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include "sumtree.h"

/* ================================================================
   SumTree Python type
   ================================================================ */

typedef struct {
    PyObject_HEAD
    SumTreeData *data;
} SumTreeObject;

static int
SumTree_init(SumTreeObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"capacity", NULL};
    Py_ssize_t capacity;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "n", kwlist, &capacity))
        return -1;
    if (capacity <= 0) {
        PyErr_SetString(PyExc_ValueError, "capacity must be positive");
        return -1;
    }
    self->data = sumtree_create((size_t)capacity);
    if (!self->data) { PyErr_NoMemory(); return -1; }
    return 0;
}

static void
SumTree_dealloc(SumTreeObject *self)
{
    sumtree_destroy(self->data);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
SumTree_update(SumTreeObject *self, PyObject *args)
{
    Py_ssize_t pos;
    double value;
    if (!PyArg_ParseTuple(args, "nd", &pos, &value)) return NULL;
    if (pos < 0 || (size_t)pos >= self->data->capacity) {
        PyErr_Format(PyExc_IndexError,
            "position %zd out of range [0, %zu)", pos, self->data->capacity);
        return NULL;
    }
    if (value < 0.0 || !isfinite(value)) {
        PyErr_SetString(PyExc_ValueError,
            "priority must be finite and non-negative");
        return NULL;
    }
    sumtree_update(self->data, (size_t)pos, value);
    Py_RETURN_NONE;
}

static PyObject *
SumTree_get(SumTreeObject *self, PyObject *args)
{
    Py_ssize_t pos;
    if (!PyArg_ParseTuple(args, "n", &pos)) return NULL;
    if (pos < 0 || (size_t)pos >= self->data->capacity) {
        PyErr_Format(PyExc_IndexError,
            "position %zd out of range [0, %zu)", pos, self->data->capacity);
        return NULL;
    }
    return PyFloat_FromDouble(sumtree_get(self->data, (size_t)pos));
}

static PyObject *
SumTree_sample_batch(SumTreeObject *self, PyObject *args)
{
    PyObject *values_obj;
    if (!PyArg_ParseTuple(args, "O", &values_obj)) return NULL;

    double total = sumtree_total(self->data);
    if (total <= 0.0) {
        PyErr_SetString(PyExc_RuntimeError,
            "cannot sample from empty tree (total == 0)");
        return NULL;
    }

    Py_ssize_t n = PySequence_Length(values_obj);
    if (n < 0) return NULL;

    PyObject *result = PyList_New(n);
    if (!result) return NULL;

    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PySequence_GetItem(values_obj, i);
        if (!item) { Py_DECREF(result); return NULL; }
        double value = PyFloat_AsDouble(item);
        Py_DECREF(item);
        if (value == -1.0 && PyErr_Occurred()) { Py_DECREF(result); return NULL; }
        if (value < 0.0 || value >= total) {
            PyErr_Format(PyExc_ValueError,
                "draw value %f out of range [0, %f)", value, total);
            Py_DECREF(result); return NULL;
        }
        size_t pos = sumtree_locate(self->data, value);
        PyList_SET_ITEM(result, i, PyLong_FromSize_t(pos));
    }
    return result;
}

static PyObject *
SumTree_get_total(SumTreeObject *self, void *closure)
{
    return PyFloat_FromDouble(sumtree_total(self->data));
}

static PyObject *
SumTree_get_tree_capacity(SumTreeObject *self, void *closure)
{
    return PyLong_FromSize_t(self->data->tree_capacity);
}

static PyMethodDef SumTree_methods[] = {
    {"update",       (PyCFunction)SumTree_update,       METH_VARARGS, "update(pos, value)"},
    {"get",          (PyCFunction)SumTree_get,          METH_VARARGS, "get(pos) -> float"},
    {"sample_batch", (PyCFunction)SumTree_sample_batch, METH_VARARGS, "sample_batch(values) -> list[int]"},
    {NULL, NULL, 0, NULL}
};

static PyGetSetDef SumTree_getsetters[] = {
    {"total",         (getter)SumTree_get_total,         NULL, "sum of all leaf priorities", NULL},
    {"tree_capacity", (getter)SumTree_get_tree_capacity, NULL, "internal tree capacity (power of 2)", NULL},
    {NULL}
};

static PyTypeObject SumTreeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "reservoir._sumtree.SumTree",
    .tp_basicsize = sizeof(SumTreeObject),
    .tp_dealloc   = (destructor)SumTree_dealloc,
    .tp_flags     = Py_TPFLAGS_DEFAULT,
    .tp_doc       = "C-backed sum-tree for O(log N) priority sampling",
    .tp_methods   = SumTree_methods,
    .tp_getset    = SumTree_getsetters,
    .tp_init      = (initproc)SumTree_init,
    .tp_new       = PyType_GenericNew,
};


/* ================================================================
   MinTree Python type
   ================================================================ */

typedef struct {
    PyObject_HEAD
    MinTreeData *data;
} MinTreeObject;

static int
MinTree_init(MinTreeObject *self, PyObject *args, PyObject *kwds)
{
    static char *kwlist[] = {"capacity", NULL};
    Py_ssize_t capacity;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "n", kwlist, &capacity))
        return -1;
    if (capacity <= 0) {
        PyErr_SetString(PyExc_ValueError, "capacity must be positive");
        return -1;
    }
    self->data = mintree_create((size_t)capacity);
    if (!self->data) { PyErr_NoMemory(); return -1; }
    return 0;
}

static void
MinTree_dealloc(MinTreeObject *self)
{
    mintree_destroy(self->data);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *
MinTree_update(MinTreeObject *self, PyObject *args)
{
    Py_ssize_t pos;
    double value;
    if (!PyArg_ParseTuple(args, "nd", &pos, &value)) return NULL;
    if (pos < 0 || (size_t)pos >= self->data->capacity) {
        PyErr_Format(PyExc_IndexError,
            "position %zd out of range [0, %zu)", pos, self->data->capacity);
        return NULL;
    }
    if (value < 0.0 || !isfinite(value)) {
        PyErr_SetString(PyExc_ValueError,
            "priority must be finite and non-negative");
        return NULL;
    }
    mintree_update(self->data, (size_t)pos, value);
    Py_RETURN_NONE;
}

static PyObject *
MinTree_get(MinTreeObject *self, PyObject *args)
{
    Py_ssize_t pos;
    if (!PyArg_ParseTuple(args, "n", &pos)) return NULL;
    if (pos < 0 || (size_t)pos >= self->data->capacity) {
        PyErr_Format(PyExc_IndexError,
            "position %zd out of range [0, %zu)", pos, self->data->capacity);
        return NULL;
    }
    return PyFloat_FromDouble(mintree_get(self->data, (size_t)pos));
}

static PyObject *
MinTree_get_minimum(MinTreeObject *self, void *closure)
{
    return PyFloat_FromDouble(mintree_minimum(self->data));
}

static PyObject *
MinTree_get_tree_capacity(MinTreeObject *self, void *closure)
{
    return PyLong_FromSize_t(self->data->tree_capacity);
}

static PyMethodDef MinTree_methods[] = {
    {"update", (PyCFunction)MinTree_update, METH_VARARGS, "update(pos, value)"},
    {"get",    (PyCFunction)MinTree_get,    METH_VARARGS, "get(pos) -> float"},
    {NULL, NULL, 0, NULL}
};

static PyGetSetDef MinTree_getsetters[] = {
    {"minimum",       (getter)MinTree_get_minimum,       NULL, "minimum leaf value", NULL},
    {"tree_capacity", (getter)MinTree_get_tree_capacity, NULL, "internal tree capacity (power of 2)", NULL},
    {NULL}
};

static PyTypeObject MinTreeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name      = "reservoir._sumtree.MinTree",
    .tp_basicsize = sizeof(MinTreeObject),
    .tp_dealloc   = (destructor)MinTree_dealloc,
    .tp_flags     = Py_TPFLAGS_DEFAULT,
    .tp_doc       = "C-backed min-tree for IS weight normalization",
    .tp_methods   = MinTree_methods,
    .tp_getset    = MinTree_getsetters,
    .tp_init      = (initproc)MinTree_init,
    .tp_new       = PyType_GenericNew,
};


/* ================================================================
   Module definition
   ================================================================ */

static PyModuleDef _sumtreemodule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "reservoir._sumtree",
    .m_doc  = "C-backed sum-tree and min-tree for PER sampling",
    .m_size = -1,
};

PyMODINIT_FUNC
PyInit__sumtree(void)
{
    if (PyType_Ready(&SumTreeType) < 0) return NULL;
    if (PyType_Ready(&MinTreeType) < 0) return NULL;

    PyObject *m = PyModule_Create(&_sumtreemodule);
    if (!m) return NULL;

    Py_INCREF(&SumTreeType);
    if (PyModule_AddObject(m, "SumTree", (PyObject *)&SumTreeType) < 0) {
        Py_DECREF(&SumTreeType); Py_DECREF(m); return NULL;
    }
    Py_INCREF(&MinTreeType);
    if (PyModule_AddObject(m, "MinTree", (PyObject *)&MinTreeType) < 0) {
        Py_DECREF(&MinTreeType); Py_DECREF(m); return NULL;
    }
    return m;
}
```

- [ ] **Step 2: Commit the Python C API module**

```bash
git add src/reservoir/csrc/sumtreemodule.c
git commit -m "feat: add Python C API bindings for sum-tree and min-tree"
```

---

### Task 4: Build system — setup.py, pyproject.toml migration, MANIFEST.in

**Files:**
- Create: `setup.py`
- Modify: `pyproject.toml`
- Create: `MANIFEST.in`

- [ ] **Step 1: Create setup.py**

Create `setup.py` in the repo root:

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
            # -march=native intentionally omitted: unsafe for distributed packages.
            # Users wanting native tuning can set CFLAGS="-O3 -march=native" before install.
        )
    ]
)
```

- [ ] **Step 2: Update pyproject.toml**

Replace the full `pyproject.toml` with:

```toml
[build-system]
requires = ["setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "reservoir"
version = "0.2.0"
description = "Certified Exact Prioritized Experience Replay with Crash Atomicity and Sampling Attestation"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "Apache-2.0" }
dependencies = [
    "numpy>=2.2.6",
    "torch>=2.0.0",
    "gymnasium>=1.3.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "hypothesis>=6.0",
    "pytest-cov>=4.0",
]
atari = [
    "stable-baselines3>=2.9.0",
    "ale-py>=0.9",
    "gymnasium[atari]>=1.0",
    "autorom[accept-rom-license]",
    "matplotlib>=3.7",
]

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "--tb=short -v"

[tool.uv]
dev-dependencies = [
    "pytest>=7.0",
    "hypothesis>=6.0",
    "pytest-cov>=4.0",
]
```

- [ ] **Step 3: Create MANIFEST.in**

Create `MANIFEST.in` in the repo root:

```
include src/reservoir/csrc/sumtree.c
include src/reservoir/csrc/sumtreemodule.c
include src/reservoir/csrc/*.h
include setup.py
```

- [ ] **Step 4: Build the C extension**

```bash
pip install -e . --no-build-isolation
```

Expected: compilation output ending in `Successfully installed reservoir-0.2.0`.
The `.so` file appears at `src/reservoir/_sumtree.cpython-313-darwin.so` (macOS) or similar.

- [ ] **Step 5: Run C sumtree tests — expect PASS**

```bash
uv run pytest tests/test_c_sumtree.py -v
```

Expected: all tests **PASS** (no longer skipped).

- [ ] **Step 6: Run full test suite — existing tests must still pass**

```bash
uv run pytest tests/ -v
```

Expected: all 168 existing tests pass + new C sumtree tests pass.

- [ ] **Step 7: Commit build system and verify passing**

```bash
git add setup.py pyproject.toml MANIFEST.in
git commit -m "build: migrate to setuptools, add C extension build, MANIFEST.in"
```

---

## Phase 2: CFastPERBuffer

### Task 5: CFastPERBuffer implementation

**Files:**
- Create: `src/reservoir/c_buffer.py`
- Create: `tests/test_c_buffer.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_c_buffer.py`:

```python
"""Tests for CFastPERBuffer — C-tree-backed production buffer."""
import numpy as np
import pytest
import torch

pytest.importorskip(
    "reservoir._sumtree",
    reason="C extension not built — run `pip install -e .` first",
)

from reservoir.c_buffer import CFastPERBuffer
from reservoir.fast_buffer import FastBatch


OBS_SHAPE = (4,)
ACTION_DIM = 1
CAPACITY = 16


def make_buf(**kwargs):
    defaults = dict(
        capacity=CAPACITY,
        obs_shape=OBS_SHAPE,
        action_dim=ACTION_DIM,
        alpha=0.6,
        beta=0.4,
        epsilon=1e-6,
        device="cpu",
    )
    defaults.update(kwargs)
    return CFastPERBuffer(**defaults)


def fill(buf, n, priority=None):
    for i in range(n):
        obs = np.ones(OBS_SHAPE, dtype=np.float32) * i
        buf.add(obs, 0, float(i), obs, False, priority=priority)


class TestConstruction:
    def test_creates_without_error(self):
        buf = make_buf()
        assert buf.size == 0

    def test_total_priority_zero_initially(self):
        buf = make_buf()
        assert buf.total_priority == 0.0


class TestAdd:
    def test_size_increments(self):
        buf = make_buf()
        fill(buf, 5)
        assert buf.size == 5

    def test_size_capped_at_capacity(self):
        buf = make_buf()
        fill(buf, CAPACITY + 5)
        assert buf.size == CAPACITY

    def test_total_priority_increases(self):
        buf = make_buf()
        fill(buf, 4, priority=1.0)
        assert buf.total_priority > 0.0

    def test_add_sets_max_priority_for_none(self):
        buf = make_buf()
        buf.add(np.zeros(OBS_SHAPE), 0, 0.0, np.zeros(OBS_SHAPE), False, priority=2.0)
        buf.add(np.zeros(OBS_SHAPE), 0, 0.0, np.zeros(OBS_SHAPE), False, priority=None)
        # Second add uses max priority (raw value 2.0 stored in _max_priority after first add)
        assert buf.total_priority > 0.0


class TestSample:
    def setup_method(self):
        self.buf = make_buf()
        fill(self.buf, CAPACITY, priority=1.0)

    def test_sample_returns_fastbatch(self):
        batch = self.buf.sample(4)
        assert isinstance(batch, FastBatch)

    def test_sample_correct_shapes(self):
        batch = self.buf.sample(4)
        assert batch.states.shape == (4, *OBS_SHAPE)
        assert batch.actions.shape == (4,)
        assert batch.rewards.shape == (4,)
        assert batch.is_weights.shape == (4,)

    def test_sample_is_weights_in_range(self):
        batch = self.buf.sample(4)
        assert (batch.is_weights >= 0).all()
        assert (batch.is_weights <= 1.0 + 1e-6).all()

    def test_sample_indices_in_range(self):
        batch = self.buf.sample(4)
        assert (batch.indices >= 0).all()
        assert (batch.indices < CAPACITY).all()

    def test_sample_returns_tensors(self):
        batch = self.buf.sample(4)
        assert isinstance(batch.states, torch.Tensor)
        assert isinstance(batch.is_weights, torch.Tensor)


class TestUpdatePriorities:
    def test_update_changes_total(self):
        buf = make_buf()
        fill(buf, CAPACITY, priority=1.0)
        before = buf.total_priority
        indices = np.array([0, 1])
        buf.update_priorities(indices, np.array([5.0, 5.0]))
        assert buf.total_priority != pytest.approx(before)


class TestAnnealBeta:
    def test_beta_increases(self):
        buf = make_buf(beta=0.4)
        initial_beta = buf.beta
        buf.anneal_beta(step=500, total_steps=1000)
        assert buf.beta > initial_beta

    def test_beta_does_not_exceed_end(self):
        buf = make_buf(beta=0.4)
        buf.anneal_beta(step=10000, total_steps=1000)
        assert buf.beta <= 1.0
```

- [ ] **Step 2: Run — expect ImportError / ModuleNotFoundError**

```bash
uv run pytest tests/test_c_buffer.py -v
```

Expected: collection error or skip (c_buffer.py does not exist yet).

- [ ] **Step 3: Implement CFastPERBuffer**

Create `src/reservoir/c_buffer.py`:

```python
"""reservoir.c_buffer — C-tree-backed Prioritized Experience Replay buffer.

Drop-in replacement for FastPERBuffer. Uses reservoir._sumtree.SumTree and
MinTree for O(log N) priority operations in C. All numpy/torch data storage
is unchanged from FastPERBuffer.

Requires the C extension to be built:
    pip install -e .
"""
from __future__ import annotations

import numpy as np
import torch
from typing import Optional

from reservoir._sumtree import SumTree, MinTree
from reservoir.fast_buffer import FastBatch


class CFastPERBuffer:
    """Prioritized Experience Replay buffer backed by a C sum-tree.

    Public API is identical to FastPERBuffer. Only the internal tree
    implementation differs (C vs numpy).

    Parameters
    ----------
    capacity : int
        Maximum number of transitions.
    obs_shape : tuple
        Shape of a single observation.
    action_dim : int
        Number of action dimensions (1 for discrete).
    alpha : float
        Priority exponent. 0 = uniform, 1 = full prioritization.
    beta : float
        IS correction exponent. Anneal from 0.4 to 1.0 over training.
    epsilon : float
        Minimum priority offset to prevent zero priorities.
    device : str
        Torch device for returned tensors.
    """

    def __init__(
        self,
        capacity: int,
        obs_shape: tuple,
        action_dim: int = 1,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 1e-6,
        device: str = "cpu",
    ) -> None:
        self.capacity   = capacity
        self.obs_shape  = obs_shape
        self.alpha      = alpha
        self.beta       = beta
        self.epsilon    = epsilon
        self.device     = device

        self._size = 0
        self._ptr  = 0  # circular write pointer
        self._max_priority: float = 1.0

        # Pre-allocated numpy arrays (same layout as FastPERBuffer)
        self._states      = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._next_states = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self._actions     = np.zeros((capacity, action_dim), dtype=np.float32)
        self._rewards     = np.zeros((capacity,), dtype=np.float32)
        self._dones       = np.zeros((capacity,), dtype=np.float32)

        # C-backed trees (mirror FastPERBuffer's internal _tree / _min_tree)
        self._sum_tree = SumTree(capacity)
        self._min_tree = MinTree(capacity)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        return self._size

    @property
    def total_priority(self) -> float:
        return self._sum_tree.total

    # ------------------------------------------------------------------
    # Tree operations (mirrors FastPERBuffer._tree_update)
    # ------------------------------------------------------------------

    def _tree_update(self, pos: int, priority: float) -> None:
        """Update tree at pos with pre-exponentiated priority."""
        p_alpha = float(priority ** self.alpha)
        self._sum_tree.update(pos, p_alpha)
        self._min_tree.update(pos, p_alpha)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(
        self,
        state: np.ndarray,
        action,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        priority: Optional[float] = None,
    ) -> None:
        """Add a transition. O(log N)."""
        if priority is None:
            priority = self._max_priority
        else:
            self._max_priority = max(self._max_priority, priority)
            priority = abs(priority) + self.epsilon

        pos = self._ptr
        self._states[pos]      = state
        self._next_states[pos] = next_state
        self._actions[pos]     = action
        self._rewards[pos]     = reward
        self._dones[pos]       = float(done)
        self._tree_update(pos, priority)

        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def update_priorities(
        self, indices: np.ndarray, td_errors: np.ndarray
    ) -> None:
        """Update priorities after learning. Call after each training step."""
        for idx, err in zip(indices, td_errors):
            priority = float(abs(err)) + self.epsilon
            self._max_priority = max(self._max_priority, priority)
            self._tree_update(int(idx), priority)

    def sample(self, batch_size: int) -> FastBatch:
        """Sample a batch using stratified PER sampling.

        Mirrors FastPERBuffer.sample() exactly: stratified draws, IS weights,
        same FastBatch return type.
        """
        assert self._size >= batch_size, (
            f"Buffer has {self._size} transitions, need {batch_size}"
        )

        total       = self._sum_tree.total
        min_p_alpha = self._min_tree.minimum
        segment     = total / batch_size

        # Stratified draws — one per segment (matches FastPERBuffer exactly)
        offsets = np.random.uniform(0, segment, size=batch_size)
        values  = (offsets + segment * np.arange(batch_size)).astype(np.float64)

        # C tree walk — returns list of int positions
        indices = np.array(self._sum_tree.sample_batch(values.tolist()), dtype=np.int64)
        priorities = np.array(
            [self._sum_tree.get(int(i)) for i in indices], dtype=np.float64
        )

        # IS weights (same formula as FastPERBuffer)
        n = self._size
        probs      = np.maximum(priorities / total, 1e-10)
        max_weight = (n * min_p_alpha / total) ** (-self.beta) if min_p_alpha > 0 else 1.0
        weights    = ((n * probs) ** (-self.beta) / max_weight).clip(0.0, 1.0).astype(np.float32)

        def _t(arr: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(arr.copy()).to(self.device)

        return FastBatch(
            states      = _t(self._states[indices]),
            actions     = _t(self._actions[indices]).squeeze(-1).long()
                          if self._actions.shape[1] == 1
                          else _t(self._actions[indices]),
            rewards     = _t(self._rewards[indices]),
            next_states = _t(self._next_states[indices]),
            dones       = _t(self._dones[indices]),
            is_weights  = _t(weights),
            indices     = indices,
        )

    def anneal_beta(
        self, step: int, total_steps: int, beta_end: float = 1.0
    ) -> None:
        """Linearly anneal beta from initial value to beta_end."""
        self.beta = min(
            beta_end,
            self.beta + (beta_end - self.beta) * step / total_steps,
        )
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
uv run pytest tests/test_c_buffer.py -v
```

Expected: all tests **PASS**.

- [ ] **Step 5: Run full suite**

```bash
uv run pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/reservoir/c_buffer.py tests/test_c_buffer.py
git commit -m "feat: add CFastPERBuffer backed by C sum-tree"
```

---

### Task 6: Hypothesis parity tests

**Files:**
- Create: `tests/test_c_vs_python.py`

- [ ] **Step 1: Write parity tests**

Create `tests/test_c_vs_python.py`:

```python
"""Property-based parity: CFastPERBuffer must produce identical results to FastPERBuffer.

Uses Hypothesis to generate random sequences of add/update_priorities/sample calls.
Both buffers receive the same inputs and the same numpy random seed; outputs must match.
"""
import numpy as np
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

pytest.importorskip(
    "reservoir._sumtree",
    reason="C extension not built",
)

from reservoir.c_buffer import CFastPERBuffer
from reservoir.fast_buffer import FastPERBuffer

OBS_SHAPE  = (4,)
ACTION_DIM = 1
CAPACITY   = 32
ALPHA      = 0.6
BETA       = 0.4
EPSILON    = 1e-6


def make_both(seed=0):
    """Create a CFastPERBuffer and FastPERBuffer with identical config."""
    kwargs = dict(
        capacity=CAPACITY,
        obs_shape=OBS_SHAPE,
        action_dim=ACTION_DIM,
        alpha=ALPHA,
        beta=BETA,
        epsilon=EPSILON,
        device="cpu",
    )
    return CFastPERBuffer(**kwargs), FastPERBuffer(**kwargs)


def add_same(c_buf, py_buf, n_transitions, priorities):
    """Add n transitions with given float32 priorities to both buffers."""
    for i, p in zip(range(n_transitions), priorities):
        obs = np.ones(OBS_SHAPE, dtype=np.float32) * (i % 10)
        for buf in (c_buf, py_buf):
            buf.add(obs, 0, float(i % 5), obs, False, priority=float(p))


@given(
    n=st.integers(min_value=CAPACITY, max_value=CAPACITY),
    priorities=st.lists(
        st.floats(min_value=1e-3, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=CAPACITY,
        max_size=CAPACITY,
    ),
    batch_size=st.integers(min_value=1, max_value=8),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=50, deadline=5000)
def test_sample_identical(n, priorities, batch_size, seed):
    """CFastPERBuffer.sample() must return the same indices and IS weights as FastPERBuffer."""
    c_buf, py_buf = make_both()
    add_same(c_buf, py_buf, n, priorities)

    assume(c_buf.size >= batch_size and py_buf.size >= batch_size)

    np.random.seed(seed)
    c_batch = c_buf.sample(batch_size)

    np.random.seed(seed)
    py_batch = py_buf.sample(batch_size)

    np.testing.assert_array_equal(
        c_batch.indices, py_batch.indices,
        err_msg="Sampled indices differ between C and Python buffers",
    )
    np.testing.assert_allclose(
        c_batch.is_weights.numpy(), py_batch.is_weights.numpy(),
        rtol=1e-5, atol=1e-6,
        err_msg="IS weights differ between C and Python buffers",
    )


@given(
    priorities=st.lists(
        st.floats(min_value=1e-3, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=CAPACITY,
        max_size=CAPACITY,
    ),
)
@settings(max_examples=30, deadline=3000)
def test_total_priority_identical(priorities):
    """Both buffers must report identical total_priority after same inserts."""
    c_buf, py_buf = make_both()
    add_same(c_buf, py_buf, CAPACITY, priorities)
    assert c_buf.total_priority == pytest.approx(py_buf.total_priority, rel=1e-9)


@given(
    priorities=st.lists(
        st.floats(min_value=1e-3, max_value=10.0, allow_nan=False, allow_infinity=False),
        min_size=CAPACITY,
        max_size=CAPACITY,
    ),
    new_td_errors=st.lists(
        st.floats(min_value=0.0, max_value=5.0, allow_nan=False, allow_infinity=False),
        min_size=4, max_size=4,
    ),
)
@settings(max_examples=30, deadline=3000)
def test_update_priorities_identical(priorities, new_td_errors):
    """update_priorities must produce identical total_priority in both buffers."""
    c_buf, py_buf = make_both()
    add_same(c_buf, py_buf, CAPACITY, priorities)

    indices = np.array([0, 1, 2, 3])
    td = np.array(new_td_errors, dtype=np.float64)
    c_buf.update_priorities(indices, td)
    py_buf.update_priorities(indices, td)

    assert c_buf.total_priority == pytest.approx(py_buf.total_priority, rel=1e-9)
```

- [ ] **Step 2: Run parity tests**

```bash
uv run pytest tests/test_c_vs_python.py -v
```

Expected: all Hypothesis examples **PASS**. If any fail, inspect the counterexample printed by Hypothesis — it will show the exact seed and priorities causing divergence.

- [ ] **Step 3: Commit**

```bash
git add tests/test_c_vs_python.py
git commit -m "test: add Hypothesis parity tests for CFastPERBuffer vs FastPERBuffer"
```

---

## Phase 3: Package and `__init__.py`

### Task 7: Update `__init__.py` and verify full packaging

**Files:**
- Modify: `src/reservoir/__init__.py`

- [ ] **Step 1: Update __init__.py**

Replace `src/reservoir/__init__.py`:

```python
"""reservoir: Certified Exact Prioritized Experience Replay
with Crash Atomicity and Sampling Attestation.
"""

__version__ = "0.2.0"

# Exact buffer (pure Python, arbitrary-precision integer arithmetic)
from reservoir.buffer import ExactPERBuffer

# Fast buffer — auto-selects C-backed or pure-Python implementation
try:
    from reservoir.c_buffer import CFastPERBuffer as FastPERBuffer
    _BACKEND = "c"
except (ImportError, OSError):
    # C extension not built, or ABI mismatch after Python upgrade — fall back
    from reservoir.fast_buffer import FastPERBuffer  # type: ignore[assignment]
    _BACKEND = "python"

# Always importable by explicit name
from reservoir.fast_buffer import FastPERBuffer as PyFastPERBuffer

# Which backend FastPERBuffer resolves to at import time.
# Read-only. Reflects import-time selection. Do not mutate at runtime.
backend: str = _BACKEND

__all__ = [
    "ExactPERBuffer",
    "FastPERBuffer",
    "PyFastPERBuffer",
    "backend",
    "__version__",
]
```

- [ ] **Step 2: Verify auto-selection works**

```bash
uv run python -c "import reservoir; print(reservoir.backend, reservoir.__version__)"
```

Expected: `c 0.2.0`

- [ ] **Step 3: Run the full suite one more time**

```bash
uv run pytest tests/ -v
```

Expected: 168+ tests pass, no regressions.

- [ ] **Step 4: Verify sdist includes C sources**

```bash
uv run python -m build --sdist --no-isolation 2>&1 | tail -5
tar tzf dist/reservoir-0.2.0.tar.gz | grep csrc
```

Expected: lines like `reservoir-0.2.0/src/reservoir/csrc/sumtree.c`.

- [ ] **Step 5: Commit**

```bash
git add src/reservoir/__init__.py
git commit -m "feat: update __init__.py with public API and C/Python auto-selection"
```

---

## Phase 4: Atari Benchmark Infrastructure

### Task 8: Configs and scaffold

**Files:**
- Create: `benchmarks/__init__.py`
- Create: `benchmarks/configs/games.txt`
- Create: `benchmarks/configs/hyperparams.yaml`
- Create: `benchmarks/results/.gitkeep`

- [ ] **Step 1: Create the directory structure**

```bash
mkdir -p benchmarks/configs benchmarks/results
touch benchmarks/__init__.py benchmarks/results/.gitkeep
```

- [ ] **Step 2: Create games.txt**

Create `benchmarks/configs/games.txt`. The list below is the 60-game ALE set commonly used;
verify against `gymnasium.envs.registry` for your installed ALE version and trim to exactly
the 57 games you intend to run (standard: remove AirRaid, Carnival, ElevatorAction,
JourneyEscape, Pooyan if targeting the canonical Atari-57 benchmark set):

```
AirRaidNoFrameskip-v4
AlienNoFrameskip-v4
AmidarNoFrameskip-v4
AssaultNoFrameskip-v4
AsterixNoFrameskip-v4
AsteroidsNoFrameskip-v4
AtlantisNoFrameskip-v4
BankHeistNoFrameskip-v4
BattleZoneNoFrameskip-v4
BeamRiderNoFrameskip-v4
BerzerkNoFrameskip-v4
BowlingNoFrameskip-v4
BoxingNoFrameskip-v4
BreakoutNoFrameskip-v4
CarnivalNoFrameskip-v4
CentipedeNoFrameskip-v4
ChopperCommandNoFrameskip-v4
CrazyClimberNoFrameskip-v4
DemonAttackNoFrameskip-v4
DoubleDunkNoFrameskip-v4
ElevatorActionNoFrameskip-v4
EnduroNoFrameskip-v4
FishingDerbyNoFrameskip-v4
FreewayNoFrameskip-v4
FrostbiteNoFrameskip-v4
GopherNoFrameskip-v4
GravitarNoFrameskip-v4
HeroNoFrameskip-v4
IceHockeyNoFrameskip-v4
JamesbondNoFrameskip-v4
JourneyEscapeNoFrameskip-v4
KangarooNoFrameskip-v4
KrullNoFrameskip-v4
KungFuMasterNoFrameskip-v4
MontezumaRevengeNoFrameskip-v4
MsPacmanNoFrameskip-v4
NameThisGameNoFrameskip-v4
PhoenixNoFrameskip-v4
PitfallNoFrameskip-v4
PongNoFrameskip-v4
PooyanNoFrameskip-v4
PrivateEyeNoFrameskip-v4
QbertNoFrameskip-v4
RiverRaidNoFrameskip-v4
RoadRunnerNoFrameskip-v4
RobotankNoFrameskip-v4
SeaquestNoFrameskip-v4
SkiingNoFrameskip-v4
SolarisNoFrameskip-v4
SpaceInvadersNoFrameskip-v4
StarGunnerNoFrameskip-v4
TennisNoFrameskip-v4
TimePilotNoFrameskip-v4
TutankhamNoFrameskip-v4
UpNDownNoFrameskip-v4
VentureNoFrameskip-v4
VideoPinballNoFrameskip-v4
WizardOfWorNoFrameskip-v4
YarsRevengeNoFrameskip-v4
ZaxxonNoFrameskip-v4
```

- [ ] **Step 3: Create hyperparams.yaml**

Create `benchmarks/configs/hyperparams.yaml`:

```yaml
# Schaul et al. 2016 PER paper hyperparameters
alpha: 0.6
beta_start: 0.4
beta_end: 1.0
epsilon: 1.0e-6
capacity: 1000000
batch_size: 32
learning_rate: 1.0e-4
gamma: 0.99
target_update_freq: 10000
total_steps: 50000000
learning_starts: 80000
train_frequency: 4
tau: 1.0
```

- [ ] **Step 4: Commit scaffold**

```bash
git add benchmarks/
git commit -m "feat: add benchmark scaffold, 57-game list, and Schaul 2016 hyperparams"
```

---

### Task 9: run_atari.py and compare.py

**Files:**
- Create: `benchmarks/run_atari.py`
- Create: `benchmarks/compare.py`

- [ ] **Step 1: Create run_atari.py**

Create `benchmarks/run_atari.py`:

```python
"""Resumable Atari benchmark runner.

Usage:
    python -m benchmarks.run_atari --buffer-type c --games BreakoutNoFrameskip-v4 --seeds 1
    python -m benchmarks.run_atari --buffer-type c --all-57 --seeds 1 2 3
    python -m benchmarks.run_atari --buffer-type uniform --all-57 --seeds 1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


CONFIGS = Path(__file__).parent / "configs"
RESULTS = Path(__file__).parent / "results"
GAMES_FILE = CONFIGS / "games.txt"


def load_games() -> list[str]:
    return [line.strip() for line in GAMES_FILE.read_text().splitlines() if line.strip()]


def result_path(game: str, buffer_type: str, seed: int) -> Path:
    return RESULTS / f"{game}_{buffer_type}_seed{seed}.json"


def run_one(game: str, buffer_type: str, seed: int, extra_args: list[str]) -> None:
    out_path = result_path(game, buffer_type, seed)
    if out_path.exists():
        print(f"[SKIP] {game} / {buffer_type} / seed={seed} — result exists")
        return

    print(f"[RUN]  {game} / {buffer_type} / seed={seed}")
    RESULTS.mkdir(exist_ok=True)

    cmd = [
        sys.executable, "-m", "benchmarks.atari_dqn",
        "--env-id", game,
        "--buffer-type", buffer_type,
        "--seed", str(seed),
        "--output", str(out_path),
    ] + extra_args

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[FAIL] {game} / {buffer_type} / seed={seed} — exit code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Atari PER benchmarks")
    parser.add_argument("--buffer-type", choices=["c", "python", "uniform"], required=True)
    parser.add_argument("--games", nargs="+", default=None)
    parser.add_argument("--all-57", action="store_true")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--total-steps", type=int, default=50_000_000)
    args = parser.parse_args()

    if args.all_57:
        games = load_games()
    elif args.games:
        games = args.games
    else:
        parser.error("Specify --games or --all-57")

    extra = ["--total-steps", str(args.total_steps)]

    for game in games:
        for seed in args.seeds:
            run_one(game, args.buffer_type, seed, extra)

    print(f"\nDone. Results in {RESULTS}/")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Create compare.py**

Create `benchmarks/compare.py`:

```python
"""Load benchmark results and print a comparison table + save a PNG chart.

Usage:
    python -m benchmarks.compare
    python -m benchmarks.compare --results-dir benchmarks/results/
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_results(results_dir: Path) -> list[dict]:
    results = []
    for p in sorted(results_dir.glob("*.json")):
        with open(p) as f:
            results.append(json.load(f))
    return results


def mean_final_reward(result: dict, n: int = 100) -> float:
    rewards = result.get("episode_rewards", [])
    if not rewards:
        return float("nan")
    return float(np.mean(rewards[-n:]))


def print_table(data: dict[str, dict[str, float]]) -> None:
    buffer_types = sorted({bt for game_d in data.values() for bt in game_d})
    header = f"{'Game':<40} " + "  ".join(f"{bt:>10}" for bt in buffer_types)
    print(header)
    print("-" * len(header))
    for game in sorted(data):
        row = f"{game:<40} "
        row += "  ".join(
            f"{data[game].get(bt, float('nan')):>10.1f}" for bt in buffer_types
        )
        print(row)


def save_png(data: dict, buffer_types: list[str], out_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping PNG. pip install reservoir[atari]")
        return

    games = sorted(data)
    x = np.arange(len(games))
    width = 0.8 / len(buffer_types)

    fig, ax = plt.subplots(figsize=(max(12, len(games) * 0.4), 6))
    for i, bt in enumerate(buffer_types):
        values = [data[g].get(bt, 0.0) for g in games]
        ax.bar(x + i * width, values, width, label=bt)

    ax.set_xticks(x + width * (len(buffer_types) - 1) / 2)
    ax.set_xticklabels([g.replace("NoFrameskip-v4", "") for g in games],
                       rotation=90, fontsize=7)
    ax.set_ylabel("Mean episode reward (last 100 eps)")
    ax.set_title("PER Buffer Comparison — Atari 57")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved chart to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path,
                        default=Path(__file__).parent / "results")
    args = parser.parse_args()

    results = load_results(args.results_dir)
    if not results:
        print(f"No results found in {args.results_dir}")
        return

    # Aggregate: game -> buffer_type -> mean final reward (averaged over seeds)
    raw: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for r in results:
        raw[r["game"]][r["buffer_type"]].append(mean_final_reward(r))

    data = {
        game: {bt: float(np.mean(vals)) for bt, vals in game_d.items()}
        for game, game_d in raw.items()
    }

    print_table(data)

    buffer_types = sorted({bt for game_d in data.values() for bt in game_d})
    save_png(data, buffer_types, args.results_dir / "comparison.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add benchmarks/run_atari.py benchmarks/compare.py
git commit -m "feat: add Atari benchmark runner and comparison tool"
```

---

### Task 10: Adapted CleanRL DQN (`atari_dqn.py`)

**Files:**
- Create: `benchmarks/atari_dqn.py`

- [ ] **Step 1: Download CleanRL source**

```bash
curl -sL "https://raw.githubusercontent.com/vwxyzjn/cleanrl/master/cleanrl/dqn_atari.py" \
  -o benchmarks/atari_dqn_upstream.py
```

- [ ] **Step 2: Create the adapted atari_dqn.py**

Create `benchmarks/atari_dqn.py` by adapting the downloaded file. The key modifications are:

1. Add Apache-2.0 attribution comment at top
2. Add `--buffer-type` and `--output` CLI arguments
3. Replace the `rb = ReplayBuffer(...)` construction with a factory function
4. Add JSON result logging at the end

Add this to the argument parser (find the `parser.add_argument` block):
```python
parser.add_argument("--buffer-type", type=str, default="uniform",
    choices=["c", "python", "uniform"],
    help="Replay buffer: c=CFastPERBuffer, python=PyFastPERBuffer, uniform=SB3 ReplayBuffer")
parser.add_argument("--output", type=str, default=None,
    help="Path to write JSON result file")
```

Replace the replay buffer construction (find `rb = ReplayBuffer`):
```python
if args.buffer_type == "uniform":
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )
elif args.buffer_type == "c":
    from reservoir.c_buffer import CFastPERBuffer
    obs_shape = envs.single_observation_space.shape
    rb = CFastPERBuffer(
        capacity=args.buffer_size,
        obs_shape=obs_shape,
        alpha=0.6, beta=0.4, device=str(device),
    )
    rb._is_per = True
else:  # python
    from reservoir.fast_buffer import FastPERBuffer as PyFastPERBuffer
    obs_shape = envs.single_observation_space.shape
    rb = PyFastPERBuffer(
        capacity=args.buffer_size,
        obs_shape=obs_shape,
        alpha=0.6, beta=0.4, device=str(device),
    )
    rb._is_per = True
```

Add result writing at the very end of `if __name__ == "__main__":`:
```python
if args.output:
    import json, time
    from pathlib import Path
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "game": args.env_id,
            "buffer_type": args.buffer_type,
            "seed": args.seed,
            "total_steps": args.total_timesteps,
            "episode_rewards": episodic_returns,
            "final_mean_reward_100ep": float(np.mean(episodic_returns[-100:])) if episodic_returns else 0.0,
        }, f, indent=2)
    print(f"Result written to {args.output}")
```

- [ ] **Step 3: Verify atari_dqn.py imports cleanly**

```bash
uv run python -c "import benchmarks.atari_dqn; print('OK')"
```

Expected: `OK` (may need `pip install reservoir[atari]` first if gymnasium[atari] not installed).

- [ ] **Step 4: Delete the upstream file**

```bash
rm benchmarks/atari_dqn_upstream.py
```

- [ ] **Step 5: Commit**

```bash
git add benchmarks/atari_dqn.py
git commit -m "feat: vendor and adapt CleanRL DQN with --buffer-type flag (Apache-2.0)"
```

---

## Phase 5: Smoke Test and PyPI Publish

### Task 11: Benchmark smoke run (3 games)

- [ ] **Step 1: Install Atari dependencies**

```bash
pip install -e ".[atari]"
```

Expected: installs ale-py, gymnasium[atari], autorom (accepts ROM licenses), matplotlib, stable-baselines3.

- [ ] **Step 2: Download Atari ROMs**

```bash
AutoROM --accept-license
```

- [ ] **Step 3: Run smoke benchmark (3 games, 1 seed, all 3 buffer types)**

```bash
for bt in c python uniform; do
  python -m benchmarks.run_atari \
    --buffer-type $bt \
    --games BreakoutNoFrameskip-v4 PongNoFrameskip-v4 SpaceInvadersNoFrameskip-v4 \
    --seeds 1 \
    --total-steps 500000
done
```

Note: `--total-steps 500000` (not 50M) for a quick smoke test that runs in minutes.

- [ ] **Step 4: Compare results**

```bash
python -m benchmarks.compare
```

Expected: a table showing rewards for 3 games × 3 buffer types. `c` and `python` columns should be within ~5% of each other (parity criterion).

- [ ] **Step 5: Commit results**

```bash
git add benchmarks/results/
git commit -m "bench: add smoke benchmark results (3 games, 500k steps)"
```

---

### Task 12: PyPI publish preparation

- [ ] **Step 1: Build wheel and sdist**

```bash
pip install build twine
python -m build
```

Expected: `dist/reservoir-0.2.0.tar.gz` and `dist/reservoir-0.2.0-cp313-*.whl`.

- [ ] **Step 2: Check the distribution**

```bash
twine check dist/*
```

Expected: `PASSED` for both files.

- [ ] **Step 3: Test install from wheel in a fresh venv**

```bash
python -m venv /tmp/test_reservoir_install
/tmp/test_reservoir_install/bin/pip install dist/reservoir-0.2.0-*.whl
/tmp/test_reservoir_install/bin/python -c \
  "import reservoir; print(reservoir.backend, reservoir.__version__)"
```

Expected: `c 0.2.0`

- [ ] **Step 4: Update README**

Add to `README.md` under a new `## Installation` section:

```markdown
## Installation

```bash
pip install reservoir          # installs with C extension (requires C compiler)
pip install reservoir[atari]   # + Atari benchmark dependencies
```

**Note on `[atari]`:** Installing with the `atari` extra automatically accepts Atari ROM
licenses via `autorom[accept-rom-license]`. Ensure you agree to these terms before installing.

### Verifying the C extension

```python
import reservoir
print(reservoir.backend)   # "c" if C extension built, "python" if fallback
```
```

- [ ] **Step 5: Publish to PyPI**

```bash
twine upload dist/*
```

This will prompt for PyPI credentials (or use a token: `twine upload --username __token__ dist/*`).

- [ ] **Step 6: Final commit and tag**

```bash
git add README.md
git commit -m "docs: add installation instructions and PyPI publish notes"
git tag v0.2.0
git push && git push --tags
```

---

## Quick Reference

```bash
# Build C extension
pip install -e . --no-build-isolation

# Run all tests (C + parity + existing 168)
uv run pytest tests/ -v

# Smoke benchmark (3 games, fast)
python -m benchmarks.run_atari --buffer-type c \
  --games BreakoutNoFrameskip-v4 PongNoFrameskip-v4 SpaceInvadersNoFrameskip-v4 \
  --seeds 1 --total-steps 500000

# Full 57-game run (needs GPU cluster)
python -m benchmarks.run_atari --buffer-type c --all-57 --seeds 1 2 3

# Compare results
python -m benchmarks.compare

# Build and check distribution
python -m build && twine check dist/*
```
