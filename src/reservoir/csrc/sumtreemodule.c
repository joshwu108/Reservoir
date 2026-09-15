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
    self->data = sumtree_create(capacity);
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
    if (pos < 0 || pos >= self->data->capacity) {
        PyErr_Format(PyExc_IndexError,
            "position %zd out of range [0, %zd)", pos, self->data->capacity);
        return NULL;
    }
    if (value < 0.0 || !isfinite(value)) {
        PyErr_SetString(PyExc_ValueError,
            "priority must be finite and non-negative");
        return NULL;
    }
    sumtree_update(self->data, pos, value);
    Py_RETURN_NONE;
}

static PyObject *
SumTree_get(SumTreeObject *self, PyObject *args)
{
    Py_ssize_t pos;
    if (!PyArg_ParseTuple(args, "n", &pos)) return NULL;
    if (pos < 0 || pos >= self->data->capacity) {
        PyErr_Format(PyExc_IndexError,
            "position %zd out of range [0, %zd)", pos, self->data->capacity);
        return NULL;
    }
    return PyFloat_FromDouble(sumtree_get(self->data, pos));
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
            PyErr_SetString(PyExc_ValueError,
                "draw value out of range [0, total)");
            Py_DECREF(result); return NULL;
        }
        Py_ssize_t pos = sumtree_locate(self->data, value);
        PyList_SET_ITEM(result, i, PyLong_FromSsize_t(pos));
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
    return PyLong_FromSsize_t(self->data->tree_capacity);
}

static PyMethodDef SumTree_methods[] = {
    {"update",       (PyCFunction)SumTree_update,       METH_VARARGS,
     "update(pos, value) -- set leaf priority (pre-exponentiated), propagate up"},
    {"get",          (PyCFunction)SumTree_get,          METH_VARARGS,
     "get(pos) -> float -- return leaf priority"},
    {"sample_batch", (PyCFunction)SumTree_sample_batch, METH_VARARGS,
     "sample_batch(values) -> list[int] -- locate positions for draw values"},
    {NULL, NULL, 0, NULL}
};

static PyGetSetDef SumTree_getsetters[] = {
    {"total",         (getter)SumTree_get_total,         NULL,
     "sum of all leaf priorities", NULL},
    {"tree_capacity", (getter)SumTree_get_tree_capacity, NULL,
     "internal tree capacity (next power of 2 >= capacity)", NULL},
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
    self->data = mintree_create(capacity);
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
    if (pos < 0 || pos >= self->data->capacity) {
        PyErr_Format(PyExc_IndexError,
            "position %zd out of range [0, %zd)", pos, self->data->capacity);
        return NULL;
    }
    if (value < 0.0 || !isfinite(value)) {
        PyErr_SetString(PyExc_ValueError,
            "priority must be finite and non-negative");
        return NULL;
    }
    mintree_update(self->data, pos, value);
    Py_RETURN_NONE;
}

static PyObject *
MinTree_get(MinTreeObject *self, PyObject *args)
{
    Py_ssize_t pos;
    if (!PyArg_ParseTuple(args, "n", &pos)) return NULL;
    if (pos < 0 || pos >= self->data->capacity) {
        PyErr_Format(PyExc_IndexError,
            "position %zd out of range [0, %zd)", pos, self->data->capacity);
        return NULL;
    }
    return PyFloat_FromDouble(mintree_get(self->data, pos));
}

static PyObject *
MinTree_get_minimum(MinTreeObject *self, void *closure)
{
    return PyFloat_FromDouble(mintree_minimum(self->data));
}

static PyObject *
MinTree_get_tree_capacity(MinTreeObject *self, void *closure)
{
    return PyLong_FromSsize_t(self->data->tree_capacity);
}

static PyMethodDef MinTree_methods[] = {
    {"update", (PyCFunction)MinTree_update, METH_VARARGS,
     "update(pos, value) -- set leaf value, propagate min upward"},
    {"get",    (PyCFunction)MinTree_get,    METH_VARARGS,
     "get(pos) -> float -- return leaf value"},
    {NULL, NULL, 0, NULL}
};

static PyGetSetDef MinTree_getsetters[] = {
    {"minimum",       (getter)MinTree_get_minimum,       NULL,
     "minimum leaf value (INFINITY if all slots empty)", NULL},
    {"tree_capacity", (getter)MinTree_get_tree_capacity, NULL,
     "internal tree capacity (next power of 2 >= capacity)", NULL},
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
