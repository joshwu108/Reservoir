#include "sumtree.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- helpers ---- */

static Py_ssize_t next_power_of_two(Py_ssize_t n) {
    Py_ssize_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

/* ---- SumTree ---- */

SumTreeData *sumtree_create(Py_ssize_t capacity) {
    SumTreeData *t = (SumTreeData *)malloc(sizeof(SumTreeData));
    if (!t) return NULL;
    t->capacity      = capacity;
    t->tree_capacity = next_power_of_two(capacity);
    t->tree = (double *)calloc((size_t)(2 * t->tree_capacity), sizeof(double));
    if (!t->tree) { free(t); return NULL; }
    return t;
}

void sumtree_destroy(SumTreeData *t) {
    if (t) { free(t->tree); free(t); }
}

void sumtree_update(SumTreeData *t, Py_ssize_t pos, double value) {
    Py_ssize_t idx = t->tree_capacity - 1 + pos;
    t->tree[idx] = value;
    while (idx > 0) {
        idx = (idx - 1) >> 1;  /* parent */
        t->tree[idx] = t->tree[2*idx+1] + t->tree[2*idx+2];
    }
}

double sumtree_get(SumTreeData *t, Py_ssize_t pos) {
    return t->tree[t->tree_capacity - 1 + pos];
}

double sumtree_total(SumTreeData *t) {
    return t->tree[0];
}

Py_ssize_t sumtree_locate(SumTreeData *t, double value) {
    Py_ssize_t idx = 0;
    while (idx < t->tree_capacity - 1) {
        Py_ssize_t left = 2 * idx + 1;
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

MinTreeData *mintree_create(Py_ssize_t capacity) {
    MinTreeData *t = (MinTreeData *)malloc(sizeof(MinTreeData));
    if (!t) return NULL;
    t->capacity      = capacity;
    t->tree_capacity = next_power_of_two(capacity);
    Py_ssize_t n = 2 * t->tree_capacity;
    t->tree = (double *)malloc((size_t)n * sizeof(double));
    if (!t->tree) { free(t); return NULL; }
    for (Py_ssize_t i = 0; i < n; i++) t->tree[i] = INFINITY;
    return t;
}

void mintree_destroy(MinTreeData *t) {
    if (t) { free(t->tree); free(t); }
}

void mintree_update(MinTreeData *t, Py_ssize_t pos, double value) {
    Py_ssize_t idx = t->tree_capacity - 1 + pos;
    t->tree[idx] = value;
    while (idx > 0) {
        idx = (idx - 1) >> 1;
        double l = t->tree[2*idx+1], r = t->tree[2*idx+2];
        t->tree[idx] = (l < r) ? l : r;
    }
}

double mintree_get(MinTreeData *t, Py_ssize_t pos) {
    return t->tree[t->tree_capacity - 1 + pos];
}

double mintree_minimum(MinTreeData *t) {
    return t->tree[0];
}
