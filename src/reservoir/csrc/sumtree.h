#ifndef RESERVOIR_SUMTREE_H
#define RESERVOIR_SUMTREE_H

#include <stddef.h>
#include <Python.h>

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

SumTreeData *sumtree_create(Py_ssize_t capacity);
void         sumtree_destroy(SumTreeData *t);
void         sumtree_update(SumTreeData *t, Py_ssize_t pos, double value);
double       sumtree_get(SumTreeData *t, Py_ssize_t pos);
double       sumtree_total(SumTreeData *t);
Py_ssize_t   sumtree_locate(SumTreeData *t, double value);

MinTreeData *mintree_create(Py_ssize_t capacity);
void         mintree_destroy(MinTreeData *t);
void         mintree_update(MinTreeData *t, Py_ssize_t pos, double value);
double       mintree_get(MinTreeData *t, Py_ssize_t pos);
double       mintree_minimum(MinTreeData *t);

#endif /* RESERVOIR_SUMTREE_H */
