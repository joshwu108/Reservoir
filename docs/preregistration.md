# Preregistration — T2 Float-Divergence Campaign

**FROZEN BEFORE CAMPAIGN RUNS. DO NOT MODIFY AFTER CAMPAIGN DATA EXISTS.**

This document is committed before `campaigns/divergence.py` is run.
Thresholds are not adjusted after seeing any data.

---

## Thesis T2 (verbatim from spec)

Faithful reimplementations of the two dominant float sum-tree idioms produce
**decision-relevant divergences** from the exact reference — i.e., matched-seed
runs where a different transition index is returned — under realistic
adversarial-but-plausible priority workloads.

## Search Grid

The preregistered search exhausts the following parameter grid:

| Dimension | Values |
|-----------|--------|
| Buffer capacities | 2^10 (1024), 2^14 (16384), 2^17 (131072) |
| Priority magnitude spans | 10^3, 10^8, 10^12 |
| Update:sample ratios | 1:1, 10:1, 100:1 |
| Workload programs per (idiom × cell) | 10,000 |
| Float idioms | OpenAI-baselines segment-tree, labml/tutorial array sum-tree |

Total workload programs: 2 idioms × 3 capacities × 3 magnitude_spans × 3 ratios × 10,000 = 540,000

## Kill Rule (T2 dies if ALL of the following hold)

T2 is **dead** (thesis falsified) if:
1. Zero decision-relevant divergences are found across the entire search grid, AND
2. The exact-vs-float total-variation distance over sampled-index distributions
   stays below **2^−40** in every (capacity × magnitude_span × ratio) cell.

T2 is **alive** if any single divergence is found OR any TV distance ≥ 2^−40.

## Negative Result Protocol

If T2 dies, the negative result is reported prominently at the top of README.md.
Thresholds are NOT softened after any data exists.

## Decision-Relevant Divergence Definition

A divergence is "decision-relevant" if:
- The exact buffer and the float buffer, given the **same draw integer** (mapped
  correctly to each tree's scale), return a **different transition index**.
- This is not merely a probability differing in low-order bits; it is a case
  where the sampling decision changes.

The mapping from an exact draw integer to the float tree's [0, root_float) interval
is done via exact rational scaling:
    float_draw = float(Fraction(draw_int, exact_total) * Fraction(float_total))
where float_total is the float tree's root sum. This ensures the comparison is
apples-to-apples. See `campaigns/divergence.py` for the exact implementation.

## Workload Program Structure

A workload program is a sequence of operations:
1. Insert N transitions with priorities drawn from the magnitude span
2. Update M priorities (deterministic, keyed by campaign seed)
3. Sample K transitions using matched draw integers

Priority generation: `p_i = base * magnitude_factor^(i mod span_levels)`
where `base` is randomly drawn from [1, 10] and `span_levels` is 8.

## Minimal Reproducer Protocol

For every divergence found:
1. Record the full workload program
2. Shrink: find the shortest prefix of operations that still produces the divergence
3. Commit the minimal reproducer to `results/divergence_reproducers/`

## TV Distance Computation

For each (capacity × magnitude_span × ratio) cell:
1. Run 100 samples from the exact buffer
2. For each sample, compute the exact sampling distribution over all N positions
3. Compute the total-variation distance between the exact distribution and
   the float tree's *implied* distribution (read off from float node values)
4. Report the maximum TV distance across all 100 samples in the cell

The implied float distribution: float_prob[i] = float_tree.leaf[i] / float_tree.root_sum

---

**FROZEN:** This document must not be modified after campaign data is collected.
Commit hash of this file is the preregistration timestamp.
