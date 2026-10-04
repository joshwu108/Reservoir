# Preregistration: deviations and run history

`docs/preregistration.md` is frozen and must not be edited. This file
records how the committed campaign runs relate to it.

## The run committed before 2026-10-04

The first committed `results/divergence_campaign_report.json` (campaign code
before 2026-10-04) declared "T2 DEAD" after a run that departed from the
preregistered protocol in four ways:

| preregistered | what was run |
|---|---|
| capacities 1024, 16384, 131072 | 512, 4096, 16384 |
| 10,000 workload programs per cell (270,000; 540,000 idiom evaluations) | 5 per cell (135) |
| draw mapped by exact rational scaling, rounded once | `(draw_int / exact_total) * float_total` in float64 |
| total-variation distance over all N positions, maximum over 100 draws | absolute probability difference at the sampled position only, 10 draws |

The "DEAD" verdict was therefore not supported by the preregistered
protocol, and the README sentence that cited it ("a pre-registered search
... found none") over-claimed. Both are corrected.

Two further departures of the same code, found in review on 2026-10-04:
the workload priorities were ``base`` times a hashed spread with 12.5% of
values near zero rather than the preregistered ``p_i = base *
magnitude_factor^(i mod 8)``; and the 100 draws were 64-bit hashes reduced
modulo a total that is 69 to 87 bits wide, so they only ever reached the
first few leaves of the tree. The minimal-reproducer protocol was not
implemented (no divergence has been found, so it has never been exercised).

## The protocol as implemented from 2026-10-04

`campaigns/divergence.py` now runs the preregistered capacities, maps draws
by exact rational scaling, checks 100 draws per workload for a
decision-relevant divergence, and computes the total-variation distance
`0.5 * sum_i |p_exact_i - p_float_i|` over every position. Nothing mutates
between the 100 draws of one workload, so the distribution is the same for
all of them and the distance is computed once per workload; the cell value
is the maximum over its workloads, as preregistered.

Workload priorities follow the preregistered ``p_i = base *
magnitude_factor^(i mod 8)`` with ``base`` in [1, 10]; each draw is exactly
uniform on ``[0, exact_total)`` through the library's keyed
rejection-sampling draw. A divergence, if found, is written with its full
workload parameters to `results/divergence_reproducers/`; the preregistered
shrinking step is not automated.

**Grid count.** The preregistration counts "10,000 workload programs per
(idiom × cell)", 540,000 in all. The code runs 10,000 programs per cell and
evaluates each against both idioms, which gives the same number of
idiom evaluations (540,000) from 270,000 distinct programs. Reports state
both figures.

`--workloads-per-cell` runs a fraction of the grid. The report records the
count run, and the preregistered kill rule is declared fired only on the
full 10,000 per cell. Any smaller run is reported as "not falsified in N of
270,000 workload programs" with no verdict.

Runtime of the full grid, measured on an Apple Silicon laptop: about 0.01 s
per workload at capacity 1024, 0.1 s at 16384 and 0.7 s at 131072, so about
28 CPU-hours for the grid, or 3 to 4 hours on 8 cores with `--jobs 8`.

## Committed runs

| date | workloads per cell | programs run | verdict |
|---|---|---|---|
| before 2026-10-04 | 5, wrong capacities, wrong TV and draw mapping | 135 | "DEAD" (withdrawn) |
| 2026-10-04, first pass | 200 | 5,400 of 270,000 | not falsified; withdrawn after review found the two departures above still present |
| 2026-10-04, final | 200 | 5,400 of 270,000 | not falsified in the reduced run; no verdict |

The full grid has not been run. Until it is, Reservoir reports: no
decision-relevant divergence and total-variation distance below 2^-40 in
the fraction of the preregistered grid that was run, and nothing more.
