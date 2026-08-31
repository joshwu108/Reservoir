# Claude Code Setup Prompt — Project A: `reservoir`
## Certified Exact Prioritized Experience Replay with Crash Atomicity and Sampling Attestation

Copy everything below this line into Claude Code as your kickoff prompt.

---

You are building **`reservoir`**: a correctness-oriented research codebase for the **experience/data side of off-policy reinforcement learning**. The subject is prioritized experience replay (PER), which in every public implementation rests on a floating-point sum-tree: priorities are float TD-errors, internal nodes accumulate float sums, sampling walks the tree with float comparisons, and importance-sampling corrections are computed in float. Nobody has built an exact-arithmetic reference, nobody attests which transitions were sampled and why, nobody makes the buffer crash-atomic, and nobody has measured where the float trees actually make *decision-relevant* mistakes. You are building all four, in the falsification-first style of serious systems research: every claim independently checkable, every checker mutation-tested, every non-claim stated explicitly, every experiment preregistered with kill thresholds frozen before any data is collected.

## Research theses (all falsifiable — freeze kill rules in `docs/preregistration.md` before implementing any campaign)

**T1 (exactness).** Proportional PER (Schaul et al. 2016 semantics: p_i = |δ_i| + ε, P(i) = p_i^α / Σ_k p_k^α, IS weights w_i = (N·P(i))^(−β) / max_j w_j) can be implemented with **zero floats on any decision path**: priorities integerized exactly from their binary64 representations, α-exponentiation handled by an exact scheme you must design (see M1 for the required design discussion), tree sums as arbitrary-precision integers, sampling as a deterministic keyed draw of a uniform integer below the exact root total, and IS weights emitted as exact rationals. The resulting sampler's distribution is *exactly* the declared one.

**T2 (float divergence).** Faithful reimplementations of the two dominant float sum-tree idioms (the OpenAI-baselines-style segment tree and the labml/tutorial-style array sum-tree) produce **decision-relevant divergences** from the exact reference — i.e., matched-seed runs where a different transition index is returned, not merely a probability differing in low-order bits — under realistic adversarial-but-plausible priority workloads: many near-zero priorities plus a few dominant ones, long update histories causing accumulated stale-sum drift in internal nodes, priorities spanning ≥12 orders of magnitude, and capacity-wrap overwrite patterns. **Kill rule (freeze before running):** T2 is dead if a budgeted search — 10^5 workload programs per idiom, buffer capacities {2^10, 2^14, 2^17}, priority magnitude spans {10^3, 10^8, 10^12}, update:sample ratios {1:1, 10:1, 100:1} — finds zero decision-relevant divergences AND the exact-vs-float total-variation distance over sampled-index distributions (computed exactly against the float tree's *implied* distribution, which you can read off its node values) stays below 2^−40 in every cell. Report a negative result prominently if it dies; never soften thresholds after data exists.

**T3 (crash atomicity).** A durable replay buffer supporting insert, priority-update, capacity-wrap eviction, and batch-sample-with-attestation can be made **failure-atomic on a POSIX filesystem**: a `kill -9` at *any* instrumented cut leaves the buffer recoverable to a state that is exactly either the pre-operation or post-operation state — never a torn hybrid — verified by byte-exact comparison against oracle states, across every cut point, with fsync/rename discipline (fsync file, fsync parent directory, atomic rename commit).

**T4 (attestation).** Every sampled batch can carry a compact, hash-chained attestation record from which an **independent checker** — sharing no code with the buffer — re-derives and verifies: the exact tree total, each sampled index, each exact sampling probability (as reduced numerator/denominator), each exact IS weight, the draw values, and the chain linkage to the buffer's mutation history. A mutation campaign of ≥60 single-fault forgeries must be rejected at 100%.

## Hard constraints

- Python 3.10+, PyTorch 2.x **CPU only** (torch is used only for the tiny demo agent in M6; the buffer itself must be pure Python + `fractions`/integers). `uv` with committed lockfile. No GPU code anywhere.
- Exact arithmetic: Python integers and `fractions.Fraction` only on decision paths. Floats enter exactly once — as raw TD-error inputs — and are immediately converted exactly (every finite binary64 is a rational; reject inf/nan loudly). Document the exact conversion in `docs/design.md`.
- Randomness: keyed BLAKE2b block expansion + rejection sampling to draw uniform integers below exact totals, keyed by `(seed, buffer_id, op_counter)`. No `random`, no `numpy.random`, no `torch` RNG on decision paths. State explicitly in `docs/nonclaims.md` that this is deterministic replay identity, **not** a cryptographic or anti-tamper security boundary.
- Independent checkers live in `checker/`, import nothing from `src/` (CI-enforced with an import-graph assertion), and are written to *reject*: they recompute everything from first principles.
- **macOS durability rule:** this project is developed on macOS/APFS, where `fsync()` does not guarantee data reached durable storage. Every durability point must call `fcntl(fd, F_FULLFSYNC)` on Darwin (falling back to `os.fsync` on Linux, selected at runtime via `sys.platform`), including on parent directories after renames. Document in `docs/design.md` and `docs/nonclaims.md` that crash-atomicity evidence is established on APFS with `F_FULLFSYNC`; power-loss durability on other filesystems is untested. All `multiprocessing` uses the `spawn` start method explicitly.
- Crash injection uses real process kills: a subprocess driver runs buffer operations with instrumented cut points (environment-variable-selected), the parent `SIGKILL`s at each cut, then runs recovery and byte-compares. Simulated exceptions are not acceptable as the primary evidence; they may exist as fast smoke tests only.
- All tests deterministic; `hypothesis` allowed with a pinned derandomized profile.
- Every file under ~400 lines. `make check` runs everything.

## Repository layout

```
reservoir/
  pyproject.toml, uv.lock, Makefile, README.md, LICENSE (Apache-2.0)
  docs/
    design.md            # exact-arithmetic rules, α-exponentiation scheme, durability protocol, attestation schema
    preregistration.md   # frozen T2 search protocol + kill thresholds (committed BEFORE campaigns/)
    nonclaims.md         # what this repo does NOT establish
  spec/
    ReplayLifecycle.tla  # finite-state TLA+ safety model of the durable buffer (M5)
    check.sh             # pinned TLA+ tools download + model check
  src/reservoir/
    rational.py          # exact float→Fraction, integerization, exact power scheme
    sumtree.py           # exact integer sum-tree: insert, update, prefix-locate by integer compare
    draw.py              # keyed BLAKE2b uniform-below-N with rejection
    buffer.py            # in-memory exact PER: insert/update/evict/sample, IS weights as Fractions
    durable.py           # write-ahead durable buffer over POSIX: staged writes, fsync discipline, atomic rename commit, recovery
    attest.py            # hash-chained SampleAttestation + MutationRecord schemas, canonical serialization
  checker/
    verify.py            # independent verifier for attestation chains
  campaigns/
    float_baselines.py   # faithful float segment-tree and array sum-tree reimplementations (the "defendants"), each mirroring a named public idiom, with citations in comments
    divergence.py        # preregistered T2 search
    crash.py             # T3 kill-9 campaign across all cuts
    mutation.py          # T4 forgery campaign against checker/verify.py
  demo/
    tiny_dqn.py          # 5-state chain-MDP tabular-ish DQN on CPU proving the buffer drives real learning end-to-end
  tests/
  results/               # committed campaign reports with exact counts
```

## Milestones — strictly in order; each ends with `make check` green and a shown test transcript

**M1 — Exact core + the α-exponentiation design decision.** Before code: write the `docs/design.md` section resolving the central design problem — p_i^α for non-integer α (canonical α=0.6) is irrational, so "exact" needs a defined boundary. Present me **three options** with full correctness trade-offs: (a) restrict α to rationals with small denominator and use exact integer roots via integer Newton iteration with an exactness certificate (p^(3/5) via exact 5th-root-floor of p^3, documenting the floor semantics as *the declared distribution*); (b) compute p_i^α in float64 once, then exactly integerize the binary64 result and declare *that* integerized value the priority (float enters once, at a declared point, and everything downstream is exact — mirroring the honest-boundary style); (c) interval arithmetic with outward rounding and a certified tie-refinement loop that narrows intervals until the sampling decision is unambiguous. Recommend one (I expect (b) as default with (c) as stretch), wait for my choice, then implement `rational.py` and `sumtree.py`. The sum-tree must support insert, point-update, and prefix-sum locate with pure integer comparisons, plus a `verify_invariant()` that recomputes every internal node from leaves.

**M2 — Deterministic draw + in-memory buffer.** `draw.py` (uniform-below-N: covering power of two, rejection, document the uniformity argument in the docstring and test it by exhaustive enumeration of hash-block outcomes for tiny N). `buffer.py` composes tree + draw into full PER semantics including exact IS weights (max-weight normalization done by exact rational comparison via a parallel exact min-priority tree — note the standard implementations use a min-tree for this too; yours must be exact). Property tests: for tiny buffers (N ≤ 8) with fixed priorities, exhaustively enumerate all draw outcomes to a rejection depth and verify empirical-exact counts equal theoretical exact probabilities *as integers*, no statistics involved.

**M3 — Attestation + independent checker.** `attest.py`: every mutation (insert/update/evict) appends a MutationRecord (op, index, old/new integerized priority, op_counter, previous-record digest); every sampled batch appends a SampleAttestation (root total, per-sample: leaf index, draw integer, exact probability num/den, exact IS weight num/den, rejection count) chained into the same log. Canonical serialization: sorted-key JSON, integers as strings, BLAKE2b digests. `checker/verify.py` replays the entire chain from records alone — reconstructing the tree from mutation history and re-deriving every sample — and must reject on any inconsistency. Then run the **T4 mutation campaign**: ≥60 single-fault forgeries spanning digest bit-flips, off-by-one draw integers, swapped sampled indices with internally-consistent-but-chain-inconsistent probabilities, deleted mutation records, reordered records, replayed stale suffixes, and probability num/den pairs that are correct as reals but not in lowest terms where the schema demands reduced form. 100% rejection required; a surviving mutant is a checker bug — fix the checker, never delete the mutant. Commit the report with exact counts to `results/`.

**M4 — Durable buffer + crash campaign (T3).** `durable.py`: content-addressed segment files for transitions, a write-ahead intent record, full-durability sync (`F_FULLFSYNC` on Darwin, `fsync` elsewhere) on both file and parent directory, atomic rename as the single commit point, and a recovery routine that classifies any on-disk state as pre-commit or post-commit — never guessing. Instrument named cut points at minimum: after-intent-write, after-intent-fsync, mid-segment-write (torn write simulated by truncating at a byte offset then killing), after-segment-fsync, before-rename, after-rename-before-dir-fsync, after-dir-fsync. `campaigns/crash.py` spawns a child per (operation-type × cut), `SIGKILL`s it at the cut, recovers, and byte-compares the recovered logical state against both oracle states, asserting exact match with exactly one of them. Additionally verify the attestation chain remains verifiable across every recovery. Run the full matrix (4 op types × ≥7 cuts × ≥5 seeds) and commit the report.

**M5 — TLA+ model.** `spec/ReplayLifecycle.tla`: a finite-state safety model of the durable protocol — states for intent/segment/manifest visibility, a crash action enabled at every step, and a recovery action — checking two invariants: (I1) recovered state always equals pre-state or post-state of the interrupted op; (I2) the attestation chain head always corresponds to a committed state. Small finite scope (capacity 2, priorities from a 2-value set, ≤3 operations) is fine and must be documented as such. Include one **required semantic counterexample**: show that removing the parent-directory fsync admits a violating trace, and check in the violating config alongside the passing one. Pin the TLA+ tools release by digest in `spec/check.sh`.

**M6 — Preregistered float-divergence campaign (T2).** First commit `docs/preregistration.md` with the frozen search grid and kill thresholds from T2 above. Then `campaigns/float_baselines.py`: two faithful float reimplementations, each a separate class with comments citing which public idiom it mirrors and which implementation choices (float32 vs float64 nodes, whether updates repropagate full paths, epsilon handling) it copies. Then `campaigns/divergence.py`: generate workload programs (deterministic from a campaign seed), run exact and float implementations over matched keyed draws (map the exact draw integer into the float tree's [0, root_float) interval by exact rational scaling so the comparison is apples-to-apples — document this mapping carefully, it is the subtlest part of the campaign), record every decision-relevant divergence with a minimal reproducer (smallest prefix of the workload program that still diverges, found by automated shrinking), and compute the exact implied-distribution total-variation distance per cell. Commit the full report: divergence counts per cell, minimal reproducers, TV tables. If the kill rule fires, the negative result goes at the top of the README.

**M7 — End-to-end demo + README.** `demo/tiny_dqn.py`: a deliberately tiny DQN (linear Q-network, 5-state chain MDP, CPU, deterministic) trained twice from the same seed through the durable attested buffer — with a `SIGKILL` mid-training in one run — demonstrating that recovery plus deterministic draws yields **bitwise-identical final Q-network parameters and a fully verifiable attestation chain** across both runs. Then the README in full evidence-boundary style: implemented scope, thesis status (alive/dead per T1–T4 with links to committed reports), exact non-claims (not a performance result, not a distributed buffer, not a security boundary, not evidence about large-scale RL training quality, α-exponentiation boundary as declared in M1), and complete run instructions.

## Working rules

- After each milestone, show me the full test output before proceeding.
- Any change to the attestation schema, the α-exponentiation boundary, or a preregistered threshold requires my explicit approval — present options and wait.
- Never delete a failing test or surviving mutant to go green. If a thesis dies, the death is the result: document it, don't rescue it.
- When implementation and spec disagree, treat the spec as the claim under test — investigate which is wrong before touching either.

Start with M1: write the `docs/design.md` α-exponentiation section with the three options and your recommendation, and show it to me before writing any code.
