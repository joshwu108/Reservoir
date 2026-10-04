# Non-Claims

This document explicitly states what `reservoir` does NOT establish.

## What this repository does NOT claim

### 1. Performance

This is a **correctness-oriented research codebase**, not a performance result.
The exact-arithmetic buffer using Python integers and `fractions.Fraction` is
orders of magnitude slower than float-based implementations. This is expected
and intentional. No throughput, latency, or scalability claims are made.

### 2. Large-Scale RL Training Quality

We do not claim that using exact PER improves or changes RL training quality
at scale. The buffer correctness proofs apply to the sampling semantics only.
Whether exact sampling meaningfully affects agent performance in practice is
a separate empirical question not addressed here.

### 3. Distributed Buffers

This is a single-process, single-machine buffer. No claims about distributed
experience replay, multi-actor systems, or network-replicated buffers are made.

### 4. Security Boundary

The keyed BLAKE2b draw is **deterministic replay identity**, not a cryptographic
or anti-tamper security boundary. The attestation chain is a consistency
verification tool, not a tamper-proof audit log. A sufficiently privileged
adversary with filesystem access can forge attestation records by computing valid
BLAKE2b digests. We make no claims about security against such adversaries.

### 5. α-Exponentiation Semantics

The declared priority for transition i is proportional to the **binary64 value**
of `p_i^α`, not the true real value `p_i^α`. The float64-once integerization
boundary is declared in `docs/design.md`. The resulting distribution is exactly
the declared distribution (T1), but it differs from a hypothetical exact-real
distribution. This gap is bounded by the ULP error of float64 `pow()`, which
is at most 1 ULP ≈ 2^(-52) relative error on the exponentiated value.

### 6. Crash Durability on Non-APFS Filesystems

Crash-atomicity evidence is established on **macOS/APFS with `F_FULLFSYNC`**.
Power-loss durability on Linux (where `fsync()` semantics may differ by kernel
and filesystem), Windows, or other filesystems is **not tested** and not claimed.
The `os.fsync()` fallback on Linux is provided for correctness of the logical
protocol, not as evidence of physical durability on those platforms.

### 7. β Exponentiation for IS Weights

The IS weight computation uses the same float64-once boundary as α-exponentiation.
The declared IS weights are exact Fraction normalizations of float64 evaluations
of `(N·P(i))^(-β)`. The gap from the true real IS weights is bounded by 1 ULP
of float64 `pow()`.

### 8. TLA+ Model Scope

The TLA+ model (`spec/ReplayLifecycle.tla`) uses a finite scope (capacity 2,
2-value priority set, ≤3 operations). It establishes the safety properties
within that finite scope only. It does not constitute a proof for all possible
buffer sizes, priority values, or operation sequences. It is a falsification tool:
if the model checker finds a counterexample in the small scope, the protocol is wrong.
If it finds no counterexample, that is evidence (not proof) of correctness.


### 9. Age Decay Semantics

Age decay is base-2 with an integer half-life and two floor roundings at
declared points (design.md §7.1). It is not `p · exp(−Δ/τ)` evaluated in the
reals; FreshPER's τ is matched only approximately by `h ≈ τ · ln 2`. The
declared decayed distribution is exact; its closeness to any real-valued
decay is not claimed.

### 10. Replay and Training Quality

No claim is made that replaying rollouts with any of the shipped priority
strategies improves GRPO-style training. The strategies implement published
heuristics; the library guarantees only that the sampling distribution is
the declared one and is verifiable.

### 11. C Backend and Decayed Priorities

The C extension stores `double` priorities and does not support age decay,
versions, or the attested rollout path. `RolloutBuffer` is pure Python.

### 12. Durable Rollout Buffer Scope

`DurableRolloutBuffer` writes a full snapshot per operation. It is
crash-atomic (140/140 SIGKILL tests on macOS/APFS, see §6) but makes no
throughput claim, does not persist custom success predicates, and requires
JSON-serialisable rollout metadata.

### 13. Replay in TRL and Training Quality

`ReservoirGRPOTrainer` is not claimed to improve or stabilise GRPO training.
The runs under `benchmarks/modal/results/trl_replay_*` are integration
checks on a test-size model that cannot learn the task: they show that the
adapter survives TRL's real call path, that dead groups are replaced, and
that the attestation log verifies. Loss curves in those files are recorded,
not interpreted, and TRL's reward statistics (`reward`, `reward_std`,
`frac_reward_zero_std`) are computed before the hook replaces dead rows, so
they describe the generated batch rather than the batch trained on.

### 14. TRL Integration Scope

The adapter is tested against TRL 1.13.0 only, text-only, single process.
It does not handle tool masks, vLLM importance-sampling ratios, vision
inputs or multi-process training, and refuses them rather than guessing.
With vLLM generation it accepts a batch only when the importance-sampling
correction and off-policy masking are off, in which case it drops vLLM's
unused sampling logprobs from the batch; replayed rows carry no vLLM
logprobs.
Generation, reward computation and TRL's own row shuffling are outside the
attestation log; the log covers what the buffer stored, drew and evicted.
Behavior logprobs stored for replay are those of the training model at
generation time; when TRL does not compute them the adapter runs one extra
no-grad forward, which under dropout consumes RNG state, so a step without
replay is not promised to be bit-identical to a plain `GRPOTrainer` step.
Priorities are fixed at insertion; the adapter does not re-score replayed
rows from their training loss.

### 15. Relation to Verifiable Fine-Tuning

Reservoir implements the data-commitment component (content digests,
source tags, manifest, per-source exposure and quota counts) and the
public-replayable half of the verifiable-sampler component of Verifiable
Fine-Tuning (arXiv 2510.16830), and only a subset of those: there is no
licence or preprocessing binding, and quota counts are computed by the
checker after the fact rather than committed and enforced. It does not
implement that protocol's zero-knowledge update circuits, recursive proof
aggregation, provenance binding or index-hiding sampling, and it offers no
proof that the parameter update used the sampled batch.

### 16. EU AI Act Article 53(1)(d)

A transcript is evidence a team can use to produce and defend per-source
statements about what the replay buffer inserted and replayed. Fresh
rollouts used directly in a step, prompt selection and data outside the
buffer are not in the log. It is not the public summary
of training content the Commission's template requires, and using
Reservoir does not make a provider compliant with anything.

### 17. Content Commitments Need the Manifest to Be Opened

The log commits to each stored example through its content digest; a
verifier without the manifest can confirm the commitments are consistent
but not what they are commitments to. A chain-consistent change to an
insert's digest or source is not detectable from the log alone; the
mutation campaign measures and reports this. `source` is self-declared: the
log records what the caller said, not where the data truly came from.
Content digests are over token ids and so depend on the tokenizer.

### 18. Reproducibility Depends on the Engine

Two runs produce identical transcripts only when everything upstream of
the buffer (generation, rewards, dead groups) is identical. The CPU demo
shows this with HF ``generate`` under a fixed seed on a tiny model. On a
GPU, and with any engine that is not batch-invariant, the generated data
may differ between runs; ``checker.diff`` then locates the first insert
where it did and shows that Reservoir's draws were identical before it,
but Reservoir does not make the engine deterministic.

### 19. Attestation Overhead

The numbers in `results/attestation_overhead.json` measure the pure-Python
buffer on one laptop and describe the relative cost of attestation and
verification. They are not throughput claims (§1) and say nothing about
training quality (§10, §13).
