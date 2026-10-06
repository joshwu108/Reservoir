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
of float64 `pow()`. When a log records `beta`, the checker recomputes the
weights with its own float64 `pow()`; the two agree on one platform, and on
a platform whose C library rounds `pow()` differently the checker reports a
mismatch rather than accept it. The declared value is the producer's.

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

`DurableRolloutBuffer` logs each operation's inputs and snapshots every
`compact_every` operations; recovery replays the log, which is exact
because every operation is deterministic. It is crash-atomic (SIGKILL
tests on macOS/APFS at every log and snapshot cut, see §6) but makes no
throughput claim, does not persist custom success predicates, and requires
JSON-serialisable rollout metadata. Per-operation cost is flat but
compaction and reopen cost grow with the attestation log, which the
snapshot carries (`results/durable_overhead.json`); a run long enough for
that to matter should raise `compact_every` or rotate the log, neither of
which has been measured. The attestation and manifest files are written as
each operation runs, before its command is fsynced, so a reader of those
files during a crash window can see a record that recovery then retracts;
only after reopen are the files and the buffer guaranteed to agree.
Checkpoint binding rewinds the buffer to the trainer's step; it does not
verify that the model checkpoint is the one the buffer was bound to, and
it requires a durable buffer (an in-memory `ReservoirReplay` refuses to
resume at a non-zero step rather than continue from an empty buffer). A
trainer checkpoint written in the window before the buffer's `on_save`
ran has no buffer checkpoint and resume fails closed. With
`steps_per_generation > 1` a checkpoint taken inside a generation window
holds the buffer operations of that window; the resumed trainer
regenerates the window, so the buffer samples it a second time. The chain
records both samples and verifies; it is not the chain an uninterrupted
run would have produced. Untested against a real resume.

### 13. Replay in TRL and Training Quality

`ReservoirGRPOTrainer` is not claimed to improve or stabilise GRPO training.
The runs under `benchmarks/modal/results/trl_replay_*` are integration
checks on a test-size model that cannot learn the task: they show that the
adapter survives TRL's real call path, that dead groups are replaced, and
that the attestation log verifies. Loss curves in those files are recorded,
not interpreted, and TRL's reward statistics (`reward`, `reward_std`,
`frac_reward_zero_std`) are computed before the hook replaces dead rows, so
they describe the generated batch rather than the batch trained on.

The batch witness (version 0.5.0) closes the gap between the sampled batch
and the batch the adapter hands back up to the adapter boundary. The
adapter checks the written rows against the sampled rollouts and refuses
otherwise; the checker proves the adapter's declared row-to-draw mapping
consistent with the sample record. The checker cannot see the tensors:
the tensor commitment can be opened only by someone holding the batch, and
nothing in the log ties it to the declared rows. TRL's later row shuffle
and the optimizer are outside.

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
Priorities are fixed at insertion unless the priority strategy implements
`rescore`, which the adapter calls at placement time with the weighted
advantage, importance weight, log-ratio and age; the training loss itself
is not available to it. No rescoring strategy is shipped, and no claim is
made that any rescoring improves training.

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
shows this with HF ``generate`` under a fixed seed on a tiny model, and
one T4 run of the same demo was identical too, with and without
PyTorch's deterministic kernels (`benchmarks/modal/results/repro_hf_t4_*`).
That is an observation about a tiny model and twelve steps. On a GPU, and
with any engine that is not batch-invariant, the generated data may
differ between runs; ``checker.diff`` then locates the first insert where
it did and shows that Reservoir's draws were identical before it, but
Reservoir does not make the engine deterministic. vLLM's batch-invariant
mode cannot share a process with the trainer (its kernels have no
backward), so the vLLM tier uses TRL's server mode on a second GPU.

### 19. Attestation Overhead

The numbers in `results/attestation_overhead.json` measure the pure-Python
buffer on one laptop and describe the relative cost of attestation and
verification. They are not throughput claims (§1) and say nothing about
training quality (§10, §13).


### 20. Telemetry

The effective sample size and staleness in a telemetry record are
recomputed by the checker; the log-ratio statistics are carried from the
adapter and listed under `reported`, and nothing verifies them. No claim
is made that any threshold on them improves training; the drift gate is
off by default and, when on, only makes its declines visible.

### 21. More Than One Process

The multi-process path of the TRL adapter is tested with a fake
accelerator of two and four ranks in one interpreter and by parity with
the single-process adapter on the concatenated batch. It has not been run
on a real `torch.distributed` process group or on more than one GPU; the
`accelerate` collectives it wraps are called as documented and nothing
more is claimed about them. Rank 0 owns the buffer and the log, and the
log records the global batch in rank order; it does not record which rank
generated or trained which row. The rank-0 ownership rule relies on the
launcher setting `RANK` (or a non-zero `LOCAL_RANK`), or on the trainer
attaching the accelerator before the buffer is first used; a log,
manifest or directory is not opened before then, but a `.buffer` read on
another rank before the attach opens it there, and the attach then refuses
on every rank after the file was touched. Behavior logprobs come from each rank's own forward, so
the dropout caveat of §14 applies on every rank. No claim is made about
throughput: every rank receives every rank's slice in the all-gather and
the whole rewritten batch in the broadcast, once per generation step.

### 21. Quarantine and Reward Provenance

A quarantine eviction records the predicate text and the operator's note;
the checker verifies that the record is a well-formed eviction of a live
slot, not that the text describes the predicate that ran or that the
reason is true. The blast radius lists what the batch witnesses saw: rows
replayed from the buffer. The step at which an example was generated and
trained fresh is the `entry_version` of its insert (the trainer's step in
the TRL adapter), which the transcript reports; the row it occupied then
is not in the log. The manifest's per-reward-function values are the
adapter's statement, numeric only and outside the content digest: the log
does not commit to them, a changed or dropped value passes the checker
(the mutation campaign measures this), and the adapter wiring that
supplies them to the row conversion is not yet in `ReservoirReplay`.
