# Changelog

All notable changes to `reservoir-replay`. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/) with the pre-1.0 convention that a
minor bump may change interfaces.

## [0.6.0] - 2026-10-06

Phase 5: the developer-facing release. Everything below is implemented,
tested (about 1,880 tests) and, where it makes a claim, backed by a committed
result; `docs/nonclaims.md` lists what is not claimed.

### Added

- **Batch witness.** After every replayed step the adapter re-reads each
  replaced row against the sampled rollout (`verify_written_rows`) and
  writes a `batch` record naming which training-batch row holds which draw,
  with a digest of the final tensors. The checker proves each witnessed row
  holds the example its draw selected; `reservoir-transcript --explain STEP
  ROW` says why a row is in the batch. Mutation category `batch`.
- **Replay health telemetry and drift gate.** Per step: replay fraction,
  dead and near-dead groups, effective sample size and staleness of the
  replayed rows (exact, recomputed by the checker), and the sequence
  log-ratio between stored and current logprobs (reported). Logged under
  `reservoir/*` in the trainer's metrics and as a `telemetry` record. An
  optional drift gate (`max_log_ratio=`, `max_declines_per_step=`), off by
  default, declines drifted rows with an `evict reason=drift` record, never
  silently. Mutation category `telemetry`.
- **Priority rescore hook.** `PriorityStrategy.rescore(rollout, group,
  signal)` and `RolloutBuffer.update_priorities` from the loss step, mapping
  rows to slots through the batch witness. No new strategy ships.
- **Durable command log and checkpoint binding.** `DurableRolloutBuffer`
  appends each operation's inputs to `wal.jsonl` and snapshots every
  `compact_every` commands through the crash-atomic protocol; recovery
  replays the log onto the snapshot, cuts a torn tail and refuses mid-log
  damage. `checkpoint(tag)` / `restore_checkpoint(tag)` bind the buffer to a
  trainer checkpoint; a restore starts a new log epoch. The TRL adapter
  snapshots on every save and rewinds on resume, failing closed when the
  snapshot is missing. Crash campaign: 315/315 cut points on macOS/APFS, 0 torn.
- **Multi-process TRL adapter.** Under `accelerate` with more than one
  process, rank 0 owns the buffer and the single log writer; behavior
  logprobs are computed per rank, slices gathered, the single-process hook
  run on the global batch, the result broadcast. One gather and one
  broadcast per hook call; a failure on any rank is raised on every rank.
  Verified on two A10G GPUs (`benchmarks/modal/results/trl_replay_*_a10g-x2.*`).
- **Quarantine and blast radius.** `buffer.quarantine(predicate, reason)`
  evicts every matching entry with a record carrying the predicate text and
  the reason; `reservoir-transcript --blast-radius DIGEST` lists every step
  and training-batch row the example (or, with the manifest, its prompt)
  reached. Mutation category `provenance`.
- **Reward provenance.** Per-reward-function values and the function names
  travel in insert metadata and in the manifest (`rewards`), numeric only
  and outside the content digest, so "verifier high, judge low" can be
  queried after the fact. The TRL adapter captures them from the trainer's
  `_calculate_rewards`; the verl adapter records the summed score only.
  The committed GPU manifests were produced before this wiring and have no
  `rewards` field.
- **Offline replay.** `reservoir-replay-offline` rebuilds every witnessed
  batch as content (rows, draws, exact importance weights, tokens, rewards,
  sources) from the log and manifest alone, importing nothing from the
  library; two replays of one run are byte-identical. Mutation category
  `replay_manifest`.
- **verl adapter.** `reservoir.integrations.verl` replays dead GRPO groups
  in verl's `DataProto` trainer (`RayPPOTrainer`, `trainer.use_v1=false`),
  pinned to verl 0.9.1, with the same contract as the TRL adapter. Verified
  on one T4 (`benchmarks/modal/results/verl_replay_*`).
- Console scripts `reservoir-verify`, `reservoir-transcript`,
  `reservoir-diff`, `reservoir-replay-offline`; the checker ships in the
  wheel as the sibling package `reservoir_checker` (the top-level `checker/`
  package re-exports it for `python -m checker.*`).
- GitHub Actions: tests on Ubuntu and macOS, the import-isolation check,
  the forgery campaign and the TLA+ model check (including the
  no-parent-fsync counterexample).
- GPU evidence under `benchmarks/modal/results/`: the reproducibility
  triplet on a T4 with HF generation (`IDENTICAL` with and without
  deterministic kernels), the two-GPU TRL run and the verl run, each
  re-verified by the test suite.

### Changed

- **Distribution renamed** from `reservoir` to `reservoir-replay` (the PyPI
  name `reservoir` belongs to an unrelated package). The import name stays
  `reservoir`.
- **No required dependencies.** `import reservoir`, the rollout buffer, the
  attestation log and the checker need the standard library only; numpy
  and torch come with the extras that use them (`classic`, `trl`, `verl`,
  `prefcheck`, `anchor`) and resolve lazily with a message naming the
  extra. Enforced by `tests/test_import_light.py`.
- **Attestation log format 3.** The log records `seed`, `buffer_id` and
  `beta` (a draw moved inside its leaf's range, or a reweighted sample, is
  invisible to a log without them), `batch` and `telemetry` records, and
  `quarantine` records. The checker reads formats 1 to 3 and refuses a
  quarantine record in an older format.
- The dead-group criterion is "advantages exactly zero" (exactly "zero
  gradient under the loss"); a `near_dead_groups` counter makes groups whose
  advantages are a float residue visible. In verl, an equal-score group with
  such a residue is counted as near-dead.
- `audit.py` no longer reports a divergence count it never computed; the
  divergence campaign is the decision-relevant measurement.
- Checker, transcript and diff documentation moved to `docs/design.md` §9
  to §12 and `docs/nonclaims.md` §13 to §24.

### Fixed

Review round of 2026-10-06 (five independent reviews over the release
tree; every item below has a regression test in
`tests/test_review_round_0_6.py`):

- Checker: integer fields given as floats or negatives are refused instead
  of truncated (`0.7` no longer verifies as `0`); every malformed field
  surfaces as `CheckerError` from the library, not a raw Python error; a
  `decay_config` may not declare more than 2^24 slots and a witness more
  than 2^20 rows, so a one-line log can no longer make the checker
  allocate unbounded memory; the CLIs report `MemoryError` as a failure.
- Durable buffer: the crash-test cut points need `RESERVOIR_CRASH_TEST=1`
  in addition to `RESERVOIR_CUT_POINT`, so a stray name in a launcher's
  environment cannot kill a run; the checkpoint a buffer last restored
  from is never pruned; slot positions given to the durable quarantine
  must be plain ints.
- Buffer: rollout metadata is deep-copied, so a caller mutating a nested
  value after `add_group` cannot make live and replayed state differ;
  quarantine on a buffer without a log warns that no record is kept.
- TRL adapter: reward provenance is wired (it was accepted by the row
  converter but never supplied); owner-only steps (building the buffer,
  binding and restoring checkpoints) broadcast their outcome so a failure
  on rank 0 raises on every rank instead of hanging the others; sharded
  strategies (DeepSpeed, FSDP, Megatron) are refused because the
  owner-only telemetry forward would deadlock under them; `max_log_ratio`
  must be finite and `max_declines_per_step` a non-negative int; a NaN
  under the completion mask no longer poisons a row's log-ratio (TRL and
  verl); the Modal result writer raises when the checker rejects a log.
- Transcript: `--blast-radius` reports `declined_draws` separately.
- Docs reconciled with the evidence files (demo output, crash and test
  counts, version attributions, the TLA+ state count, CI status).

- Crash and divergence campaigns report what they measured (cut points that
  never fired are failures, not passes; the reduced divergence grid is
  labelled as such).
- The multi-process adapter moved CPU shards to rank 0's device before the
  telemetry forward (found by the first real two-GPU run).
- A durable buffer opened without attestation no longer misreads an empty
  log's format and refuses a replayed quarantine.
- Resuming a verl run no longer trips the hook-ran guard for the steps the
  loaded checkpoint already contained.

### Not in this release

- The TRL vLLM server-mode reproducibility tier (blocked on an NCCL
  weight-sync hang, `docs/reproducible-training.md`).
- A verl adapter for the TransferQueue-based V1 trainer (`docs/design.md`
  §12.1).
- Stable-Baselines3 and TorchRL adapters for the classic buffers.

## [0.5.0] - 2026-10-03

Phase 4: content commitments (BLAKE2b content digests on every insert, the
manifest that opens them), `checker.transcript` and `checker.diff`, the
reproducible GRPO demo, and the rollout buffer with exact age decay, keyed
deterministic draws and the attestation log checked by an independent
checker. Earlier phases: the exact prioritized replay buffer with crash-atomic
durability (TLA+ model of the lifecycle protocol), the C-backed
`FastPERBuffer` with n-step and HER wrappers, and the fine-tuning tools
(preference noise detection, forgetting monitor).
