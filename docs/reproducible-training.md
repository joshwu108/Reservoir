# Reproducible and auditable replay for LLM RL

This note explains what Reservoir's attestation log establishes about a
training run, how to reproduce and verify one, and what the log can and
cannot be used for. It is the user-facing companion to `docs/design.md`
§4, §7 and §10 and to the limits in `docs/nonclaims.md`.

## The claim

A GRPO training run that replays rollouts through Reservoir produces a
**sampling transcript**: a hash-chained log of every example the buffer
stored (with a digest of its content and a source tag), every draw it made
(with the exact probability and importance weight), and every eviction.
Anyone holding the log can verify, without Reservoir's code or the model,
that the transcript is internally consistent and that every draw followed
from the recorded state. Holding the manifest as well, they can verify
*which* examples the transcript refers to.

Two runs with the same inputs produce byte-identical transcripts. When two
transcripts differ, the first differing record says why: different data,
different step cadence, different buffer parameters, or different draws.

Deterministic inference engines make the compute side of a run
reproducible. The transcript is the data side: which example was used
when, under what probability.

## The protocol

1. **Store.** `add_group(prompt_id, model_version, rollouts, source=...)`
   writes one `insert` record per rollout with the slot, the exact priority
   leaf, the age-decay inputs, the `content_digest` of
   `(prompt_id, tokens, reward)` and the `source`. With `manifest=`, the
   opening of the digest (the prompt id, tokens, reward and source
   themselves) is written to a second file, one line per insert.
2. **Draw.** `sample()` draws keyed BLAKE2b integers below the tree total
   and walks the exact sum-tree; one `sample` record holds every draw of
   the batch with its probability and importance weight as reduced
   fractions.
3. **Chain.** Each record's BLAKE2b-256 digest covers its content and the
   previous record's digest.
4. **Check.** `python -m checker.verify <log> [--manifest <manifest>]`
   rebuilds the tree from the mutation records, recomputes every decayed
   leaf and every digest, confirms that each draw lands on the recorded
   slot, that every sampled slot held a committed example, and, with the
   manifest, that every line opens its digest. The checker shares no code
   with the library.

## What is inside the log and what is not

Inside: every insert, update, eviction, version advance, rebase and sampled
batch of the buffer; the content digest and source of every stored
example; the exact probability and weight of every draw.

Outside: generation, reward computation, which groups were dead, the
optimizer. The log records what the buffer was *given* and what it *did*
with it. A transcript that differs between two runs at an `insert` record
means the generated data differed; the checker's `diff` says so and shows
that every draw before that point was identical.

The source tag is self-declared: the log commits to what the caller said
the source was, not to where the data truly came from.

## Running the demo

```bash
uv run python -m demo.reproducible_grpo            # three CPU runs, about a minute
```

This runs the Phase 2 GRPO integration (`ReservoirGRPOTrainer` on TRL's
tiny Qwen2 test model) three times: **a** and **b** with the same seeds,
**c** with a different data seed. The committed result
(`results/reproducible_grpo_report.json`, logs under
`benchmarks/modal/results/repro_cpu_12steps_seed42/`) shows a and b with
byte-identical logs and manifests and the same head digest, and a and c
first differing at record 1, an insert, classified `data`. The GPU
versions are `benchmarks/modal/reproducible_grpo_real.py` (HF generate on a
T4, with and without PyTorch's deterministic kernels) and
`benchmarks/modal/reproducible_grpo_vllm.py` (vLLM batch-invariant mode on
an A10G, on a 0.5B model); neither has been run yet.

## Reading a diff

```bash
python -m checker.diff a/attest.jsonl c/attest.jsonl
```

| class | meaning |
|---|---|
| `identical` | same records, same head digest |
| `config` | the `decay_config` records differ, or two sample records have identical draws and slots but different weights (a different `beta`) |
| `data` | the first differing record is an insert or update, the runs perform different operations at that point (other than the cases below), or two sample records have different sizes: the stored examples, their scores or the group sizes differed upstream; every draw before it was identical |
| `schedule` | the runs advanced through model versions on a different cadence |
| `sampler` | two sample records of the same size on an identical prefix differ: the draws differed on identical state. The seed is not in the log, so this is what two runs with different buffer seeds look like; with the same seed it would be a Reservoir defect |
| `internal` | an evict or rebase differs on identical state, or a rebase appears where the other log has a different record; both are deterministic and this must never happen |
| `truncated` | one log is a prefix of the other |

## Using the transcript for audits

```bash
python -m checker.transcript run/attest.jsonl --manifest run/manifest.jsonl --by source
python -m checker.transcript run/attest.jsonl --quota scraped=0 --quota licensed=50000
python -m checker.transcript run/attest.jsonl --find <content digest> --json report.json
```

- **Exposure.** For every committed example: how many times it was drawn,
  the exact sum of its importance weights, when it was first and last
  drawn, and every copy of it with the record that evicted it.
- **Mixture.** For every source: examples inserted, rows sampled, share of
  all sampled rows, overall and per model version.
- **Quota.** `--quota source=N` exits 2 if more than N sampled rows came
  from that source.
- **Find.** `--find <digest>` lists every sample record and batch position
  where that example appears, the question an unlearning audit asks.

Every number comes from the log alone; the manifest only adds the
human-readable example next to its digest. A log without content digests
(written before version 0.5.0, or by the classic transition buffers)
supports a per-slot view only, and `--quota` and `--find` refuse it rather
than pass silently.

## Relation to Verifiable Fine-Tuning

Verifiable Fine-Tuning (arXiv 2510.16830) proposes data commitments that
bind sources, preprocessing, licences and per-epoch quota counters to a
manifest; a verifiable sampler with public replayable and private
index-hiding batch selection; zero-knowledge update circuits; recursive
proof aggregation; and provenance binding of code identity.

Reservoir covers a subset of VFT's data-commitment and sampler components:
content digests and self-declared source tags opened by a manifest, and a
keyed, publicly replayable draw under a hash chain. The per-source counts
and quota verdicts are computed afterwards by the checker; they are not
committed counters, and there is no licence or preprocessing binding. There
is no proof that the parameter update used the sampled batch, no index
hiding, and no aggregation. Reservoir makes no claim about the update step.

## Relation to the EU AI Act, Article 53(1)(d)

Providers of general-purpose AI models must publish a summary of training
content following the Commission's template (in force since 2 August
2025). The template asks for a high-level account of modalities, sizes,
large public datasets and the kinds of licensed, scraped, user and
synthetic sources used. A Reservoir transcript is evidence a team can use
internally to produce and defend per-source statements about what the replay
buffer inserted and replayed during post-training. Fresh rollouts used
directly in a step, prompt selection, and any data outside the buffer are
not in the log. It is not the summary, and using Reservoir does not make a
provider compliant.

## Costs

From `results/attestation_overhead.json` (medians of three on an Apple
Silicon laptop, 2000 groups of 8 rollouts of 256 tokens, 200 batches of
64, pure-Python buffer):

| configuration | add_group, µs per rollout | sample, µs per draw | storage |
|---|---|---|---|
| no attestation | 32 | 15 | |
| in-memory log | 103 | 18 | |
| log file | 112 | 19 | 465 bytes per record |
| log file and manifest | 187 | 19 | plus 1643 bytes per inserted rollout |

Verifying a log with its manifest takes about 0.3 seconds per 10,000
records; building the transcript about 0.01 seconds per 10,000 records.
These are the costs of reproducibility and verification. Generation
dominates a GRPO step by orders of magnitude; no training-quality claim is
made for replay (`docs/nonclaims.md` §10, §13).

## Limits

- Without the manifest, a chain-consistent change to an insert's content
  digest or source is not detectable. The mutation campaign measures this:
  3 such forgeries survive the log-only check and all 3 are rejected with
  the manifest (`results/mutation_campaign_report.json`, `content_limit`).
- The log is a consistency-verification tool, not a security boundary
  (`docs/nonclaims.md` §4).
- Content digests are over token ids, so the same text under two
  tokenizers has two digests.
- Prompt selection (which prompts to generate rollouts for) is outside the
  log; `DatasetBuffer` is not attested.
- GPU runs are reproducible only when the inference engine is. No GPU
  result exists yet; the GPU demo script reports which case it observes
  when it is run.
