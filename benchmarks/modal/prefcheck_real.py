"""benchmarks/modal/prefcheck_real.py

Real-data benchmark for reservoir-prefcheck.

Loads pairs from Dahoas/rm-static (deduplicated by prompt, degenerate pairs
dropped), injects label noise (flips chosen/rejected for a known subset),
trains a reward model, and detects the flips from per-example trajectories.

Measurement design
------------------
Trajectories come from periodic EVAL PASSES (model.eval(), no dropout,
whole dataset scored at the same model state), not from on-the-fly training
losses. Each checkpoint logs per-example loss AND margin r_chosen−r_rejected;
mean margin across checkpoints is the AUM-style signal from the noisy-label
literature. On-the-fly training losses are still logged as the baseline the
eval-pass method must beat.

Caveat: rm-static's human labels are themselves noisy (~60-75% annotator
agreement), so "true-CLEAN" contains naturally mislabeled pairs. Measured
FLIPPED precision is therefore a LOWER BOUND on real precision.

Usage
-----
    modal run benchmarks/modal/prefcheck_real.py                       # 3 seeds
    modal run benchmarks/modal/prefcheck_real.py --seeds 42            # 1 seed
    modal run benchmarks/modal/prefcheck_real.py --n-pairs 5000 --n-epochs 3

Per-seed results (including per-example trajectories for free local
re-analysis) are written to benchmarks/modal/results/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal image — pin versions so the build is reproducible
# ---------------------------------------------------------------------------

_DEPS = [
    "torch==2.2.2",
    "transformers==4.40.2",
    "datasets==2.19.1",
    "numpy==1.26.4",
    "accelerate==0.30.1",
    "sentencepiece==0.2.0",   # DeBERTa-v3 tokenizer
    "protobuf==4.25.3",
]

try:
    _repo_src = str(Path(__file__).parents[2] / "src")
except IndexError:
    _repo_src = "."  # running inside container; image is already built

hf_cache = modal.Volume.from_name("reservoir-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*_DEPS)
    .env({"HF_HOME": "/hf_cache", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
    .add_local_dir(_repo_src, remote_path="/reservoir_src")
)

app = modal.App("reservoir-prefcheck-real")

# ---------------------------------------------------------------------------
# Pure helpers (unit-tested locally; run inside the container too)
# ---------------------------------------------------------------------------

def _rankdata_average(a):
    """Average-tie ranks (1-based), equivalent to scipy.stats.rankdata."""
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    sorter = np.argsort(a, kind="mergesort")
    inv = np.empty_like(sorter)
    inv[sorter] = np.arange(len(a))
    a_sorted = a[sorter]
    group_starts = np.r_[True, a_sorted[1:] != a_sorted[:-1]]
    dense = group_starts.cumsum()[inv]
    boundaries = np.r_[np.nonzero(group_starts)[0], len(a)]
    return 0.5 * (boundaries[dense] + boundaries[dense - 1] + 1)


def separation_stats(pos, neg) -> dict:
    """Distribution stats + AUROC for one feature, split by ground truth.

    pos = feature values for true-FLIPPED examples, neg = true-CLEAN.
    AUROC 1.0 means the feature perfectly ranks flipped above clean;
    0.5 means no signal; ~0.0 means the ranking is inverted.
    """
    import numpy as np

    pos = np.asarray(pos, dtype=np.float64)
    neg = np.asarray(neg, dtype=np.float64)

    def _group(v) -> dict:
        if len(v) == 0:
            return {"n": 0, "mean": None, "median": None, "p25": None, "p75": None}
        return {
            "n": int(len(v)),
            "mean": float(np.mean(v)),
            "median": float(np.median(v)),
            "p25": float(np.percentile(v, 25)),
            "p75": float(np.percentile(v, 75)),
        }

    if len(pos) == 0 or len(neg) == 0:
        auroc = None
    else:
        ranks = _rankdata_average(np.concatenate([pos, neg]))
        rank_sum_pos = float(ranks[: len(pos)].sum())
        auroc = (rank_sum_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))

    return {"auroc": auroc, "pos": _group(pos), "neg": _group(neg)}


def dedup_by_prompt(prompts) -> list:
    """Indices of the first occurrence of each distinct prompt, in order."""
    seen: set = set()
    keep = []
    for i, p in enumerate(prompts):
        if p not in seen:
            seen.add(p)
            keep.append(i)
    return keep


def encode_pair_consistent(tokenizer, prompt: str, chosen: str, rejected: str,
                           max_len: int) -> tuple[list, list]:
    """Encode both sides of a pair with IDENTICAL prompt context.

    The full response is always retained for both sides (truncated only if a
    response alone exceeds the budget); the prompt is left-truncated once,
    using the same tail for chosen and rejected. This avoids both failure
    modes of naive truncation: losing the response entirely (right-trunc)
    and scoring the two sides against different visible context (left-trunc
    of the combined string).
    """
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    budget = max_len - 2  # [CLS] ... [SEP]

    p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    c_ids = tokenizer(chosen, add_special_tokens=False)["input_ids"]
    r_ids = tokenizer(rejected, add_special_tokens=False)["input_ids"]

    resp_budget = max(len(c_ids), len(r_ids))
    if resp_budget > budget:
        c_ids = c_ids[:budget]
        r_ids = r_ids[:budget]
        resp_budget = max(len(c_ids), len(r_ids))

    prompt_keep = budget - resp_budget
    p_tail = p_ids[len(p_ids) - prompt_keep:] if prompt_keep > 0 else []

    return (
        [cls_id] + p_tail + c_ids + [sep_id],
        [cls_id] + p_tail + r_ids + [sep_id],
    )


def pad_batch(id_lists, pad_id: int):
    """Pad variable-length id lists to the batch max; returns (ids, mask)."""
    import torch

    max_len = max(len(x) for x in id_lists)
    ids = torch.full((len(id_lists), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(id_lists), max_len), dtype=torch.long)
    for i, x in enumerate(id_lists):
        ids[i, : len(x)] = torch.tensor(x, dtype=torch.long)
        mask[i, : len(x)] = 1
    return ids, mask


# ---------------------------------------------------------------------------
# GPU function
# ---------------------------------------------------------------------------

@app.function(
    gpu="A10G",
    image=image,
    timeout=7200,
    volumes={"/hf_cache": hf_cache},
)
def run_prefcheck(
    n_pairs: int = 5000,
    noise_rate: float = 0.20,
    n_epochs: int = 3,
    batch_size: int = 16,
    seed: int = 42,
    model_name: str = "distilbert-base-uncased",
    evals_per_epoch: int = 2,
    max_len: int = 256,
    lr: float = 2e-5,
) -> dict:
    """Train a reward model on noisy preference data and detect the noise."""
    import math
    import random as _random

    import numpy as np
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from torch.optim import AdamW
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    sys.path.insert(0, "/reservoir_src")
    from reservoir.report import PreferenceQualityReport
    from reservoir.trajectory import TrajectoryLogger

    # Determinism (best-effort on GPU; CUBLAS_WORKSPACE_CONFIG set in image env)
    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[seed={seed}] Device: {device}")

    # -----------------------------------------------------------------------
    # Load dataset, deduplicate prompts, encode, drop degenerate pairs
    # -----------------------------------------------------------------------
    print(f"[seed={seed}] Loading Dahoas/rm-static...")
    raw = load_dataset("Dahoas/rm-static", split="train")
    unique_idx = dedup_by_prompt(raw["prompt"])
    n_dup_dropped = len(raw) - len(unique_idx)
    _random.shuffle(unique_idx)
    candidates = unique_idx[: int(n_pairs * 1.1) + 16]
    subset = raw.select(candidates)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pad_id = tokenizer.pad_token_id

    chosen_ids: list[list[int]] = []
    rejected_ids: list[list[int]] = []
    n_identical_dropped = 0
    for ex in subset:
        c_enc, r_enc = encode_pair_consistent(
            tokenizer, ex["prompt"], ex["chosen"], ex["rejected"], max_len
        )
        if c_enc == r_enc:
            n_identical_dropped += 1  # unlearnable and undetectable — exclude
            continue
        chosen_ids.append(c_enc)
        rejected_ids.append(r_enc)
        if len(chosen_ids) == n_pairs:
            break
    n_pairs = len(chosen_ids)
    print(f"[seed={seed}] {n_pairs} pairs kept "
          f"(dropped {n_identical_dropped} identical-encoding, "
          f"{n_dup_dropped} duplicate prompts in corpus)")

    hf_cache.commit()  # persist dataset/tokenizer downloads for future runs

    # -----------------------------------------------------------------------
    # Inject noise — flip chosen/rejected for noise_rate fraction
    # -----------------------------------------------------------------------
    n_noisy = int(n_pairs * noise_rate)
    noisy_indices = set(_random.sample(range(n_pairs), n_noisy))
    for i in noisy_indices:
        chosen_ids[i], rejected_ids[i] = rejected_ids[i], chosen_ids[i]
    is_flipped = [i in noisy_indices for i in range(n_pairs)]
    print(f"[seed={seed}] Injected {n_noisy} flips ({noise_rate*100:.0f}%)")

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1
    ).to(device)
    hf_cache.commit()  # persist model weights
    optimizer = AdamW(model.parameters(), lr=lr)

    steps_per_epoch = math.ceil(n_pairs / batch_size)
    total_steps = steps_per_epoch * n_epochs
    eval_every = max(1, steps_per_epoch // evals_per_epoch)

    train_logger = TrajectoryLogger(n_examples=n_pairs)   # old method (baseline)
    eval_logger = TrajectoryLogger(n_examples=n_pairs)    # eval-pass losses
    margins: list[list[float]] = [[] for _ in range(n_pairs)]
    eval_steps: list[int] = []

    def _score_batch(idx_list):
        c_t, c_m = pad_batch([chosen_ids[i] for i in idx_list], pad_id)
        r_t, r_m = pad_batch([rejected_ids[i] for i in idx_list], pad_id)
        r_c = model(input_ids=c_t.to(device), attention_mask=c_m.to(device)).logits.squeeze(-1)
        r_r = model(input_ids=r_t.to(device), attention_mask=r_m.to(device)).logits.squeeze(-1)
        return r_c, r_r

    def eval_pass(step: int) -> None:
        """Score every pair at the SAME model state, dropout off."""
        model.eval()
        with torch.no_grad():
            for start in range(0, n_pairs, 64):
                idx_list = list(range(start, min(start + 64, n_pairs)))
                r_c, r_r = _score_batch(idx_list)
                margin = r_c - r_r
                loss = -F.logsigmoid(margin)
                for local, i in enumerate(idx_list):
                    eval_logger.log(i, step, float(loss[local]))
                    margins[i].append(float(margin[local]))
        eval_steps.append(step)
        model.train()

    # -----------------------------------------------------------------------
    # Training loop with periodic eval passes
    # -----------------------------------------------------------------------
    print(f"[seed={seed}] Training: {n_epochs} epochs x {steps_per_epoch} steps "
          f"= {total_steps}; eval pass every {eval_every} steps")
    eval_pass(0)

    global_step = 0
    for epoch in range(n_epochs):
        order = list(range(n_pairs))
        _random.shuffle(order)
        for batch_start in range(0, n_pairs, batch_size):
            batch_ids = order[batch_start:batch_start + batch_size]
            r_c, r_r = _score_batch(batch_ids)
            per_example_loss = -F.logsigmoid(r_c - r_r)
            loss = per_example_loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            for local, i in enumerate(batch_ids):
                train_logger.log(i, global_step, float(per_example_loss[local].detach()))

            global_step += 1
            if global_step % eval_every == 0 and global_step < total_steps:
                eval_pass(global_step)
            if global_step % 100 == 0:
                print(f"[seed={seed}] epoch={epoch+1} step={global_step}/{total_steps} "
                      f"loss={loss.item():.4f}")
    eval_pass(total_steps)

    # -----------------------------------------------------------------------
    # Features + signal check
    # -----------------------------------------------------------------------
    train_logger.finalize(total_steps=total_steps)
    eval_logger.finalize(total_steps=total_steps)
    eval_feats = eval_logger.get_all_features()
    train_feats = train_logger.get_all_features()

    margin_mean = [float(np.mean(m)) for m in margins]     # AUM-style
    margin_final = [m[-1] for m in margins]

    feature_sources = {
        "eval_mean_loss_last_k": lambda i: eval_feats[i].mean_loss_last_k,
        "eval_variance": lambda i: eval_feats[i].variance,
        "eval_residual_variance": lambda i: eval_feats[i].residual_variance,
        "eval_slope": lambda i: eval_feats[i].slope,
        "eval_final_loss": lambda i: eval_feats[i].loss_history[-1],
        "neg_aum_margin": lambda i: -margin_mean[i],        # higher = more suspicious
        "neg_final_margin": lambda i: -margin_final[i],
        "train_mean_loss_last_k": lambda i: train_feats[i].mean_loss_last_k,  # old method
        "train_variance": lambda i: train_feats[i].variance,                  # old method
    }
    signal_check = {}
    print(f"\n[seed={seed}] {'='*66}")
    print(f"[seed={seed}] SIGNAL CHECK: true-FLIPPED vs true-CLEAN (AUROC, higher=better)")
    for name, fn in feature_sources.items():
        pos = [fn(i) for i in range(n_pairs) if is_flipped[i]]
        neg = [fn(i) for i in range(n_pairs) if not is_flipped[i]]
        stats = separation_stats(pos, neg)
        signal_check[name] = stats
        print(f"[seed={seed}]   {name:<26} AUROC={stats['auroc']:.3f}")
    print(f"[seed={seed}] {'='*66}\n")

    # -----------------------------------------------------------------------
    # Report (rate-calibrated buckets) + ranked evaluation
    # -----------------------------------------------------------------------
    report = PreferenceQualityReport(eval_feats, expected_noise_rate=noise_rate)

    true_flipped = {i for i in range(n_pairs) if is_flipped[i]}
    pred_flipped = {r.example_idx for r in report.flipped}
    pred_clean = {r.example_idx for r in report.clean}

    def prf(predicted: set, truth: set) -> tuple[float, float, float]:
        tp = len(predicted & truth)
        fp = len(predicted - truth)
        fn = len(truth - predicted)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return p, r, f

    fp_p, fp_r, fp_f1 = prf(pred_flipped, true_flipped)
    cl_p, cl_r, cl_f1 = prf(pred_clean, set(range(n_pairs)) - true_flipped)

    ranked = report.ranked()
    top_k = {r.example_idx for r in ranked[:n_noisy]}
    precision_at_k = len(top_k & true_flipped) / n_noisy if n_noisy else 0.0
    score_auroc = separation_stats(
        [r.score for r in ranked if r.example_idx in true_flipped],
        [r.score for r in ranked if r.example_idx not in true_flipped],
    )["auroc"]

    summary = report.summary()
    per_example = [
        {
            "idx": i,
            "flipped": is_flipped[i],
            "eval_losses": [round(x, 5) for x in eval_feats[i].loss_history],
            "margins": [round(x, 5) for x in margins[i]],
            "train_losses": [round(x, 5) for x in train_feats[i].loss_history],
        }
        for i in range(n_pairs)
    ]

    results = {
        "n_pairs": n_pairs,
        "n_noisy": n_noisy,
        "noise_rate": noise_rate,
        "n_epochs": n_epochs,
        "seed": seed,
        "model": model_name,
        "dataset": "Dahoas/rm-static",
        "evals_per_epoch": evals_per_epoch,
        "eval_steps": eval_steps,
        "n_identical_dropped": n_identical_dropped,
        "n_duplicate_prompts_in_corpus": n_dup_dropped,
        "flipped": {"precision": fp_p, "recall": fp_r, "f1": fp_f1},
        "clean": {"precision": cl_p, "recall": cl_r, "f1": cl_f1},
        "precision_at_k": precision_at_k,
        "score_auroc": score_auroc,
        "summary": summary,
        "signal_check": signal_check,
        "caveat": "rm-static labels are naturally noisy; FLIPPED precision "
                  "vs injected flips is a lower bound.",
        "per_example": per_example,
    }

    print(f"[seed={seed}] FLIPPED P={fp_p:.3f} R={fp_r:.3f} F1={fp_f1:.3f} | "
          f"precision@k={precision_at_k:.3f} score_AUROC={score_auroc:.3f}")
    return results


# ---------------------------------------------------------------------------
# Local entrypoint: modal run benchmarks/modal/prefcheck_real.py
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    n_pairs: int = 5000,
    n_epochs: int = 3,
    noise_rate: float = 0.20,
    seeds: str = "42,43,44",
    model_name: str = "distilbert-base-uncased",
    evals_per_epoch: int = 2,
):
    import statistics

    seed_list = [int(s) for s in seeds.split(",")]
    arg_tuples = [
        (n_pairs, noise_rate, n_epochs, 16, s, model_name, evals_per_epoch)
        for s in seed_list
    ]
    all_results = list(run_prefcheck.starmap(arg_tuples))

    out_dir = Path("benchmarks/modal/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    for res in all_results:
        model_tag = res["model"].split("/")[-1]
        out = out_dir / (
            f"prefcheck_{model_tag}_{res['n_pairs']}pairs_{res['n_epochs']}epochs_"
            f"evalpass_seed{res['seed']}.json"
        )
        with open(out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Saved {out}")

    def agg(path_fn):
        vals = [path_fn(r) for r in all_results]
        mean = statistics.mean(vals)
        std = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return f"{mean:.3f} ± {std:.3f}"

    print(f"\n{'='*70}")
    print(f"AGGREGATE over seeds {seed_list} "
          f"({n_pairs} pairs, {n_epochs} epochs, {model_name})")
    print(f"{'='*70}")
    print(f"FLIPPED precision : {agg(lambda r: r['flipped']['precision'])}")
    print(f"FLIPPED recall    : {agg(lambda r: r['flipped']['recall'])}")
    print(f"FLIPPED F1        : {agg(lambda r: r['flipped']['f1'])}")
    print(f"precision@k       : {agg(lambda r: r['precision_at_k'])}")
    print(f"score AUROC       : {agg(lambda r: r['score_auroc'])}")
    for feat in ("eval_mean_loss_last_k", "eval_residual_variance",
                 "neg_aum_margin", "neg_final_margin",
                 "train_mean_loss_last_k", "train_variance"):
        print(f"AUROC {feat:<26}: "
              f"{agg(lambda r, f=feat: r['signal_check'][f]['auroc'])}")
    print(f"{'='*70}")
