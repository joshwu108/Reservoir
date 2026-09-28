"""benchmarks/modal/prefcheck_real.py

benchmark for reservoir-prefcheck.

Loads 1 000 pairs from Dahoas/rm-static, injects 20 % label noise,
train DistilBERT reward model for 3 epochs, then runs
PreferenceQualityReport over the per-example loss trajectories and
reports Precision / Recall / F1 vs. the known ground-truth noise labels.

Usage
-----
    # Install Modal once:
    pip install modal
    modal setup

    # Run (spins up a T4 GPU, ~25-40 min, ~$0.25-0.40):
    modal run benchmarks/modal/prefcheck_real.py

    # Download results JSON after the run:
    # (printed to stdout; redirect as needed)
"""

from __future__ import annotations

import json
import random
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
    "scikit-learn",
]

#docker image on modal servers   
try:
    _repo_src = str(Path(__file__).parents[2] / "src")
except IndexError:
    _repo_src = "."  # running inside container; image is already built

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*_DEPS)
    .add_local_dir(_repo_src, remote_path="/reservoir_src")
)

app = modal.App("reservoir-prefcheck-real")

# ---------------------------------------------------------------------------
# Signal-separation diagnostics (pure numpy; runs inside the container too)
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
    0.5 means no signal; ~0.0 means the ranking is inverted (memorization).
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


# ---------------------------------------------------------------------------
# GPU function
# ---------------------------------------------------------------------------

@app.function(
    gpu="T4",
    image=image,
    timeout=7200,
)
def run_prefcheck(
    n_pairs: int = 1000,
    noise_rate: float = 0.20,
    n_epochs: int = 7,
    batch_size: int = 16,
    seed: int = 42,
) -> dict:
    """Train a DistilBERT reward model on noisy preference data and detect noise."""
    import random as _random
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from torch.optim import AdamW

    sys.path.insert(0, "/reservoir_src")
    from reservoir.trajectory import TrajectoryLogger
    from reservoir.report import PreferenceQualityReport

    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -----------------------------------------------------------------------
    # Load dataset
    # -----------------------------------------------------------------------
    print(f"\nLoading Dahoas/rm-static ({n_pairs} pairs)...")
    raw = load_dataset("Dahoas/rm-static", split="train")
    indices = list(range(len(raw)))
    _random.shuffle(indices)
    indices = indices[:n_pairs]
    dataset = raw.select(indices)
    print(f"  Loaded {len(dataset)} pairs")

    # -----------------------------------------------------------------------
    # Inject noise — flip chosen/rejected for noise_rate fraction
    # -----------------------------------------------------------------------
    n_noisy = int(n_pairs * noise_rate)
    noisy_indices = set(_random.sample(range(n_pairs), n_noisy))
    print(f"  Injecting {n_noisy} flipped labels ({noise_rate*100:.0f}% noise)")

    # Build final dataset as lists
    prompts, chosen_texts, rejected_texts, is_flipped = [], [], [], []
    for i, example in enumerate(dataset):
        c, r = example["chosen"], example["rejected"]
        if i in noisy_indices:
            c, r = r, c   # flip
            is_flipped.append(True)
        else:
            is_flipped.append(False)
        prompts.append(example["prompt"])
        chosen_texts.append(c)
        rejected_texts.append(r)

    # -----------------------------------------------------------------------
    # Tokenize
    # -----------------------------------------------------------------------
    print("\nTokenizing...")
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Keep the END of prompt+response when truncating. rm-static prompts are
    # long multi-turn dialogues; with the default right-truncation ~18% of
    # pairs lose the response entirely, making chosen/rejected encodings
    # identical (loss pinned at ln 2, undetectable in principle).
    tokenizer.truncation_side = "left"

    MAX_LEN = 256

    def encode_pair(prompt: str, text: str) -> dict:
        combined = prompt + " " + text
        return tokenizer(
            combined,
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

    chosen_enc  = [encode_pair(p, c) for p, c in zip(prompts, chosen_texts)]
    rejected_enc = [encode_pair(p, r) for p, r in zip(prompts, rejected_texts)]

    # Guardrail: pairs whose encodings are identical are unlearnable and
    # undetectable — report how many remain after left-truncation.
    n_identical = sum(
        int(torch.equal(c["input_ids"], r["input_ids"]))
        for c, r in zip(chosen_enc, rejected_enc)
    )
    print(f"  Identical chosen/rejected encodings: {n_identical}/{n_pairs} "
          f"({100.0*n_identical/n_pairs:.1f}%)")

    # -----------------------------------------------------------------------
    # Model — binary reward classifier (chosen > rejected)
    # -----------------------------------------------------------------------
    print(f"\nLoading {model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=1
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)

    # -----------------------------------------------------------------------
    # Training loop — log per-example loss trajectories
    # -----------------------------------------------------------------------
    logger = TrajectoryLogger(n_examples=n_pairs)

    steps_per_epoch = n_pairs // batch_size
    total_steps = steps_per_epoch * n_epochs
    print(f"\nTraining: {n_epochs} epochs × {steps_per_epoch} steps = {total_steps} total")

    global_step = 0
    for epoch in range(n_epochs):
        # Shuffle order each epoch
        order = list(range(n_pairs))
        _random.shuffle(order)

        for batch_start in range(0, n_pairs - batch_size + 1, batch_size):
            batch_ids = order[batch_start:batch_start + batch_size]

            # Stack chosen and rejected tensors
            c_ids  = torch.cat([chosen_enc[i]["input_ids"]      for i in batch_ids]).to(device)
            c_mask = torch.cat([chosen_enc[i]["attention_mask"]  for i in batch_ids]).to(device)
            r_ids  = torch.cat([rejected_enc[i]["input_ids"]     for i in batch_ids]).to(device)
            r_mask = torch.cat([rejected_enc[i]["attention_mask"] for i in batch_ids]).to(device)

            # Forward pass
            r_chosen   = model(input_ids=c_ids, attention_mask=c_mask).logits.squeeze(-1)
            r_rejected = model(input_ids=r_ids, attention_mask=r_mask).logits.squeeze(-1)

            # Reward model loss
            per_example_loss = -F.logsigmoid(r_chosen - r_rejected)

            loss = per_example_loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log per-example losses to trajectory logger
            for local_i, idx in enumerate(batch_ids):
                logger.log(idx, global_step, float(per_example_loss[local_i].detach()))

            global_step += 1
            if global_step % 20 == 0:
                print(f"  epoch={epoch+1} step={global_step}/{total_steps} loss={loss.item():.4f}")

    # -----------------------------------------------------------------------
    # Build noise report
    # -----------------------------------------------------------------------
    logger.finalize(total_steps=total_steps)
    features = logger.get_all_features()

    # -----------------------------------------------------------------------
    # Debug dump: does the signal exist at all?
    # Compare feature distributions for true-FLIPPED vs true-CLEAN using the
    # ground truth (available here because we injected the noise ourselves).
    # -----------------------------------------------------------------------
    signal_check: dict[str, dict] = {}
    print(f"\n{'='*60}")
    print("SIGNAL CHECK: true-FLIPPED vs true-CLEAN feature separation")
    print(f"{'='*60}")
    for feat_name in ("mean_loss_last_k", "variance", "slope"):
        pos_vals = [getattr(features[i], feat_name) for i in features if is_flipped[i]]
        neg_vals = [getattr(features[i], feat_name) for i in features if not is_flipped[i]]
        stats = separation_stats(pos_vals, neg_vals)
        signal_check[feat_name] = stats
        p, n = stats["pos"], stats["neg"]
        print(f"\n{feat_name}:  AUROC={stats['auroc']:.3f}"
              f"  (1.0=perfect, 0.5=no signal, <0.5=inverted/memorized)")
        print(f"  FLIPPED (n={p['n']}):  mean={p['mean']:.6f}  median={p['median']:.6f}"
              f"  p25={p['p25']:.6f}  p75={p['p75']:.6f}")
        print(f"  CLEAN   (n={n['n']}):  mean={n['mean']:.6f}  median={n['median']:.6f}"
              f"  p25={n['p25']:.6f}  p75={n['p75']:.6f}")
    print(f"{'='*60}\n")

    report = PreferenceQualityReport(features)

    true_flipped = set(i for i, f in enumerate(is_flipped) if f)
    pred_flipped = {r.example_idx for r in report.flipped}
    pred_ambiguous = {r.example_idx for r in report.ambiguous}
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
    true_clean = set(range(n_pairs)) - true_flipped
    cl_p, cl_r, cl_f1 = prf(pred_clean, true_clean)

    summary = report.summary()
    results = {
        "n_pairs": n_pairs,
        "n_noisy": n_noisy,
        "noise_rate": noise_rate,
        "n_epochs": n_epochs,
        "model": model_name,
        "dataset": "Dahoas/rm-static",
        "flipped": {"precision": fp_p, "recall": fp_r, "f1": fp_f1},
        "clean":   {"precision": cl_p, "recall": cl_r, "f1": cl_f1},
        "summary": summary,
        "signal_check": signal_check,
        "n_identical_encodings": n_identical,
    }

    # Print table
    print(f"\n{'='*60}")
    print("reservoir-prefcheck: Real-Data Benchmark (Dahoas/rm-static)")
    print(f"{'='*60}")
    print(f"Dataset   : {n_pairs} pairs with {n_noisy} injected flips ({noise_rate*100:.0f}%)")
    print(f"Model     : {model_name}")
    print(f"Training  : {n_epochs} epochs")
    print()
    print(f"{'Label':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 45)
    print(f"{'FLIPPED':<12} {fp_p:>10.3f} {fp_r:>10.3f} {fp_f1:>10.3f}")
    print(f"{'CLEAN':<12} {cl_p:>10.3f} {cl_r:>10.3f} {cl_f1:>10.3f}")
    print()
    print(f"Report: FLIPPED={summary['n_flipped']} AMBIGUOUS={summary['n_ambiguous']} CLEAN={summary['n_clean']}")
    print(f"{'='*60}\n")

    return results


# ---------------------------------------------------------------------------
# Local entrypoint run: modal run benchmarks/modal/prefcheck_real.py
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(n_pairs: int = 5000, n_epochs: int = 3, noise_rate: float = 0.20):
    results = run_prefcheck.remote(
        n_pairs=n_pairs, n_epochs=n_epochs, noise_rate=noise_rate
    )
    print("\nFull results JSON:")
    print(json.dumps(results, indent=2))
