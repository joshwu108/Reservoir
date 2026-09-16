"""benchmarks/anchor/forgetting_benchmark.py

Validates that reservoir-anchor detects catastrophic forgetting with
lead time before eval accuracy visibly drops.

Experiment
----------
1. Task A: binary classification on features [0:10] (synthetic)
2. Task B: binary classification on features [10:20] (different features)
3. MLP trains on Task A until convergence
4. Sequential fine-tune on Task B — model forgets Task A
5. AnchorSet monitors Task A examples every EVAL_EVERY steps
6. Key result: anchor forgetting score rises BEFORE Task A eval accuracy drops

No GPU, no transformers, no HuggingFace needed.
Runs in ~3-5 minutes on CPU.

Usage
-----
    uv run python -m benchmarks.anchor.forgetting_benchmark

Output
------
    Console: per-step forgetting scores and lead-time result
    benchmarks/results/forgetting_benchmark.json
    benchmarks/results/forgetting_benchmark.png  (if matplotlib available)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from reservoir.anchor_set import AnchorSet, AnchorExample

# ── Config ────────────────────────────────────────────────────────────
N_FEATURES    = 20
N_TASK_A      = 2000       # training examples for Task A
N_TASK_B      = 2000       # training examples for Task B
N_EVAL        = 500        # eval examples per task
N_ANCHORS     = 200        # anchor examples from Task A
PRETRAIN_EPOCHS = 10       # epochs on Task A before fine-tuning
FINETUNE_STEPS  = 300      # gradient steps on Task B
EVAL_EVERY      = 20       # evaluate anchors every N fine-tune steps
ALERT_THRESHOLD = 0.5      # forgetting score that triggers an alert
BATCH_SIZE      = 64
LR              = 3e-3
SEED            = 42

RESULTS = Path(__file__).parent.parent / "results"
RESULTS.mkdir(exist_ok=True)


# ── Synthetic data ────────────────────────────────────────────────────

def make_task(n: int, signal_features: slice, noise: float, rng: np.random.Generator):
    """Binary classification where only signal_features determine the label."""
    X = rng.standard_normal((n, N_FEATURES)).astype(np.float32)
    w = rng.standard_normal(N_FEATURES // 2).astype(np.float32)
    logits = X[:, signal_features] @ w + noise * rng.standard_normal(n).astype(np.float32)
    y = (logits > 0).astype(np.int64)
    return X, y


# ── Model ─────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, in_features: int = N_FEATURES, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),      nn.ReLU(),
            nn.Linear(hidden, 2),
        )
    def forward(self, x): return self.net(x)


# ── Helpers ───────────────────────────────────────────────────────────

def compute_losses(model: MLP, X: np.ndarray, y: np.ndarray,
                   device: torch.device) -> np.ndarray:
    """Per-example cross-entropy loss, returned as numpy array."""
    model.eval()
    with torch.no_grad():
        xT = torch.tensor(X, device=device)
        yT = torch.tensor(y, device=device)
        logits = model(xT)
        losses = F.cross_entropy(logits, yT, reduction="none").cpu().numpy()
    model.train()
    return losses


def eval_accuracy(model: MLP, X: np.ndarray, y: np.ndarray,
                  device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X, device=device))
        preds  = logits.argmax(dim=1).cpu().numpy()
    model.train()
    return float((preds == y).mean())


# ── Main ──────────────────────────────────────────────────────────────

def run() -> dict:
    rng    = np.random.default_rng(SEED)
    device = torch.device("cpu")   # CPU is fine for this benchmark

    # ── Generate tasks
    task_a_slice = slice(0, N_FEATURES // 2)   # features 0-9 → Task A
    task_b_slice = slice(N_FEATURES // 2, N_FEATURES)  # features 10-19 → Task B

    X_a_train, y_a_train = make_task(N_TASK_A, task_a_slice, noise=0.3, rng=rng)
    X_a_eval,  y_a_eval  = make_task(N_EVAL,   task_a_slice, noise=0.3, rng=rng)
    X_b_train, y_b_train = make_task(N_TASK_B, task_b_slice, noise=0.3, rng=rng)

    # ── Pre-train on Task A
    print("Phase 1: Pre-training on Task A...")
    model     = MLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    ds_a      = TensorDataset(torch.tensor(X_a_train), torch.tensor(y_a_train))
    loader_a  = DataLoader(ds_a, batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(PRETRAIN_EPOCHS):
        for xb, yb in loader_a:
            loss = F.cross_entropy(model(xb.to(device)), yb.to(device))
            optimizer.zero_grad(); loss.backward(); optimizer.step()

    pretrain_acc = eval_accuracy(model, X_a_eval, y_a_eval, device)
    print(f"  Task A accuracy after pre-training: {pretrain_acc:.3f}")

    # ── Set up AnchorSet from Task A examples
    print("\nPhase 2: Setting up AnchorSet from Task A...")
    anchor_indices  = rng.choice(N_TASK_A, N_ANCHORS, replace=False)
    X_anchors = X_a_train[anchor_indices]
    y_anchors = y_a_train[anchor_indices]

    anchor_examples = [
        AnchorExample(idx=i, data={"x": X_anchors[i], "y": y_anchors[i]},
                      tag="task-a", baseline_loss=0.0, current_loss=0.0, priority=0.0)
        for i in range(N_ANCHORS)
    ]
    anchors = AnchorSet(examples=[{"x": X_anchors[i], "y": y_anchors[i]}
                                   for i in range(N_ANCHORS)],
                        tags="task-a", n=None)

    # Snapshot baseline losses on anchors BEFORE fine-tuning
    baseline_losses = compute_losses(model, X_anchors, y_anchors, device)
    anchors.snapshot_baseline({i: float(baseline_losses[i]) for i in range(N_ANCHORS)})
    print(f"  Anchor baseline loss: {baseline_losses.mean():.4f}")

    # ── Fine-tune on Task B, monitor anchors
    print(f"\nPhase 3: Fine-tuning on Task B ({FINETUNE_STEPS} steps)...")
    print(f"  Evaluating anchors every {EVAL_EVERY} steps, alert threshold = {ALERT_THRESHOLD}")
    print(f"  {'Step':>6}  {'Forgetting':>12}  {'TaskA Acc':>10}  {'Alert'}")
    print("  " + "-" * 45)

    ds_b     = TensorDataset(torch.tensor(X_b_train), torch.tensor(y_b_train))
    loader_b = DataLoader(ds_b, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    b_iter   = iter(loader_b)

    history: list[dict] = []
    first_alert_step: int | None = None
    first_acc_drop_step: int | None = None
    acc_drop_threshold = pretrain_acc - 0.10  # 10 percentage points drop

    for step in range(1, FINETUNE_STEPS + 1):
        try:
            xb, yb = next(b_iter)
        except StopIteration:
            b_iter = iter(loader_b)
            xb, yb = next(b_iter)

        loss = F.cross_entropy(model(xb.to(device)), yb.to(device))
        optimizer.zero_grad(); loss.backward(); optimizer.step()

        if step % EVAL_EVERY == 0:
            # Update anchor losses
            current_losses = compute_losses(model, X_anchors, y_anchors, device)
            anchors.update_current_losses({i: float(current_losses[i])
                                           for i in range(N_ANCHORS)})
            forgetting = anchors.forgetting_scores().get("task-a", 0.0)

            # Eval Task A accuracy
            task_a_acc = eval_accuracy(model, X_a_eval, y_a_eval, device)

            alert = forgetting > ALERT_THRESHOLD
            if alert and first_alert_step is None:
                first_alert_step = step

            if task_a_acc < acc_drop_threshold and first_acc_drop_step is None:
                first_acc_drop_step = step

            marker = " ← ALERT" if alert else ""
            if task_a_acc < acc_drop_threshold and first_acc_drop_step == step:
                marker += " ← ACC DROP"

            print(f"  {step:>6}  {forgetting:>12.4f}  {task_a_acc:>10.3f}{marker}")

            history.append({
                "step": step,
                "forgetting_score": round(forgetting, 6),
                "task_a_accuracy": round(task_a_acc, 4),
                "alert_fired": alert,
            })

    # ── Results
    final_acc = eval_accuracy(model, X_a_eval, y_a_eval, device)
    print(f"\n{'='*55}")
    print(f"  reservoir-anchor: Forgetting Monitor Results")
    print(f"{'='*55}")
    print(f"  Task A accuracy before fine-tuning : {pretrain_acc:.3f}")
    print(f"  Task A accuracy after fine-tuning  : {final_acc:.3f}")
    print(f"  Accuracy drop                      : {pretrain_acc - final_acc:.3f}")

    if first_alert_step and first_acc_drop_step:
        lead_time = first_acc_drop_step - first_alert_step
        print(f"\n  Forgetting alert fired  : step {first_alert_step}")
        print(f"  Accuracy drop detected  : step {first_acc_drop_step}")
        print(f"  Lead time               : {lead_time} steps ({lead_time/FINETUNE_STEPS*100:.1f}% of fine-tune)")
        if lead_time > 0:
            print(f"\n  ✓ Lead time confirmed — alert fired {lead_time} steps before accuracy collapsed")
        else:
            print(f"\n  ✗ No lead time — alert and accuracy drop were simultaneous")
    elif first_alert_step and not first_acc_drop_step:
        print(f"\n  Alert fired at step {first_alert_step} but accuracy never dropped >10%")
    elif not first_alert_step:
        print(f"\n  No alert fired (forgetting score never exceeded {ALERT_THRESHOLD})")
    print(f"{'='*55}")

    results = {
        "pretrain_accuracy": round(pretrain_acc, 4),
        "final_accuracy": round(final_acc, 4),
        "accuracy_drop": round(pretrain_acc - final_acc, 4),
        "first_alert_step": first_alert_step,
        "first_acc_drop_step": first_acc_drop_step,
        "lead_time_steps": (first_acc_drop_step - first_alert_step)
            if (first_alert_step and first_acc_drop_step) else None,
        "alert_threshold": ALERT_THRESHOLD,
        "n_anchors": N_ANCHORS,
        "finetune_steps": FINETUNE_STEPS,
        "history": history,
    }

    # Save JSON
    out_json = RESULTS / "forgetting_benchmark.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_json}")

    # Save plot
    _try_plot(history, results, RESULTS / "forgetting_benchmark.png",
              pretrain_acc, acc_drop_threshold)

    return results


def _try_plot(history: list[dict], results: dict, path: Path,
              pretrain_acc: float, acc_drop_threshold: float) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot")
        return

    steps    = [h["step"] for h in history]
    forget   = [h["forgetting_score"] for h in history]
    acc      = [h["task_a_accuracy"] for h in history]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.plot(steps, forget, "r-o", markersize=4, label="Forgetting score (anchors)")
    ax1.axhline(results["alert_threshold"], color="r", linestyle="--", alpha=0.5,
                label=f"Alert threshold ({results['alert_threshold']})")
    ax2.plot(steps, acc, "b-s", markersize=4, label="Task A accuracy")
    ax2.axhline(pretrain_acc, color="b", linestyle=":", alpha=0.5,
                label=f"Pre-finetune accuracy ({pretrain_acc:.3f})")
    ax2.axhline(acc_drop_threshold, color="b", linestyle="--", alpha=0.5,
                label=f"Acc drop threshold ({acc_drop_threshold:.3f})")

    if results["first_alert_step"]:
        ax1.axvline(results["first_alert_step"], color="r", alpha=0.3,
                    label=f"Alert step {results['first_alert_step']}")
    if results["first_acc_drop_step"]:
        ax2.axvline(results["first_acc_drop_step"], color="b", alpha=0.3,
                    label=f"Acc drop step {results['first_acc_drop_step']}")

    lead = results.get("lead_time_steps")
    title = (f"reservoir-anchor: Forgetting Monitor\n"
             f"Lead time: {lead} steps" if lead else
             "reservoir-anchor: Forgetting Monitor")
    ax1.set_title(title)
    ax1.set_xlabel("Fine-tuning step")
    ax1.set_ylabel("Forgetting score", color="r")
    ax2.set_ylabel("Task A accuracy", color="b")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center left", fontsize=8)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"Plot saved to {path}")


if __name__ == "__main__":
    run()
