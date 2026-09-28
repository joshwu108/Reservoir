"""benchmarks/modal/forgetting_real.py

Real-data benchmark for reservoir-anchor (ForgettingMonitor).

Fine-tunes DistilGPT-2 on a slice of wikitext-103 (Task A), snapshots
anchor examples, then fine-tunes on Python code from code_search_net
(Task B) and monitors per-anchor forgetting scores with ForgettingMonitor.
Reports lead time: how many steps before Task A perplexity visibly rises.

Usage
-----
    modal run benchmarks/modal/forgetting_real.py

Cost estimate:  T4 @ $0.59/hr × ~0.5 hr  ≈  $0.30
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

# ---------------------------------------------------------------------------
# Modal image
# ---------------------------------------------------------------------------

_DEPS = [
    "torch==2.2.2",
    "transformers==4.40.2",
    "datasets==2.19.1",
    "numpy==1.26.4",
    "accelerate==0.30.1",
]

try:
    _repo_src = str(Path(__file__).parents[2] / "src")
except IndexError:
    _repo_src = "."  # running inside container; image is already built

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(*_DEPS)
    .add_local_dir(_repo_src, remote_path="/reservoir_src")
)

app = modal.App("reservoir-forgetting-real")

# ---------------------------------------------------------------------------
# GPU function
# ---------------------------------------------------------------------------

@app.function(
    gpu="T4",
    image=image,
    timeout=7200,
)
def run_forgetting(
    n_task_a: int = 800,
    n_task_b: int = 800,
    n_anchors: int = 200,
    pretrain_steps: int = 200,
    finetune_steps: int = 400,
    eval_every: int = 25,
    alert_threshold: float = 0.5,
    batch_size: int = 8,
    seed: int = 42,
) -> dict:
    """Fine-tune DistilGPT-2 on two tasks and monitor forgetting."""
    import random as _random
    import numpy as np
    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from torch.optim import AdamW

    sys.path.insert(0, "/reservoir_src")
    from reservoir.anchor_set import AnchorSet, AnchorExample

    _random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_name = "distilgpt2"
    MAX_LEN = 128
    LR = 3e-5

    # -----------------------------------------------------------------------
    # Load tokenizer + model
    # -----------------------------------------------------------------------
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=LR)

    def tokenize(texts: list[str]) -> list[dict]:
        enc = tokenizer(
            texts,
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return [
            {"input_ids": enc["input_ids"][i], "attention_mask": enc["attention_mask"][i]}
            for i in range(len(texts))
        ]

    def causal_loss(model, input_ids, attention_mask) -> torch.Tensor:
        """Per-example causal LM loss."""
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        return out.loss  # scalar mean over non-padding tokens

    def per_example_loss(model, batch: list[dict]) -> list[float]:
        model.eval()
        losses = []
        with torch.no_grad():
            for ex in batch:
                ids  = ex["input_ids"].unsqueeze(0).to(device)
                mask = ex["attention_mask"].unsqueeze(0).to(device)
                losses.append(float(causal_loss(model, ids, mask).item()))
        model.train()
        return losses

    # -----------------------------------------------------------------------
    # Load Task A: wikitext-103 (general prose)
    # -----------------------------------------------------------------------
    print("\nLoading Task A: wikitext-103-raw-v1...")
    wt = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    wt_texts = [r["text"] for r in wt if len(r["text"].strip()) > 60]
    _random.shuffle(wt_texts)
    wt_texts = wt_texts[:n_task_a + n_anchors]
    task_a_train = tokenize(wt_texts[:n_task_a])
    task_a_anchor_raw = wt_texts[n_task_a: n_task_a + n_anchors]
    task_a_anchors = tokenize(task_a_anchor_raw)
    print(f"  Task A: {len(task_a_train)} train, {len(task_a_anchors)} anchors")

    # -----------------------------------------------------------------------
    # Load Task B: Python code (code_search_net)
    # -----------------------------------------------------------------------
    print("Loading Task B: code_search_net (python)...")
    code = load_dataset("code_search_net", "python", split="train", trust_remote_code=True)
    code_texts = [r["func_code_string"] for r in code if len(r["func_code_string"]) > 60]
    _random.shuffle(code_texts)
    code_texts = code_texts[:n_task_b]
    task_b_train = tokenize(code_texts)
    print(f"  Task B: {len(task_b_train)} train examples")

    # -----------------------------------------------------------------------
    # Phase 1: pre-train on Task A
    # -----------------------------------------------------------------------
    print(f"\nPhase 1: Pre-training on Task A ({pretrain_steps} steps)...")
    _random.shuffle(task_a_train)
    step = 0
    idx = 0
    while step < pretrain_steps:
        batch_exs = task_a_train[idx: idx + batch_size]
        if not batch_exs:
            _random.shuffle(task_a_train)
            idx = 0
            continue
        ids  = torch.stack([e["input_ids"]      for e in batch_exs]).to(device)
        mask = torch.stack([e["attention_mask"]  for e in batch_exs]).to(device)
        loss = causal_loss(model, ids, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        idx += batch_size
        step += 1
        if step % 50 == 0:
            print(f"  step={step}/{pretrain_steps} loss={loss.item():.4f}")

    # Baseline Task A perplexity
    anchor_losses_baseline = per_example_loss(model, task_a_anchors)
    baseline_ppl = float(np.exp(np.mean(anchor_losses_baseline)))
    print(f"  Task A perplexity after pre-training: {baseline_ppl:.2f}")

    # -----------------------------------------------------------------------
    # Phase 2: set up AnchorSet + snapshot baseline
    # -----------------------------------------------------------------------
    print(f"\nPhase 2: Snapshotting {n_anchors} Task A anchors...")
    anchor_examples = [
        {"input_ids": task_a_anchors[i]["input_ids"],
         "attention_mask": task_a_anchors[i]["attention_mask"]}
        for i in range(n_anchors)
    ]
    anchor_set = AnchorSet(examples=anchor_examples, tags="wikitext")
    anchor_set.snapshot_baseline({i: anchor_losses_baseline[i] for i in range(n_anchors)})
    print(f"  Baseline mean anchor loss: {np.mean(anchor_losses_baseline):.4f}")

    # -----------------------------------------------------------------------
    # Phase 3: fine-tune on Task B, monitor anchor forgetting
    # -----------------------------------------------------------------------
    print(f"\nPhase 3: Fine-tuning on Task B ({finetune_steps} steps)...")
    print(f"  Eval anchors every {eval_every} steps, alert threshold={alert_threshold}")
    print(f"\n  {'Step':>6}  {'Forgetting':>12}  {'Anchor PPL':>12}  {'Alert'}")
    print("  " + "-" * 48)

    history: list[dict] = []
    first_alert_step: int | None = None
    first_ppl_rise_step: int | None = None
    ppl_rise_threshold = baseline_ppl * 1.25  # 25% perplexity increase

    _random.shuffle(task_b_train)
    idx = 0
    for step in range(1, finetune_steps + 1):
        batch_exs = task_b_train[idx: idx + batch_size]
        if not batch_exs:
            _random.shuffle(task_b_train)
            idx = 0
            batch_exs = task_b_train[idx: idx + batch_size]
        ids  = torch.stack([e["input_ids"]      for e in batch_exs]).to(device)
        mask = torch.stack([e["attention_mask"]  for e in batch_exs]).to(device)
        loss = causal_loss(model, ids, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        idx += batch_size

        if step % eval_every == 0:
            # Evaluate anchor losses
            current_losses = per_example_loss(model, task_a_anchors)
            anchor_set.update_current_losses({i: current_losses[i] for i in range(n_anchors)})
            forgetting_score = anchor_set.forgetting_scores().get("wikitext", 0.0)
            anchor_ppl = float(np.exp(np.mean(current_losses)))

            alert = forgetting_score > alert_threshold
            ppl_rose = anchor_ppl > ppl_rise_threshold

            if alert and first_alert_step is None:
                first_alert_step = step
            if ppl_rose and first_ppl_rise_step is None:
                first_ppl_rise_step = step

            marker = ""
            if alert:
                marker += " ← ALERT"
            if ppl_rose and first_ppl_rise_step == step:
                marker += " ← PPL RISE"

            print(f"  {step:>6}  {forgetting_score:>12.4f}  {anchor_ppl:>12.2f}{marker}")

            history.append({
                "step": step,
                "forgetting_score": round(forgetting_score, 6),
                "anchor_ppl": round(anchor_ppl, 4),
                "alert_fired": alert,
                "ppl_rise": ppl_rose,
            })

    # -----------------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------------
    final_losses = per_example_loss(model, task_a_anchors)
    final_ppl = float(np.exp(np.mean(final_losses)))

    print(f"\n{'='*60}")
    print("reservoir-anchor: Forgetting Monitor Results (Real LLM)")
    print(f"{'='*60}")
    print(f"  Model          : {model_name}")
    print(f"  Task A dataset : wikitext-103")
    print(f"  Task B dataset : code_search_net (python)")
    print(f"  Anchors        : {n_anchors}")
    print(f"  Baseline PPL   : {baseline_ppl:.2f}")
    print(f"  Final PPL      : {final_ppl:.2f}")
    print(f"  PPL increase   : {(final_ppl/baseline_ppl - 1)*100:.1f}%")

    lead_time = None
    if first_alert_step is not None and first_ppl_rise_step is not None:
        lead_time = first_ppl_rise_step - first_alert_step
        print(f"\n  Alert fired at step  : {first_alert_step}")
        print(f"  PPL rise at step     : {first_ppl_rise_step}")
        print(f"  Lead time            : {lead_time} steps "
              f"({lead_time/finetune_steps*100:.1f}% of fine-tune)")
        if lead_time > 0:
            print(f"\n  ✓ Lead time confirmed")
        else:
            print(f"\n  ✗ No lead time (simultaneous)")
    elif first_alert_step is not None:
        print(f"\n  Alert fired at step {first_alert_step}, PPL never rose >25%")
    else:
        print(f"\n  No alert fired (threshold too high or no forgetting)")
    print(f"{'='*60}\n")

    return {
        "model": model_name,
        "n_anchors": n_anchors,
        "finetune_steps": finetune_steps,
        "baseline_ppl": round(baseline_ppl, 4),
        "final_ppl": round(final_ppl, 4),
        "ppl_increase_pct": round((final_ppl / baseline_ppl - 1) * 100, 2),
        "first_alert_step": first_alert_step,
        "first_ppl_rise_step": first_ppl_rise_step,
        "lead_time_steps": lead_time,
        "alert_threshold": alert_threshold,
        "history": history,
    }


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    results = run_forgetting.remote()
    print("\nFull results JSON:")
    print(json.dumps(results, indent=2))
