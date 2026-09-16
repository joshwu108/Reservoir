"""
reservoir.forgetting_monitor — HuggingFace TrainerCallback for catastrophic
forgetting detection.

Attach to any HuggingFace Trainer to get real-time forgetting alerts during
fine-tuning. Optionally replays most-forgotten anchors via PER.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from reservoir.anchor_set import AnchorSet, AnchorExample

try:
    from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
except ImportError:  # pragma: no cover
    TrainerCallback = object  # type: ignore[assignment,misc]
    TrainerControl = object  # type: ignore[assignment]
    TrainerState = object  # type: ignore[assignment]
    TrainingArguments = object  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ForgettingAlert:
    step: int
    tag: str
    forgetting_score: float
    baseline_score: float
    threshold: float
    message: str


@dataclass
class GroupReport:
    tag: str
    n_anchors: int
    baseline_mean_loss: float
    final_mean_loss: float
    final_forgetting_score: float
    max_forgetting_score: float
    most_forgotten_examples: list[AnchorExample]


@dataclass
class ForgettingReport:
    total_steps: int
    n_anchors: int
    n_alerts: int
    groups: dict[str, GroupReport]
    alerts: list[ForgettingAlert]

    def to_json(self, path: str) -> None:
        data = {
            "total_steps": self.total_steps,
            "n_anchors": self.n_anchors,
            "n_alerts": self.n_alerts,
            "alerts": [
                {
                    "step": a.step,
                    "tag": a.tag,
                    "forgetting_score": a.forgetting_score,
                    "threshold": a.threshold,
                    "message": a.message,
                }
                for a in self.alerts
            ],
            "groups": {
                tag: {
                    "n_anchors": gr.n_anchors,
                    "baseline_mean_loss": gr.baseline_mean_loss,
                    "final_mean_loss": gr.final_mean_loss,
                    "final_forgetting_score": gr.final_forgetting_score,
                    "max_forgetting_score": gr.max_forgetting_score,
                }
                for tag, gr in self.groups.items()
            },
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def to_html(self, path: str) -> None:
        lines = [
            "<!DOCTYPE html>",
            "<html><head><title>Forgetting Report</title></head><body>",
            f"<h1>Forgetting Report</h1>",
            f"<p>Total steps: {self.total_steps} | Anchors: {self.n_anchors} | Alerts: {self.n_alerts}</p>",
            "<h2>Per-Group Forgetting Scores</h2><table border='1'>",
            "<tr><th>Tag</th><th>N</th><th>Baseline Loss</th><th>Final Loss</th>"
            "<th>Final Score</th><th>Max Score</th></tr>",
        ]
        for tag, gr in self.groups.items():
            lines.append(
                f"<tr><td>{tag}</td><td>{gr.n_anchors}</td>"
                f"<td>{gr.baseline_mean_loss:.4f}</td><td>{gr.final_mean_loss:.4f}</td>"
                f"<td>{gr.final_forgetting_score:.4f}</td><td>{gr.max_forgetting_score:.4f}</td></tr>"
            )
        lines.append("</table>")
        lines.append("<h2>Alert Timeline</h2>")
        if self.alerts:
            lines.append("<ul>")
            for a in self.alerts:
                lines.append(f"<li>Step {a.step}: [{a.tag}] {a.message}</li>")
            lines.append("</ul>")
        else:
            lines.append("<p>No alerts fired.</p>")
        lines.append("<h2>Most Forgotten Per Group</h2>")
        for tag, gr in self.groups.items():
            lines.append(f"<h3>{tag}</h3><ol>")
            for ex in gr.most_forgotten_examples:
                lines.append(
                    f"<li>idx={ex.idx} priority={ex.priority:.4f} "
                    f"baseline={ex.baseline_loss:.4f} current={ex.current_loss:.4f}</li>"
                )
            lines.append("</ol>")
        lines.append("</body></html>")
        with open(path, "w") as f:
            f.write("\n".join(lines))

    def plot(self, path: str) -> None:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            return
        fig, ax = plt.subplots()
        for tag, gr in self.groups.items():
            ax.bar(tag, gr.final_forgetting_score, label=tag)
        ax.set_ylabel("Final Forgetting Score")
        ax.set_title("Forgetting by Group")
        ax.legend()
        fig.savefig(path)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Monitor
# ---------------------------------------------------------------------------

class ForgettingMonitor(TrainerCallback):
    """
    HuggingFace TrainerCallback that monitors for catastrophic forgetting.

    Parameters
    ----------
    anchor_sets : list[AnchorSet]
    eval_every_n_steps : int
    alert_threshold : float
    auto_replay : bool
    replay_ratio : float
    verbose : bool
    log_to_wandb : bool
    tokenizer : optional tokenizer for raw text anchors
    """

    def __init__(
        self,
        anchor_sets: list[AnchorSet],
        eval_every_n_steps: int = 500,
        alert_threshold: float = 0.5,
        auto_replay: bool = False,
        replay_ratio: float = 0.1,
        verbose: bool = True,
        log_to_wandb: bool = False,
        tokenizer=None,
    ) -> None:
        self.anchor_sets = anchor_sets
        self.eval_every_n_steps = eval_every_n_steps
        self.alert_threshold = alert_threshold
        self.auto_replay = auto_replay
        self.replay_ratio = replay_ratio
        self.verbose = verbose
        self.log_to_wandb = log_to_wandb
        self.tokenizer = tokenizer

        self._alerts: list[ForgettingAlert] = []
        # {tag: [(step, score), ...]}
        self._history: dict[str, list[tuple[int, float]]] = {}
        # {tag: max_score_seen}
        self._max_scores: dict[str, float] = {}
        self._last_step: int = 0

        self._replay_scheduler = None
        if auto_replay:
            from reservoir.replay_scheduler import ReplayScheduler
            self._replay_scheduler = ReplayScheduler(
                anchor_sets=anchor_sets,
                replay_ratio=replay_ratio,
            )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def alerts(self) -> list[ForgettingAlert]:
        return self._alerts

    def forgetting_history(self) -> dict[str, list[tuple[int, float]]]:
        return self._history

    def get_report(self) -> ForgettingReport:
        groups: dict[str, GroupReport] = {}
        total_anchors = 0
        for anchor_set in self.anchor_sets:
            for tag, anchors in anchor_set.groups().items():
                baseline_losses = [a.baseline_loss for a in anchors]
                current_losses = [a.current_loss for a in anchors]
                baseline_mean = sum(baseline_losses) / len(baseline_losses) if baseline_losses else 0.0
                final_mean = sum(current_losses) / len(current_losses) if current_losses else 0.0
                scores = anchor_set.forgetting_scores()
                final_score = scores.get(tag, 0.0)
                max_score = self._max_scores.get(tag, 0.0)
                most_forgotten = sorted(anchors, key=lambda a: a.priority, reverse=True)[:5]
                groups[tag] = GroupReport(
                    tag=tag,
                    n_anchors=len(anchors),
                    baseline_mean_loss=baseline_mean,
                    final_mean_loss=final_mean,
                    final_forgetting_score=final_score,
                    max_forgetting_score=max_score,
                    most_forgotten_examples=most_forgotten,
                )
                total_anchors += len(anchors)

        return ForgettingReport(
            total_steps=self._last_step,
            n_anchors=total_anchors,
            n_alerts=len(self._alerts),
            groups=groups,
            alerts=list(self._alerts),
        )

    # ------------------------------------------------------------------
    # TrainerCallback hooks
    # ------------------------------------------------------------------

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        losses = self._compute_anchor_losses(model, args)
        total_anchors = 0
        unique_groups: set[str] = set()
        for anchor_set in self.anchor_sets:
            anchor_set.snapshot_baseline(losses)
            total_anchors += len(anchor_set)
            unique_groups.update(anchor_set.groups().keys())
        print(
            f"[ForgettingMonitor] Anchor baseline established: "
            f"{total_anchors} anchors across {len(unique_groups)} groups"
        )

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        self._last_step = step
        if step % self.eval_every_n_steps != 0:
            return

        losses = self._compute_anchor_losses(model, args)

        for anchor_set in self.anchor_sets:
            anchor_set.update_current_losses(losses)
            scores = anchor_set.forgetting_scores()

            for tag, score in scores.items():
                # Record history
                self._history.setdefault(tag, []).append((step, score))
                self._max_scores[tag] = max(self._max_scores.get(tag, 0.0), score)

                # Verbose output
                if self.verbose:
                    print(
                        f"[ForgettingMonitor] step={step} tag={tag} "
                        f"forgetting_score={score:.4f}"
                    )

                # Alert
                if score > self.alert_threshold:
                    msg = (
                        f"Catastrophic forgetting detected in group '{tag}' at step {step}: "
                        f"forgetting_score={score:.4f} > threshold={self.alert_threshold:.4f}"
                    )
                    alert = ForgettingAlert(
                        step=step,
                        tag=tag,
                        forgetting_score=score,
                        baseline_score=0.0,
                        threshold=self.alert_threshold,
                        message=msg,
                    )
                    self._alerts.append(alert)
                    print(f"[ForgettingMonitor] WARNING: {msg}")

                # wandb logging
                if self.log_to_wandb:
                    try:
                        import wandb
                        wandb.log({f"forgetting/{tag}": score, "step": step})
                    except ImportError:
                        pass

        if self.auto_replay and self._replay_scheduler is not None:
            for anchor_set in self.anchor_sets:
                self._replay_scheduler.update_priorities(anchor_set)

    def on_train_end(self, args, state, control, **kwargs):
        print("[ForgettingMonitor] Training complete. Final forgetting summary:")
        for anchor_set in self.anchor_sets:
            scores = anchor_set.forgetting_scores()
            for tag, score in scores.items():
                print(f"  {tag}: final_forgetting_score={score:.4f}")
        print(f"[ForgettingMonitor] Total alerts: {len(self._alerts)}")
        print("[ForgettingMonitor] To generate full report: monitor.get_report()")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_anchor_losses(self, model, args) -> dict[int, float]:
        """Run a forward pass on each anchor and return {anchor_idx: loss}."""
        device = getattr(args, "device", "cpu")
        if isinstance(device, str):
            device = torch.device(device)

        losses: dict[int, float] = {}
        model.eval()
        with torch.no_grad():
            for anchor_set in self.anchor_sets:
                for anchor in anchor_set:
                    loss = self._compute_single_loss(model, anchor, device)
                    losses[anchor.idx] = loss
        model.train()
        return losses

    def _compute_single_loss(self, model, anchor: AnchorExample, device) -> float:
        """Compute cross-entropy loss for a single anchor example."""
        data = anchor.data

        # Try to get tokenized inputs from data dict
        if "input_ids" in data:
            inputs = {k: v for k, v in data.items()}
            # Convert lists to tensors if needed
            for k, v in inputs.items():
                if isinstance(v, (list, tuple)):
                    inputs[k] = torch.tensor([v], dtype=torch.long).to(device)
                elif isinstance(v, torch.Tensor):
                    inputs[k] = v.unsqueeze(0).to(device) if v.dim() == 1 else v.to(device)
        elif self.tokenizer is not None:
            text = data.get("text", data.get("input", str(data)))
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
        else:
            # Fallback: return 0 loss if we can't compute
            return 0.0

        # Determine if causal LM or classifier
        is_causal_lm = self._is_causal_lm(model)

        try:
            if is_causal_lm:
                # For causal LM, labels = input_ids (shift handled by model)
                inputs_with_labels = {**inputs, "labels": inputs["input_ids"].clone()}
                outputs = model(**inputs_with_labels)
            else:
                # Sequence classification
                label = data.get("label", data.get("labels", 0))
                label_tensor = torch.tensor([[label]], dtype=torch.long).to(device)
                outputs = model(**inputs, labels=label_tensor)

            if hasattr(outputs, "loss") and outputs.loss is not None:
                return float(outputs.loss.item())
        except Exception:
            pass

        # Fallback: just call model and use loss if present
        try:
            outputs = model(**inputs)
            if hasattr(outputs, "loss") and outputs.loss is not None:
                return float(outputs.loss.item())
        except Exception:
            pass

        return 0.0

    def _is_causal_lm(self, model) -> bool:
        """Detect if model is a causal LM or sequence classifier."""
        try:
            cfg = model.config
            num_labels = getattr(cfg, "num_labels", 1)
            if num_labels > 1:
                return False
            model_type = getattr(cfg, "model_type", "")
            # Common causal LM types
            causal_types = {"gpt2", "gpt_neo", "gpt_neox", "llama", "mistral",
                            "falcon", "bloom", "opt", "gemma", "phi"}
            if model_type.lower() in causal_types:
                return True
        except Exception:
            pass
        return True  # default to causal LM
