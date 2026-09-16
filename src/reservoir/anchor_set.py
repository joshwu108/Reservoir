"""
reservoir.anchor_set — AnchorSet for catastrophic forgetting detection.

Holds a fixed collection of labeled examples representing prior knowledge.
The ForgettingMonitor uses this to track loss increases during fine-tuning.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class AnchorExample:
    """A single example in an AnchorSet."""

    idx: int                  # position in the anchor set (0-indexed)
    data: dict                # the actual example (tokenized or raw)
    tag: str                  # user-supplied group label e.g. "legal-QA"
    baseline_loss: float = 0.0   # loss at the start of fine-tuning
    current_loss: float = 0.0    # most recently measured loss
    priority: float = 0.0        # current forgetting severity


class AnchorSet:
    """
    A fixed set of examples representing prior knowledge to preserve.

    Parameters
    ----------
    examples : list[dict]
        Raw examples. Each must be a dict with at least one text field.
    tags : str | list[str]
        Group tag(s) for all examples, or one tag per example.
        Default "default".
    n : int | None
        If provided, subsample to n examples using the strategy below.
    strategy : "random" | "priority-stratified"
        How to subsample if n < len(examples). Default "random".
    """

    def __init__(
        self,
        examples: list[dict],
        tags: str | list[str] = "default",
        n: int | None = None,
        strategy: str = "random",
    ) -> None:
        if isinstance(tags, str):
            tags = [tags] * len(examples)

        if len(tags) != len(examples):
            raise ValueError(
                f"tags length {len(tags)} must match examples length {len(examples)}"
            )

        if n is not None and n < len(examples):
            if strategy == "priority-stratified":
                raise ValueError(
                    "priority-stratified strategy requires initial losses which are "
                    "not available at construction time. Call snapshot_baseline first "
                    "or use strategy='random'."
                )
            indices = random.sample(range(len(examples)), n)
            examples = [examples[i] for i in indices]
            tags = [tags[i] for i in indices]

        self._anchors: list[AnchorExample] = [
            AnchorExample(idx=i, data=ex, tag=tag)
            for i, (ex, tag) in enumerate(zip(examples, tags))
        ]

    # ------------------------------------------------------------------
    # Sequence protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._anchors)

    def __iter__(self) -> Iterator[AnchorExample]:
        return iter(self._anchors)

    def __getitem__(self, idx: int) -> AnchorExample:
        return self._anchors[idx]

    # ------------------------------------------------------------------
    # Grouping
    # ------------------------------------------------------------------

    def groups(self) -> dict[str, list[AnchorExample]]:
        """Return examples grouped by tag."""
        result: dict[str, list[AnchorExample]] = {}
        for anchor in self._anchors:
            result.setdefault(anchor.tag, []).append(anchor)
        return result

    # ------------------------------------------------------------------
    # Loss tracking
    # ------------------------------------------------------------------

    def snapshot_baseline(self, losses: dict[int, float]) -> None:
        """Set baseline_loss for each anchor. Call once at fine-tune start."""
        for anchor in self._anchors:
            if anchor.idx in losses:
                anchor.baseline_loss = losses[anchor.idx]

    def update_current_losses(self, losses: dict[int, float]) -> None:
        """Update current_loss and recompute priority for each anchor.

        priority = max(0, (current_loss - baseline_loss) / (baseline_loss + 1e-8))
        """
        for anchor in self._anchors:
            if anchor.idx in losses:
                anchor.current_loss = losses[anchor.idx]
                # Clamp denominator to at least 0.1 so near-zero baselines
                # don't produce astronomically large relative scores.
                denom = max(anchor.baseline_loss, 0.1)
                raw = (anchor.current_loss - anchor.baseline_loss) / denom
                anchor.priority = max(0.0, raw)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def forgetting_scores(self) -> dict[str, float]:
        """Return {tag: mean_priority} for each group."""
        groups = self.groups()
        return {
            tag: sum(a.priority for a in anchors) / len(anchors)
            for tag, anchors in groups.items()
        }

    def most_forgotten(self, k: int = 10) -> list[AnchorExample]:
        """Top-k anchors by priority (most forgotten first)."""
        return sorted(self._anchors, key=lambda a: a.priority, reverse=True)[:k]

    # ------------------------------------------------------------------
    # Class method constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_dataset(
        cls,
        dataset,
        n: int = 1000,
        tags: str | list[str] = "default",
        strategy: str = "random",
    ) -> "AnchorSet":
        """Construct from a HuggingFace Dataset or list of dicts."""
        examples = list(dataset)
        return cls(examples, tags=tags, n=n, strategy=strategy)
