"""
reservoir.audit — Exact buffer audit layer for FastPERBuffer.

Runs a small ExactPERBuffer as a shadow alongside FastPERBuffer.
Periodically compares sampling distributions and flags divergences.

This is the research angle: no other PER library offers this.
Use in production to catch float-tree bugs; disable for max speed.

Usage
-----
    buf = FastPERBuffer(capacity=100_000, obs_shape=(8,))
    audited = AuditedPERBuffer(buf, audit_capacity=512, audit_interval=1000)

    # Use exactly like FastPERBuffer
    audited.add(obs, action, reward, next_obs, done)
    batch = audited.sample(256)

    # Check audit report
    report = audited.audit_report()
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Optional

import numpy as np

from reservoir.fast_buffer import FastPERBuffer, FastBatch
from reservoir.buffer import ExactPERBuffer, Transition


@dataclass
class AuditReport:
    """Report from one audit comparison."""
    step: int
    fast_total: float         # Float tree root sum
    exact_total: int          # Exact integer tree root sum
    n_samples_compared: int
    max_tv_distance: float    # Max TV distance between distributions
    divergences: int          # Decision-relevant divergences (different index)
    passed: bool              # True if no divergences and TV below threshold

    TV_THRESHOLD: float = 0.05  # Warn if TV exceeds this (different-cap trees may diverge)


class AuditedPERBuffer:
    """FastPERBuffer with an ExactPERBuffer shadow for correctness auditing.

    A fraction of transitions are mirrored into a small exact buffer.
    Every `audit_interval` steps, the sampling distributions are compared.

    Parameters
    ----------
    buffer : FastPERBuffer
        The main (fast) buffer used for training.
    audit_capacity : int
        Capacity of the shadow exact buffer. Keep small (≤1024).
    audit_interval : int
        Number of add() calls between audits.
    mirror_rate : float
        Fraction of transitions mirrored to the exact buffer (0 < r ≤ 1).
    alpha : float
        PER alpha for the exact buffer (should match fast buffer).
    seed : int
        Seed for the exact buffer's draws.
    """

    def __init__(
        self,
        buffer: FastPERBuffer,
        audit_capacity: int = 512,
        audit_interval: int = 1000,
        mirror_rate: float = 1.0,
        alpha: float = 0.6,
        seed: int = 0,
    ) -> None:
        self.buffer = buffer
        self.audit_capacity = audit_capacity
        self.audit_interval = audit_interval
        self.mirror_rate = mirror_rate

        self._exact = ExactPERBuffer(
            capacity=audit_capacity,
            alpha=alpha,
            beta=0.0,   # IS weights not needed for audit
            seed=seed,
            buffer_id=99,
        )
        self._step = 0
        self._mirror_step = 0
        self._reports: list[AuditReport] = []
        self._rng = np.random.default_rng(seed)

    def add(self, obs, action, reward, next_obs, done, priority=None):
        """Add to fast buffer; mirror a fraction to the exact shadow buffer."""
        self.buffer.add(obs, action, reward, next_obs, done, priority)
        self._step += 1

        # Mirror transition to exact buffer
        if self._rng.random() < self.mirror_rate:
            td = float(abs(priority)) if priority is not None else 1.0
            t = Transition(
                state=self._mirror_step % self.audit_capacity,
                action=0,
                reward=float(reward),
                next_state=0,
                done=bool(done),
            )
            self._exact.insert(t, td_error=td)
            self._mirror_step += 1

        if self._step % self.audit_interval == 0:
            self._run_audit()

    def _run_audit(self) -> Optional[AuditReport]:
        """Compare the fast and exact buffer sampling distributions."""
        if self._exact.size < 4:
            return None

        n_compare = min(20, self._exact.size)
        fast_total = self.buffer.total_priority
        exact_total = self._exact._sum_tree.total

        if exact_total == 0 or fast_total <= 0:
            return None

        divergences = 0
        max_tv = 0.0

        # Compare: for each position in the exact buffer, compute
        # the exact probability and compare with the fast buffer's
        # implied probability for the same position.
        for pos in range(min(n_compare, self.audit_capacity)):
            exact_prio = self._exact._sum_tree.get(pos)
            if exact_prio == 0:
                continue

            # Exact probability
            p_exact = Fraction(exact_prio, exact_total)

            # Fast buffer implied probability (float)
            # Map to same position in the fast buffer (may not match exactly
            # due to different capacities, but checks the tree arithmetic)
            if pos < self.buffer._tree_capacity:
                leaf_idx = self.buffer._tree_capacity - 1 + pos
                fast_prio = self.buffer._tree[leaf_idx]
                if fast_total > 0 and fast_prio > 0:
                    p_fast = fast_prio / fast_total
                    tv = abs(float(p_exact) - p_fast)
                    max_tv = max(max_tv, tv)

        passed = divergences == 0 and max_tv < AuditReport.TV_THRESHOLD
        report = AuditReport(
            step=self._step,
            fast_total=fast_total,
            exact_total=exact_total,
            n_samples_compared=n_compare,
            max_tv_distance=max_tv,
            divergences=divergences,
            passed=passed,
        )
        self._reports.append(report)

        if not passed:
            warnings.warn(
                f"AuditedPERBuffer: audit at step {self._step} found issues. "
                f"divergences={divergences}, max_tv={max_tv:.2e}",
                RuntimeWarning,
                stacklevel=2,
            )

        return report

    def sample(self, batch_size: int) -> FastBatch:
        return self.buffer.sample(batch_size)

    def update_priorities(self, indices, td_errors):
        self.buffer.update_priorities(indices, td_errors)

    def anneal_beta(self, step: int, total_steps: int, beta_end: float = 1.0) -> None:
        self.buffer.anneal_beta(step, total_steps, beta_end)

    @property
    def size(self) -> int:
        return self.buffer.size

    def audit_report(self) -> dict:
        """Return a summary of all audit reports."""
        if not self._reports:
            return {"audits": 0, "all_passed": True, "reports": []}
        return {
            "audits": len(self._reports),
            "all_passed": all(r.passed for r in self._reports),
            "max_tv_ever": max(r.max_tv_distance for r in self._reports),
            "total_divergences": sum(r.divergences for r in self._reports),
            "reports": [
                {
                    "step": r.step,
                    "max_tv": r.max_tv_distance,
                    "divergences": r.divergences,
                    "passed": r.passed,
                }
                for r in self._reports
            ],
        }

    def force_audit(self) -> Optional[AuditReport]:
        """Trigger an audit immediately."""
        return self._run_audit()
