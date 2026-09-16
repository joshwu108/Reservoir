"""reservoir.report — Bucket training examples by trajectory noise type."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from reservoir.trajectory import TrajectoryFeatures


class NoiseLabel(Enum):
    FLIPPED = "flipped"
    AMBIGUOUS = "ambiguous"
    CLEAN = "clean"


@dataclass
class ExampleReport:
    example_idx: int
    label: NoiseLabel
    confidence: float
    features: "TrajectoryFeatures"


class PreferenceQualityReport:
    """Buckets training examples into noise categories based on trajectory.

    Parameters
    ----------
    features : dict[int, TrajectoryFeatures] — from TrajectoryLogger.get_all_features()
    """

    def __init__(self, features: dict[int, "TrajectoryFeatures"]) -> None:
        self._all: list[ExampleReport] = []

        if not features:
            return

        # Extract arrays for threshold computation
        all_mean_loss = np.array([f.mean_loss_last_k for f in features.values()])
        all_variance = np.array([f.variance for f in features.values()])

        median_mean_loss = float(np.median(all_mean_loss))
        p75_var = float(np.percentile(all_variance, 75))

        # Compute all confidences for CLEAN label fallback
        all_reports: list[ExampleReport] = []

        for idx, feat in features.items():
            slope = feat.slope
            variance = feat.variance
            mean_loss = feat.mean_loss_last_k

            # FLIPPED confidence
            flipped_conf = float(np.clip((slope - 0.01) / 0.10, 0.0, 1.0))

            # AMBIGUOUS confidence
            if p75_var > 0:
                ambiguous_conf = float(np.clip((variance - p75_var) / p75_var, 0.0, 1.0))
            else:
                ambiguous_conf = 0.0

            # Bucketing rules (exact as specified)
            if slope > 0.01 and mean_loss > median_mean_loss:
                label = NoiseLabel.FLIPPED
                confidence = flipped_conf
            elif variance > p75_var and -0.01 <= slope <= 0.01:
                label = NoiseLabel.AMBIGUOUS
                confidence = ambiguous_conf
            else:
                label = NoiseLabel.CLEAN
                confidence = 1.0 - max(flipped_conf, ambiguous_conf)

            all_reports.append(ExampleReport(
                example_idx=idx,
                label=label,
                confidence=float(np.clip(confidence, 0.0, 1.0)),
                features=feat,
            ))

        self._all = all_reports

    @property
    def flipped(self) -> list[ExampleReport]:
        result = [r for r in self._all if r.label == NoiseLabel.FLIPPED]
        return sorted(result, key=lambda r: r.confidence, reverse=True)

    @property
    def ambiguous(self) -> list[ExampleReport]:
        result = [r for r in self._all if r.label == NoiseLabel.AMBIGUOUS]
        return sorted(result, key=lambda r: r.confidence, reverse=True)

    @property
    def clean(self) -> list[ExampleReport]:
        return [r for r in self._all if r.label == NoiseLabel.CLEAN]

    def summary(self) -> dict:
        n_total = len(self._all)
        n_flipped = len(self.flipped)
        n_ambiguous = len(self.ambiguous)
        n_clean = len(self.clean)

        if n_total > 0:
            pct_flipped = 100.0 * n_flipped / n_total
            pct_ambiguous = 100.0 * n_ambiguous / n_total
            pct_clean = 100.0 * n_clean / n_total
        else:
            pct_flipped = pct_ambiguous = pct_clean = 0.0

        return {
            "n_total": n_total,
            "n_flipped": n_flipped,
            "n_ambiguous": n_ambiguous,
            "n_clean": n_clean,
            "pct_flipped": pct_flipped,
            "pct_ambiguous": pct_ambiguous,
            "pct_clean": pct_clean,
        }

    def to_json(self, path: str) -> None:
        """Write report to JSON. Integer example indices stored as strings for precision."""
        s = self.summary()
        examples = []
        for r in self._all:
            examples.append({
                "idx": str(r.example_idx),
                "label": r.label.value,
                "confidence": r.confidence,
                "slope": r.features.slope,
                "variance": r.features.variance,
                "mean_loss_last_k": r.features.mean_loss_last_k,
                "n_observations": r.features.n_observations,
                "first_correct_step": r.features.first_correct_step,
            })
        data = {"summary": s, "examples": examples}
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def to_csv(self, path: str) -> None:
        """Write one row per example to CSV."""
        fieldnames = [
            "idx", "label", "confidence", "slope", "variance",
            "mean_loss_last_k", "n_observations", "first_correct_step",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self._all:
                writer.writerow({
                    "idx": r.example_idx,
                    "label": r.label.value,
                    "confidence": r.confidence,
                    "slope": r.features.slope,
                    "variance": r.features.variance,
                    "mean_loss_last_k": r.features.mean_loss_last_k,
                    "n_observations": r.features.n_observations,
                    "first_correct_step": r.features.first_correct_step,
                })

    def to_html(self, path: str) -> None:
        """Write HTML report with summary and top-50 FLIPPED/AMBIGUOUS tables."""
        s = self.summary()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        def _row(r: ExampleReport) -> str:
            return (
                "<tr>"
                + f"<td>{r.example_idx}</td>"
                + f"<td>{r.label.value}</td>"
                + f"<td>{r.confidence:.4f}</td>"
                + f"<td>{r.features.slope:.6f}</td>"
                + f"<td>{r.features.variance:.6f}</td>"
                + f"<td>{r.features.mean_loss_last_k:.6f}</td>"
                + f"<td>{r.features.n_observations}</td>"
                + f"<td>{r.features.first_correct_step}</td>"
                + "</tr>"
            )

        def _table(rows: list[ExampleReport], title: str) -> str:
            header = (
                "<table border='1'>"
                "<caption>" + title + "</caption>"
                "<tr>"
                "<th>idx</th><th>label</th><th>confidence</th>"
                "<th>slope</th><th>variance</th><th>mean_loss_last_k</th>"
                "<th>n_observations</th><th>first_correct_step</th>"
                "</tr>"
            )
            body = "".join(_row(r) for r in rows)
            return header + body + "</table>"

        html = (
            "<!DOCTYPE html>"
            "<html>"
            "<head><meta charset='utf-8'><title>Preference Quality Report</title></head>"
            "<body>"
            "<h1>Preference Quality Report</h1>"
            "<h2>Summary</h2>"
            "<table border='1'>"
            "<tr><th>Bucket</th><th>Count</th><th>Percentage</th></tr>"
            + f"<tr><td>FLIPPED</td><td>{s['n_flipped']}</td><td>{s['pct_flipped']:.1f}%</td></tr>"
            + f"<tr><td>AMBIGUOUS</td><td>{s['n_ambiguous']}</td><td>{s['pct_ambiguous']:.1f}%</td></tr>"
            + f"<tr><td>CLEAN</td><td>{s['n_clean']}</td><td>{s['pct_clean']:.1f}%</td></tr>"
            + f"<tr><td><b>TOTAL</b></td><td>{s['n_total']}</td><td>100.0%</td></tr>"
            "</table>"
            + _table(self.flipped[:50], "Top-50 FLIPPED Examples")
            + _table(self.ambiguous[:50], "Top-50 AMBIGUOUS Examples")
            + f"<p>Generated: {now} | Dataset size: {s['n_total']}</p>"
            "</body></html>"
        )

        with open(path, "w") as f:
            f.write(html)
