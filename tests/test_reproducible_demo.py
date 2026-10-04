"""Two GRPO runs through the real TRL trainer must produce one attestation transcript.

These tests need TRL and the tiny test model TRL itself uses (downloaded
once into the HF cache); they are skipped when TRL is not installed. They
run on CPU with HF ``generate`` under a fixed seed, which is deterministic,
so the whole pipeline (generation, rewards, advantages, Reservoir
store/draw/evict) is reproducible and the log is its fingerprint. A run
with a different data seed must differ, and ``checker.diff`` must say the
difference is in the stored data, not in the sampler.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

for _module in ("trl", "transformers", "datasets", "modal"):
    pytest.importorskip(_module)

from checker.diff import diff_logs  # noqa: E402
from checker.transcript import build_transcript  # noqa: E402
from checker.verify import verify_json_lines  # noqa: E402
from benchmarks.modal.trl_replay_real import DATASET_ID  # noqa: E402
from demo.reproducible_grpo import run_once  # noqa: E402

STEPS = 3
SOURCE = DATASET_ID.rsplit("/", 1)[-1]


def _records(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l]


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict]:
    root = tmp_path_factory.mktemp("repro")
    try:
        return {
            "a": run_once(root / "a", max_steps=STEPS, seed=42, buffer_seed=0),
            "b": run_once(root / "b", max_steps=STEPS, seed=42, buffer_seed=0),
            "c": run_once(root / "c", max_steps=STEPS, seed=43, buffer_seed=0),
        }
    except (OSError, ConnectionError) as exc:
        # The tiny model and dataset come from the HF Hub; without them (or
        # offline) this is an environment limitation, not a failure.
        pytest.skip(f"model or dataset unavailable: {exc}")


def test_same_seeds_give_byte_identical_log_and_manifest(runs):
    a, b = runs["a"], runs["b"]
    assert Path(a["attest"]).read_bytes() == Path(b["attest"]).read_bytes()
    assert Path(a["manifest"]).read_bytes() == Path(b["manifest"]).read_bytes()
    assert a["head_digest"] == b["head_digest"]
    assert diff_logs(_records(Path(a["attest"])), _records(Path(b["attest"])))["identical"]


def test_each_log_verifies_with_its_manifest_and_tags_its_source(runs):
    for run in runs.values():
        result = verify_json_lines(Path(run["attest"]).read_text(), manifest=Path(run["manifest"]).read_text())
        assert result.content.manifest_matched == run["totals"]["ingested_rows"] > 0
        report = build_transcript(result)
        assert set(report["sources"]) == {SOURCE}


def test_a_different_data_seed_is_classified_as_data_not_sampler(runs):
    result = diff_logs(_records(Path(runs["a"]["attest"])), _records(Path(runs["c"]["attest"])))
    assert not result["identical"]
    assert result["class"] == "data", result
