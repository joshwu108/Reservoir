"""Tests for the checker's recomputation of draws and importance weights.

When ``decay_config`` records the buffer's ``seed``, ``buffer_id``, ``alpha``
and ``beta``, the checker re-derives every ``draw_int`` from the keyed
BLAKE2b definition (``checker/draw.py``, which shares no code with
``reservoir.draw``) and every importance weight from the declared formula.
A forged draw that stays inside the recorded leaf's range, or a reweighted
sample, is then rejected; before this, both passed. Logs without the
fields verify as before.
"""

from __future__ import annotations

import copy
import json
from fractions import Fraction
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from checker.diff import diff_logs
from checker.draw import draw_uniform_below as checker_draw
from checker.verify import CheckerError, verify_chain, verify_json_lines
from reservoir.attest import AttestationLog, _digest_record
from reservoir.draw import draw_uniform_below as library_draw
from reservoir.rollout import Rollout
from reservoir.rollout_buffer import RolloutBuffer

RESULTS = Path(__file__).parents[1] / "benchmarks" / "modal" / "results"


def rollouts(rewards: list[float]) -> list[Rollout]:
    return [Rollout(tokens=[1, 2], logprobs=[-0.1, -0.2], reward=r) for r in rewards]


def rechain(records: list[dict], start: int = 0) -> list[dict]:
    out = copy.deepcopy(records)
    prev = out[start - 1]["digest"] if start > 0 else "genesis"
    for rec in out[start:]:
        rec["prev_digest"] = prev
        rec["digest"] = _digest_record(rec)
        prev = rec["digest"]
    return out


def build_log(seed: int = 3, buffer_id: int = 7, beta: float = 0.4) -> list[dict]:
    log = AttestationLog()
    buf = RolloutBuffer(capacity=8, half_life=1, max_policy_age=2, seed=seed, buffer_id=buffer_id,
                        beta=beta, attest=log)
    for v in range(8):
        buf.add_group(f"g{v}", v, rollouts([1.0, 0.0, 0.5]), source="s")
        buf.sample(3, current_version=v)
    assert buf.n_rebases >= 1
    return [dict(r) for r in log.records]


def first_sample(records: list[dict]) -> int:
    return next(i for i, r in enumerate(records) if r["op"] == "sample" and len(r["samples"]) > 1)


class TestIndependentDerivation:
    @settings(max_examples=300)
    @given(
        n=st.integers(min_value=1, max_value=1 << 70),
        seed=st.integers(min_value=0, max_value=(1 << 64) - 1),
        buffer_id=st.integers(min_value=0, max_value=(1 << 64) - 1),
        counter=st.integers(min_value=0, max_value=(1 << 64) - 1),
    )
    def test_matches_library(self, n, seed, buffer_id, counter) -> None:
        assert checker_draw(n, seed, buffer_id, counter) == library_draw(n, seed, buffer_id, counter)

    def test_known_answer(self) -> None:
        # Pinned values, so a change to the definition is caught even if both sides change together.
        assert checker_draw(1000, 0, 0, 0) == 469
        assert checker_draw(3 * (1 << 254), 1, 2, 3) == \
            85013347618758328123349507276990521947358918500787807600310725381725066559567
        assert checker_draw(1 << 256, 7, 7, 7) == \
            1936313988289704038115521225734860625083037098012813944748441275482897865568
        assert checker_draw(1, 5, 5, 5) == 0

    @given(seed=st.integers(min_value=0, max_value=1 << 40), counter=st.integers(min_value=0, max_value=1 << 40))
    def test_matches_library_where_rejection_is_frequent(self, seed, counter) -> None:
        # With n = 3 * 2^254 a quarter of the 256-bit blocks are rejected.
        n = 3 * (1 << 254)
        assert checker_draw(n, seed, 0, counter) == library_draw(n, seed, 0, counter)

    @pytest.mark.parametrize("bad", [0, -1, (1 << 256) + 1, 1.5, True])
    def test_bad_bound_rejected(self, bad) -> None:
        with pytest.raises(CheckerError):
            checker_draw(bad, 0, 0, 0)

    def test_bad_key_material_rejected(self) -> None:
        with pytest.raises(CheckerError, match="seed"):
            checker_draw(10, 1 << 64, 0, 0)
        with pytest.raises(CheckerError, match="draw counter"):
            checker_draw(10, 0, 0, -1)


class TestConfigFields:
    def test_buffer_records_the_draw_config(self) -> None:
        cfg = build_log(seed=11, buffer_id=4, beta=0.5)[0]
        assert cfg["seed"] == "11" and cfg["buffer_id"] == "4"
        assert cfg["alpha"] == (1.0).hex() and cfg["beta"] == (0.5).hex()

    def test_log_with_draw_config_verifies(self) -> None:
        verify_chain(build_log())

    def test_legacy_config_without_draw_fields_still_verifies(self) -> None:
        records = build_log()
        for name in ("seed", "buffer_id", "alpha", "beta"):
            del records[0][name]
        verify_chain(rechain(records))

    def test_committed_phase2_log_still_verifies(self) -> None:
        logs = sorted(RESULTS.glob("trl_replay_*.attest.jsonl"))
        assert logs
        for log in logs:
            verify_json_lines(log.read_text())

    def test_decay_log_without_draw_config_advances_the_counter_harmlessly(self) -> None:
        records = build_log()
        for name in ("seed", "buffer_id", "alpha", "beta"):
            del records[0][name]
        result = verify_chain(rechain(records))
        assert sum(len(r["samples"]) for r in records if r["op"] == "sample") == len(result.content.samples)

    @pytest.mark.parametrize("missing", ["seed", "buffer_id", "alpha", "beta"])
    def test_partial_draw_config_rejected(self, missing) -> None:
        records = build_log()
        del records[0][missing]
        with pytest.raises(CheckerError, match="all of"):
            verify_chain(rechain(records))

    @pytest.mark.parametrize("field, value, message", [
        ("seed", "-1", "seed"), ("seed", 3, "seed"), ("seed", str(1 << 64), "64 bits"),
        ("alpha", "1.0", "canonical"), ("alpha", "0x1p+0", "canonical"), ("alpha", (0.0).hex(), "alpha must be"),
        ("beta", (-0.5).hex(), "beta >= 0"), ("beta", (float("inf")).hex(), "canonical"),
    ])
    def test_malformed_draw_config_rejected(self, field, value, message) -> None:
        records = build_log()
        records[0][field] = value
        with pytest.raises(CheckerError, match=message):
            verify_chain(rechain(records))


class TestRejectsForgedDraws:
    def _moved_within_leaf(self, records: list[dict]) -> tuple[list[dict], int]:
        """Move one draw to another value that lands in the same leaf (found by replaying the tree)."""
        from checker.verify import _SumTree
        i = first_sample(records)
        tree = _SumTree(8)
        for r in records[:i]:
            if r["op"] in ("insert", "update", "evict"):
                tree.update(r["index"], int(r["new_priority_int"]))
        s = records[i]["samples"][0]
        draw, leaf = int(s["draw_int"]), s["leaf_index"]
        for candidate in (draw + 1, draw - 1):
            if 0 <= candidate < int(records[i]["root_total"]) and tree.prefix_sum_locate(candidate) == leaf:
                forged = copy.deepcopy(records)
                forged[i]["samples"][0]["draw_int"] = str(candidate)
                return forged, i
        pytest.skip("the first draw sits on a leaf boundary; no same-leaf neighbour")

    def test_draw_moved_within_its_leaf_passes_without_seed_and_fails_with_it(self) -> None:
        forged, i = self._moved_within_leaf(build_log())
        legacy = copy.deepcopy(forged)
        for name in ("seed", "buffer_id", "alpha", "beta"):
            del legacy[0][name]
        verify_chain(rechain(legacy))          # the range check alone cannot see it
        with pytest.raises(CheckerError, match="not the keyed draw"):
            verify_chain(rechain(forged, i))

    def test_changed_seed_is_rejected_at_the_first_sample(self) -> None:
        records = build_log(seed=3)
        records[0]["seed"] = "4"
        with pytest.raises(CheckerError, match="not the keyed draw"):
            verify_chain(rechain(records))

    def test_changed_buffer_id_is_rejected(self) -> None:
        records = build_log(buffer_id=7)
        records[0]["buffer_id"] = "8"
        with pytest.raises(CheckerError, match="not the keyed draw"):
            verify_chain(rechain(records))

    def test_reweighted_sample_is_rejected(self) -> None:
        records = build_log()
        i = first_sample(records)
        s = records[i]["samples"][0]
        w = Fraction(int(s["is_weight_num"]), int(s["is_weight_den"]))
        forged = w / 2 if w > Fraction(1, 2) else w * 2 if w * 2 <= 1 else w / 3
        s["is_weight_num"], s["is_weight_den"] = str(forged.numerator), str(forged.denominator)
        with pytest.raises(CheckerError, match="declared formula"):
            verify_chain(rechain(records, i))

    def test_changed_beta_is_rejected(self) -> None:
        records = build_log(beta=0.4)
        records[0]["beta"] = (0.5).hex()
        with pytest.raises(CheckerError, match="declared formula"):
            verify_chain(rechain(records))

    @pytest.mark.parametrize("beta", [2000.0, 1e308])
    def test_absurd_beta_is_a_checker_error_not_a_traceback(self, beta) -> None:
        records = build_log()
        records[0]["beta"] = beta.hex()
        with pytest.raises(CheckerError, match="unrepresentable|declared formula"):
            verify_chain(rechain(records))

    def test_deleted_sample_record_breaks_the_draw_counter(self) -> None:
        # Removing a whole sample record keeps every remaining draw in range but
        # shifts the counter every later draw is keyed on.
        records = build_log()
        i = first_sample(records)
        del records[i]
        with pytest.raises(CheckerError, match="not the keyed draw"):
            verify_chain(rechain(records, i))

    def test_beta_zero_weights_are_one(self) -> None:
        records = build_log(beta=0.0)
        verify_chain(records)
        for r in records:
            if r["op"] == "sample":
                assert all(s["is_weight_num"] == "1" and s["is_weight_den"] == "1" for s in r["samples"])


class TestDiffWithSeedInConfig:
    def test_different_seeds_are_a_config_difference(self) -> None:
        a, b = build_log(seed=3), build_log(seed=4)
        result = diff_logs(a, b)
        assert result["class"] == "config" and result["first_difference"] == 0
        assert result["differing_fields"] == ["seed"]
