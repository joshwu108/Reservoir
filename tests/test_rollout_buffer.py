"""Tests for reservoir.rollout_buffer — the LLM-RL replay buffer.

Covers the README contract (add_group / sample / update_priorities with
age decay and staleness eviction), determinism of the keyed draw, exact
agreement between the sampling distribution and ``decay.inflated_priority``,
and that every run produces an attestation log the independent checker
accepts — including runs that trigger a rebase.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from fractions import Fraction

import pytest

from checker.verify import CheckerError, verify_json_lines
from reservoir.attest import AttestationLog
from reservoir.decay import DecayParams, inflated_priority, quantize_priority
from reservoir.priorities import AdvantagePriority, PassRateVariance, PriorityStrategy
from reservoir.rollout import Rollout, RolloutGroup
from reservoir.rollout_buffer import RolloutBatch, RolloutBuffer


def rollouts(rewards: list[float], n_tokens: int = 2) -> list[Rollout]:
    return [
        Rollout(tokens=list(range(1, n_tokens + 1)), logprobs=[-0.1] * n_tokens, reward=r)
        for r in rewards
    ]


def make_buffer(**kw) -> RolloutBuffer:
    defaults = dict(capacity=8, half_life=4, max_policy_age=16, seed=0)
    defaults.update(kw)
    return RolloutBuffer(**defaults)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_defaults_match_readme(self) -> None:
        buf = RolloutBuffer(capacity=50_000)
        assert buf.capacity >= 50_000
        assert buf.size == 0
        assert buf.current_version == 0
        assert isinstance(buf.priority, AdvantagePriority)
        assert buf.alpha == 1.0
        assert buf.beta == 0.4
        assert isinstance(buf.params, DecayParams)
        assert buf.params.half_life == 4 and buf.params.max_policy_age == 16

    def test_bit_budget_violation_surfaces_unchanged(self) -> None:
        with pytest.raises(ValueError, match="bit budget"):
            RolloutBuffer(capacity=1 << 24, half_life=1, max_policy_age=40)

    @pytest.mark.parametrize("bad", [0.0, -1.0, math.nan, math.inf])
    def test_bad_alpha_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match="alpha"):
            make_buffer(alpha=bad)

    @pytest.mark.parametrize("bad", [-0.1, math.nan, math.inf])
    def test_bad_beta_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match="beta"):
            make_buffer(beta=bad)

    def test_priority_must_be_a_strategy(self) -> None:
        with pytest.raises(TypeError, match="PriorityStrategy"):
            make_buffer(priority=lambda r, g: 1.0)

    def test_attest_accepts_log_instance(self) -> None:
        log = AttestationLog()
        buf = make_buffer(attest=log)
        assert buf.attestation_log is log

    def test_attest_path_must_not_exist(self, tmp_path) -> None:
        path = tmp_path / "attest.jsonl"
        path.write_text("stale\n")
        with pytest.raises(FileExistsError):
            make_buffer(attest=path)

    def test_attest_accepts_path(self, tmp_path) -> None:
        path = tmp_path / "attest.jsonl"
        buf = make_buffer(attest=path)
        buf.add_group("p", 0, rollouts([1.0, 0.0]))
        buf.close()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            assert json.loads(line)["op"] == "insert"


# ---------------------------------------------------------------------------
# add_group
# ---------------------------------------------------------------------------

class TestAddGroup:
    def test_stores_each_rollout_in_its_own_slot(self) -> None:
        buf = make_buffer()
        idx = buf.add_group("p1", 0, rollouts([1.0, 0.0, 0.5]))
        assert idx == (0, 1, 2)
        assert buf.size == 3
        r, g = buf.entry(1)
        assert r.reward == 0.0
        assert isinstance(g, RolloutGroup)
        assert g.prompt_id == "p1" and g.size == 3
        assert buf.entry(0)[1] is g  # group is shared, not copied per rollout

    def test_priority_pipeline_score_alpha_quantize_inflate(self) -> None:
        buf = make_buffer(alpha=0.5)
        buf.add_group("p", 0, rollouts([1.0, 0.0]))
        g = buf.entry(0)[1]
        raw = AdvantagePriority().score(g.rollouts[0], g)  # 0.5 + 1e-6
        q = quantize_priority(raw ** 0.5, buf.params)
        assert buf.base_priority(0) == q
        assert buf.leaf(0) == inflated_priority(q, 0, 0, buf.params)

    def test_advances_version_to_the_group_version(self) -> None:
        buf = make_buffer()
        buf.add_group("p", 7, rollouts([1.0]))
        assert buf.current_version == 7

    def test_older_group_version_is_accepted_if_not_expired(self) -> None:
        buf = make_buffer(max_policy_age=5)
        buf.add_group("a", 10, rollouts([1.0]))
        buf.add_group("b", 8, rollouts([1.0]))  # async arrival, still live
        assert buf.current_version == 10
        assert buf.entry_version(1) == 8
        with pytest.raises(ValueError, match="expired"):
            buf.add_group("c", 4, rollouts([1.0]))

    def test_group_larger_than_capacity_rejected(self) -> None:
        buf = make_buffer(capacity=4)
        with pytest.raises(ValueError, match="capacity"):
            buf.add_group("p", 0, rollouts([1.0] * 5))
        assert buf.size == 0

    def test_bad_score_leaves_buffer_untouched(self) -> None:
        class Bad(PriorityStrategy):
            def score(self, rollout, group) -> float:
                return -1.0 if rollout.reward == 0.0 else 1.0

        buf = make_buffer(priority=Bad())
        with pytest.raises(ValueError, match="Bad"):
            buf.add_group("p", 0, rollouts([1.0, 0.0]))
        assert buf.size == 0 and buf.total == 0

    def test_priority_out_of_fixed_point_range_names_the_prompt(self) -> None:
        class Huge(PriorityStrategy):
            def score(self, rollout, group) -> float:
                return 1e9

        buf = make_buffer(priority=Huge())
        with pytest.raises(ValueError, match="prompt-xyz"):
            buf.add_group("prompt-xyz", 0, rollouts([1.0]))

    def test_full_buffer_evicts_oldest_version_first(self) -> None:
        # "older" is inserted second but has the lower version, so it is the victim.
        buf = make_buffer(capacity=4, max_policy_age=100)
        buf.add_group("newer", 5, rollouts([1.0, 0.0]))
        buf.add_group("older", 3, rollouts([1.0, 0.0]))
        buf.add_group("latest", 6, rollouts([1.0]))
        prompts = [buf.entry(i)[1].prompt_id for i in buf.live_positions()]
        assert prompts.count("older") == 1 and prompts.count("newer") == 2
        assert "latest" in prompts

    def test_full_buffer_ties_broken_by_insertion_order(self) -> None:
        buf = make_buffer(capacity=4, max_policy_age=100)
        buf.add_group("first", 0, rollouts([1.0, 0.0]))
        buf.add_group("second", 0, rollouts([1.0, 0.0]))
        buf.add_group("third", 0, rollouts([1.0]))
        prompts = [buf.entry(i)[1].prompt_id for i in buf.live_positions()]
        assert prompts.count("first") == 1 and prompts.count("second") == 2

    def test_freed_slots_are_reused_lowest_first(self) -> None:
        buf = make_buffer(capacity=8, max_policy_age=0)
        buf.add_group("a", 0, rollouts([1.0, 0.0, 0.5]))
        buf.add_group("b", 1, rollouts([1.0]))  # group "a" expires → slots 0,1,2 freed
        assert buf.live_positions() == (0,)
        assert buf.entry(0)[1].prompt_id == "b"

    def test_full_buffer_refuses_group_older_than_its_victims(self) -> None:
        buf = make_buffer(capacity=4, max_policy_age=100)
        buf.add_group("a", 5, rollouts([1.0, 0.0]))
        buf.add_group("b", 6, rollouts([1.0, 0.0]))
        before = (buf.live_positions(), buf.total, buf.size)
        with pytest.raises(ValueError, match="never displaces newer"):
            buf.add_group("late", 3, rollouts([1.0]))
        assert (buf.live_positions(), buf.total, buf.size) == before
        buf.add_group("same-age", 5, rollouts([1.0]))  # equal version may evict "a"
        assert buf.size == 4

    def test_full_buffer_counts_stale_evictions_as_free(self) -> None:
        buf = make_buffer(capacity=4, max_policy_age=2)
        buf.add_group("a", 0, rollouts([1.0, 0.0]))
        buf.add_group("b", 1, rollouts([1.0, 0.0]))
        # At version 3 group "a" expires; the new group fits without displacing "b".
        buf.add_group("c", 3, rollouts([1.0, 0.0]))
        prompts = sorted(buf.entry(i)[1].prompt_id for i in buf.live_positions())
        assert prompts == ["b", "b", "c", "c"]

    def test_custom_success_predicate_reaches_the_group(self) -> None:
        buf = make_buffer(priority=PassRateVariance(epsilon=0.0))
        buf.add_group("p", 0, rollouts([0.9, 0.4]), is_success=lambda r: r.reward >= 0.5)
        assert buf.entry(0)[1].pass_rate == 0.5


# ---------------------------------------------------------------------------
# sample
# ---------------------------------------------------------------------------

class TestSample:
    def test_batch_shape_and_fields(self) -> None:
        buf = make_buffer()
        buf.add_group("p", 2, rollouts([1.0, 0.0, 0.5], n_tokens=3))
        batch = buf.sample(batch_size=4, current_version=2)
        assert isinstance(batch, RolloutBatch)
        assert len(batch.indices) == 4
        assert all(0 <= i < buf.capacity for i in batch.indices)
        assert len(batch.rollouts) == len(batch.logprobs) == len(batch.model_versions) == 4
        assert all(len(lp) == 3 for lp in batch.logprobs)
        assert all(v == 2 for v in batch.model_versions)
        assert all(isinstance(w, Fraction) and 0 < w <= 1 for w in batch.is_weights)
        assert len(batch.float_is_weights) == 4
        assert batch.root_total == buf.total
        assert all(batch.priorities[k] == buf.leaf(batch.indices[k]) for k in range(4))
        assert all(batch.groups[k].prompt_id == "p" for k in range(4))
        assert all(batch.rewards[k] == batch.rollouts[k].reward for k in range(4))

    def test_with_replacement_so_batch_may_exceed_live_count(self) -> None:
        buf = make_buffer()
        buf.add_group("p", 0, rollouts([1.0, 0.0]))
        batch = buf.sample(batch_size=10)
        assert len(batch.indices) == 10 and set(batch.indices) <= {0, 1}

    def test_empty_buffer_and_bad_batch_size(self) -> None:
        buf = make_buffer()
        with pytest.raises(RuntimeError, match="no live"):
            buf.sample(batch_size=1)
        for bad in (0, -1, 1.5, True):
            with pytest.raises(ValueError, match="batch_size"):
                buf.sample(batch_size=bad)  # type: ignore[arg-type]

    def test_zero_total_is_an_error(self) -> None:
        buf = make_buffer(priority=AdvantagePriority(epsilon=0.0))
        buf.add_group("p", 0, rollouts([1.0, 1.0]))  # identical rewards → q = 0
        assert buf.total == 0
        with pytest.raises(RuntimeError, match="zero"):
            buf.sample(batch_size=1)

    def test_current_version_advances_and_defaults_to_current(self) -> None:
        buf = make_buffer()
        buf.add_group("p", 0, rollouts([1.0, 0.0]))
        buf.sample(1, current_version=3)
        assert buf.current_version == 3
        buf.sample(1)
        assert buf.current_version == 3
        with pytest.raises(ValueError, match="monoton"):
            buf.sample(1, current_version=2)

    def test_same_seed_same_ops_same_transcript(self) -> None:
        def run() -> tuple[list[list[int]], str]:
            log = AttestationLog()
            buf = make_buffer(seed=7, attest=log)
            buf.add_group("a", 0, rollouts([1.0, 0.0, 0.3]))
            buf.add_group("b", 1, rollouts([0.0, 0.0, 1.0]))
            out = [buf.sample(4, current_version=2).indices]
            buf.update_priorities(out[0], [0.1, 0.9, 0.5, 0.2])
            out.append(buf.sample(4, current_version=3).indices)
            return out, log.to_json_lines()

        assert run() == run()

    def test_draws_never_repeat_across_batches_of_different_sizes(self) -> None:
        buf = make_buffer(seed=5)
        buf.add_group("p", 0, rollouts([1.0, 0.0, 0.3, 0.7, 0.2]))
        seen: list[int] = []
        for size in (4, 2, 3, 1, 4):
            seen.extend(buf.sample(size).draw_integers)
        assert len(seen) == len(set(seen)), "a draw key was reused between batches"

    def test_different_seed_different_draws(self) -> None:
        a = make_buffer(seed=1)
        b = make_buffer(seed=2)
        for buf in (a, b):
            buf.add_group("p", 0, rollouts([1.0, 0.0, 0.3, 0.7, 0.2]))
        assert a.sample(8).draw_integers != b.sample(8).draw_integers

    def test_empirical_frequencies_match_declared_distribution(self) -> None:
        buf = make_buffer(capacity=8, half_life=2, max_policy_age=8, seed=3)
        buf.add_group("old", 0, rollouts([1.0, 0.0]))     # advantage 0.5 each
        buf.add_group("new", 4, rollouts([1.0, 0.0]))     # 2 half-lives later → 4x weight
        buf.add_group("mid", 2, rollouts([0.75, 0.25]))   # 0.25 each, 1 half-life → 2x
        total = buf.total
        declared = {pos: Fraction(buf.leaf(pos), total) for pos in buf.live_positions()}
        # Exact ratio check from the primitive, independent of the buffer.
        q_old, q_new = buf.base_priority(0), buf.base_priority(2)
        assert q_old == q_new
        assert declared[2] / declared[0] == Fraction(4)
        counts: Counter[int] = Counter()
        n_draws = 4000
        for _ in range(n_draws // 8):
            counts.update(buf.sample(8).indices)
        for pos, prob in declared.items():
            assert abs(counts[pos] / n_draws - float(prob)) < 0.03

    def test_is_weights_follow_declared_formula(self) -> None:
        buf = make_buffer(beta=0.5)
        buf.add_group("p", 0, rollouts([1.0, 0.0, 0.5]))
        batch = buf.sample(3)
        n = buf.size
        for k, pos in enumerate(batch.indices):
            p_i = (n * buf.leaf(pos)) / batch.root_total
            p_min = (n * batch.min_priority_int) / batch.root_total
            expected = Fraction(Fraction(p_i ** -0.5), Fraction(p_min ** -0.5))
            assert batch.is_weights[k] == expected
            assert batch.float_is_weights[k] == pytest.approx(float(expected))

    def test_zero_weight_entries_are_never_sampled(self) -> None:
        buf = make_buffer(priority=AdvantagePriority(epsilon=0.0))
        buf.add_group("flat", 0, rollouts([1.0, 1.0]))      # q = 0, live, unsampleable
        buf.add_group("varied", 0, rollouts([1.0, 0.0]))
        assert buf.size == 4
        seen = set()
        for _ in range(20):
            seen.update(buf.sample(4).indices)
        assert seen <= {2, 3}


# ---------------------------------------------------------------------------
# update_priorities
# ---------------------------------------------------------------------------

class TestUpdatePriorities:
    def test_updates_leaf_and_keeps_version_by_default(self) -> None:
        buf = make_buffer()
        buf.add_group("p", 0, rollouts([1.0, 0.0]))
        buf.sample(1, current_version=5)
        buf.update_priorities([0, 1], [2.0, 0.25])
        q0 = quantize_priority(2.0, buf.params)
        assert buf.base_priority(0) == q0
        assert buf.entry_version(0) == 0
        assert buf.leaf(0) == inflated_priority(q0, 0, buf.base_epoch, buf.params)

    def test_reset_age_on_update_restamps_version(self) -> None:
        buf = make_buffer(reset_age_on_update=True)
        buf.add_group("p", 0, rollouts([1.0, 0.0]))
        buf.sample(1, current_version=5)
        buf.update_priorities([0], [2.0])
        assert buf.entry_version(0) == 5
        assert buf.entry_version(1) == 0

    def test_length_mismatch_rejected(self) -> None:
        buf = make_buffer()
        buf.add_group("p", 0, rollouts([1.0, 0.0]))
        with pytest.raises(ValueError, match="length"):
            buf.update_priorities([0, 1], [1.0])

    def test_non_live_index_rejected_and_nothing_changes(self) -> None:
        buf = make_buffer()
        buf.add_group("p", 0, rollouts([1.0, 0.0]))
        before = [buf.leaf(i) for i in range(buf.capacity)]
        with pytest.raises(ValueError, match="live"):
            buf.update_priorities([0, 5], [1.0, 1.0])
        assert [buf.leaf(i) for i in range(buf.capacity)] == before

    def test_bad_score_rejected_before_any_write(self) -> None:
        buf = make_buffer()
        buf.add_group("p", 0, rollouts([1.0, 0.0]))
        before = [buf.leaf(i) for i in range(buf.capacity)]
        with pytest.raises(ValueError):
            buf.update_priorities([0, 1], [1.0, math.nan])
        assert [buf.leaf(i) for i in range(buf.capacity)] == before

    @pytest.mark.parametrize("bad", [1.9, "0", None, True])
    def test_non_integer_indices_rejected(self, bad: object) -> None:
        buf = make_buffer()
        buf.add_group("p", 0, rollouts([1.0, 0.0]))
        with pytest.raises(ValueError, match="index"):
            buf.update_priorities([bad], [1.0])  # type: ignore[list-item]

    def test_accepts_numpy_arrays(self) -> None:
        np = pytest.importorskip("numpy")
        buf = make_buffer()
        buf.add_group("p", 0, rollouts([1.0, 0.0]))
        batch = buf.sample(2)
        buf.update_priorities(np.array(batch.indices), np.array([0.5, 0.25]))
        assert buf.base_priority(batch.indices[0]) == quantize_priority(0.5, buf.params)


# ---------------------------------------------------------------------------
# Staleness, rebase, and the attestation log
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_stale_entries_are_evicted_on_sample(self) -> None:
        log = AttestationLog()
        buf = make_buffer(max_policy_age=3, attest=log)
        buf.add_group("old", 0, rollouts([1.0, 0.0]))
        buf.add_group("new", 2, rollouts([1.0, 0.0]))
        buf.sample(1, current_version=4)  # "old" (age 4) expires
        assert buf.live_positions() == (2, 3)
        evicts = [r for r in log.records if r["op"] == "evict"]
        assert [r["index"] for r in evicts] == [0, 1]
        assert all(r["new_priority_int"] == "0" for r in evicts)

    def test_rebase_happens_and_sampling_continues(self) -> None:
        buf = make_buffer(half_life=1, max_policy_age=2)  # max_shift = 2
        for v in range(12):
            buf.add_group(f"g{v}", v, rollouts([1.0, 0.0]))
            buf.sample(2, current_version=v)
        assert buf.base_epoch > 0
        assert buf.n_rebases >= 1
        assert buf.verify_trees()
        for pos in buf.live_positions():
            q, t = buf.base_priority(pos), buf.entry_version(pos)
            assert buf.leaf(pos) == inflated_priority(q, t, buf.base_epoch, buf.params)

    def test_long_idle_jump_keeps_buffer_log_and_slots_consistent(self) -> None:
        log = AttestationLog()
        buf = make_buffer(capacity=8, half_life=4, max_policy_age=16, attest=log)
        buf.add_group("p", 0, rollouts([1.0, 0.0, 0.5]))
        buf.advance(1000)  # >= 64 epochs: everything expires, no leaf to shift
        assert buf.size == 0 and buf.total == 0
        assert buf.base_epoch > 0
        evicts = [r["index"] for r in log.records if r["op"] == "evict"]
        assert evicts == [0, 1, 2]
        idx = buf.add_group("q", 1000, rollouts([1.0, 0.0]))
        assert idx == (0, 1)  # released slots were reused
        batch = buf.sample(2, current_version=1001)
        assert set(batch.indices) <= {0, 1}
        verify_json_lines(log.to_json_lines(), buf.capacity)

    def test_checker_accepts_log_with_evictions_and_rebases(self) -> None:
        log = AttestationLog()
        buf = make_buffer(capacity=8, half_life=1, max_policy_age=2, seed=11, attest=log)
        for v in range(10):
            buf.add_group(f"g{v}", v, rollouts([1.0, 0.0, 0.5]))
            batch = buf.sample(4, current_version=v)
            buf.update_priorities(batch.indices[:2], [0.3, 0.6])
        assert buf.n_rebases >= 1
        ops = Counter(r["op"] for r in log.records)
        assert ops["insert"] > 0 and ops["evict"] > 0 and ops["update"] > 0 and ops["sample"] == 10
        verify_json_lines(log.to_json_lines(), buf.capacity)

    def test_checker_rejects_tampered_log(self) -> None:
        log = AttestationLog()
        buf = make_buffer(attest=log)
        buf.add_group("p", 0, rollouts([1.0, 0.0, 0.5]))
        buf.sample(2)
        records = [dict(r) for r in log.records]
        records[0]["new_priority_int"] = str(int(records[0]["new_priority_int"]) + 1)
        tampered = "\n".join(json.dumps(r, sort_keys=True, separators=(",", ":")) for r in records)
        with pytest.raises(CheckerError):
            verify_json_lines(tampered, buf.capacity)

    def test_readme_quick_start_runs(self) -> None:
        from reservoir.priorities import AdvantagePriority

        buf = RolloutBuffer(
            capacity=50_000,
            priority=AdvantagePriority(),
            half_life=4,
            max_policy_age=16,
            seed=0,
        )
        step = 0
        completions = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]
        behavior_logprobs = [[-0.1, -0.2, -0.3], [-0.5, -0.4], [-0.1] * 4]
        rewards = [1.0, 0.0, 1.0]
        buf.add_group(
            prompt_id="gsm8k-0412",
            model_version=step,
            rollouts=[
                Rollout(tokens=ids, logprobs=lp, reward=r)
                for ids, lp, r in zip(completions, behavior_logprobs, rewards)
            ],
        )
        batch = buf.sample(batch_size=3, current_version=step)
        assert len(batch.rollouts) == len(batch.logprobs) == len(batch.model_versions) == 3
        buf.update_priorities(batch.indices, [0.5, 0.5, 0.5])
