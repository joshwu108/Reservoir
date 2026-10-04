"""Tests for reservoir.rollout — Rollout and RolloutGroup value types.

These are the entries of the LLM-RL replay buffer: whole rollouts with
behavior logprobs and the model version that produced them. The tests
cover three things: inputs are validated at construction with an error
that names the field, instances cannot be mutated afterwards (directly or
through the caller's original lists), and the cached group statistics are
correct and stay finite for every accepted input.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from types import MappingProxyType

import numpy as np
import pytest
from hypothesis import given, strategies as st

from reservoir.rollout import MAX_ABS_REWARD, MAX_SOURCE_LENGTH, Rollout, RolloutGroup, content_digest_of


def make_rollout(reward: float = 1.0, n: int = 3, **kw) -> Rollout:
    """A valid Rollout with ``n`` tokens and the given reward; ``kw`` is passed through."""
    return Rollout(
        tokens=list(range(n)),
        logprobs=[-0.5] * n,
        reward=reward,
        **kw,
    )


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

class TestRolloutConstruction:
    def test_sequences_are_stored_as_tuples(self) -> None:
        r = Rollout(tokens=[1, 2, 3], logprobs=[-0.1, -0.2, -0.3], reward=1.0)
        assert r.tokens == (1, 2, 3)
        assert isinstance(r.tokens, tuple)
        assert r.logprobs == (-0.1, -0.2, -0.3)
        assert isinstance(r.logprobs, tuple)
        assert r.reward == 1.0

    def test_len_is_token_count(self) -> None:
        assert len(make_rollout(n=7)) == 7

    def test_caller_mutating_input_lists_does_not_affect_rollout(self) -> None:
        tokens = [1, 2]
        logprobs = [-1.0, -2.0]
        r = Rollout(tokens=tokens, logprobs=logprobs, reward=0.0)
        tokens.append(3)
        logprobs[0] = 99.0
        assert r.tokens == (1, 2)
        assert r.logprobs == (-1.0, -2.0)

    def test_is_frozen(self) -> None:
        r = make_rollout()
        with pytest.raises(dataclasses.FrozenInstanceError):
            r.reward = 2.0  # type: ignore[misc]

    def test_is_hashable_and_equal_by_value(self) -> None:
        a = make_rollout()
        b = make_rollout()
        assert a == b
        assert hash(a) == hash(b)
        assert len({a, b}) == 1

    def test_integer_reward_is_accepted(self) -> None:
        r = Rollout(tokens=[1], logprobs=[0.0], reward=1)
        assert r.reward == 1.0
        assert isinstance(r.reward, float)

    def test_metadata_defaults_to_empty_immutable_mapping(self) -> None:
        r = make_rollout()
        assert len(r.metadata) == 0
        with pytest.raises(TypeError):
            r.metadata["x"] = 1  # type: ignore[index]

    def test_metadata_is_copied_into_immutable_mapping(self) -> None:
        meta = {"source": "gsm8k", "idx": 3}
        r = make_rollout(metadata=meta)
        meta["source"] = "changed"
        assert r.metadata["source"] == "gsm8k"
        assert isinstance(r.metadata, MappingProxyType)
        with pytest.raises(TypeError):
            r.metadata["idx"] = 4  # type: ignore[index]


class TestRolloutValidation:
    def test_empty_tokens_rejected(self) -> None:
        with pytest.raises(ValueError, match="tokens"):
            Rollout(tokens=[], logprobs=[], reward=0.0)

    @pytest.mark.parametrize("bad", [[1.5], ["a"], [True], [None]])
    def test_non_int_token_rejected(self, bad: list) -> None:
        with pytest.raises(ValueError, match="tokens"):
            Rollout(tokens=bad, logprobs=[0.0], reward=0.0)

    def test_negative_token_rejected(self) -> None:
        with pytest.raises(ValueError, match="tokens"):
            Rollout(tokens=[-1], logprobs=[0.0], reward=0.0)

    def test_logprob_length_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="logprobs"):
            Rollout(tokens=[1, 2], logprobs=[0.0], reward=0.0)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_non_finite_logprob_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError, match="logprobs"):
            Rollout(tokens=[1], logprobs=[bad], reward=0.0)

    @pytest.mark.parametrize("bad", ["x", None, True])
    def test_non_numeric_logprob_rejected(self, bad: object) -> None:
        with pytest.raises(ValueError, match="logprobs"):
            Rollout(tokens=[1], logprobs=[bad], reward=0.0)  # type: ignore[list-item]

    @pytest.mark.parametrize("bad", [math.nan, math.inf, "1", None, True])
    def test_bad_reward_rejected(self, bad: object) -> None:
        with pytest.raises(ValueError, match="reward"):
            Rollout(tokens=[1], logprobs=[0.0], reward=bad)  # type: ignore[arg-type]

    def test_string_tokens_rejected_as_a_whole(self) -> None:
        # A str is iterable; it must be refused, not split into characters.
        with pytest.raises(ValueError, match="tokens"):
            Rollout(tokens="123", logprobs=[0.0, 0.0, 0.0], reward=0.0)  # type: ignore[arg-type]

    def test_huge_int_reward_rejected_with_value_error(self) -> None:
        with pytest.raises(ValueError, match="reward"):
            Rollout(tokens=[1], logprobs=[0.0], reward=10**400)

    def test_reward_above_magnitude_limit_rejected(self) -> None:
        with pytest.raises(ValueError, match="reward"):
            Rollout(tokens=[1], logprobs=[0.0], reward=1e200)
        Rollout(tokens=[1], logprobs=[0.0], reward=MAX_ABS_REWARD)  # exactly at the limit is accepted

    def test_positive_logprob_rejected(self) -> None:
        with pytest.raises(ValueError, match="logprobs"):
            Rollout(tokens=[1], logprobs=[0.5], reward=0.0)
        Rollout(tokens=[1], logprobs=[0.0], reward=0.0)  # log(1) == 0.0 is valid

    def test_numpy_scalars_accepted_and_normalized(self) -> None:
        np = pytest.importorskip("numpy")
        r = Rollout(
            tokens=np.array([1, 2], dtype=np.int64),
            logprobs=np.array([-0.1, -0.2], dtype=np.float32),
            reward=np.float32(1.0),
        )
        assert r.tokens == (1, 2)
        assert all(type(t) is int for t in r.tokens)
        assert all(type(lp) is float for lp in r.logprobs)
        assert type(r.reward) is float

    @pytest.mark.parametrize("bad", [{1: 0}, {1, 2}, 5])
    def test_non_sequence_tokens_rejected(self, bad: object) -> None:
        with pytest.raises(ValueError, match="tokens"):
            Rollout(tokens=bad, logprobs=[0.0], reward=0.0)  # type: ignore[arg-type]

    def test_2d_array_tokens_rejected(self) -> None:
        np = pytest.importorskip("numpy")
        with pytest.raises(ValueError, match="tokens"):
            Rollout(tokens=np.zeros((2, 2), dtype=np.int64), logprobs=[0.0, 0.0], reward=0.0)

    def test_non_mapping_metadata_rejected(self) -> None:
        with pytest.raises(ValueError, match="metadata"):
            make_rollout(metadata=[("a", 1)])  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# RolloutGroup
# ---------------------------------------------------------------------------

class TestRolloutGroupConstruction:
    def test_basic_fields(self) -> None:
        rs = [make_rollout(1.0), make_rollout(0.0)]
        g = RolloutGroup(prompt_id="p1", model_version=7, rollouts=rs)
        assert g.prompt_id == "p1"
        assert g.model_version == 7
        assert g.rollouts == tuple(rs)
        assert isinstance(g.rollouts, tuple)
        assert g.size == 2
        assert len(g) == 2

    def test_is_frozen(self) -> None:
        g = RolloutGroup(prompt_id="p", model_version=0, rollouts=[make_rollout()])
        with pytest.raises(dataclasses.FrozenInstanceError):
            g.model_version = 1  # type: ignore[misc]

    def test_equal_by_value(self) -> None:
        a = RolloutGroup(prompt_id="p", model_version=0, rollouts=[make_rollout()])
        b = RolloutGroup(prompt_id="p", model_version=0, rollouts=[make_rollout()])
        assert a == b
        assert hash(a) == hash(b)


class TestRolloutGroupValidation:
    @pytest.mark.parametrize("bad", ["", None, 3])
    def test_bad_prompt_id_rejected(self, bad: object) -> None:
        with pytest.raises(ValueError, match="prompt_id"):
            RolloutGroup(prompt_id=bad, model_version=0, rollouts=[make_rollout()])  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad", [-1, 1.0, True, "0", None])
    def test_bad_model_version_rejected(self, bad: object) -> None:
        with pytest.raises(ValueError, match="model_version"):
            RolloutGroup(prompt_id="p", model_version=bad, rollouts=[make_rollout()])  # type: ignore[arg-type]

    def test_empty_rollouts_rejected(self) -> None:
        with pytest.raises(ValueError, match="rollouts"):
            RolloutGroup(prompt_id="p", model_version=0, rollouts=[])

    def test_non_rollout_element_rejected(self) -> None:
        with pytest.raises(ValueError, match="rollouts"):
            RolloutGroup(prompt_id="p", model_version=0, rollouts=[make_rollout(), "x"])  # type: ignore[list-item]

    def test_non_callable_is_success_rejected(self) -> None:
        with pytest.raises(ValueError, match="is_success"):
            RolloutGroup(
                prompt_id="p", model_version=0, rollouts=[make_rollout()],
                is_success="yes",  # type: ignore[arg-type]
            )


class TestRolloutGroupStatistics:
    def test_mean_reward(self) -> None:
        g = RolloutGroup(
            prompt_id="p", model_version=0,
            rollouts=[make_rollout(1.0), make_rollout(0.0), make_rollout(0.5)],
        )
        assert g.mean_reward == pytest.approx(0.5)

    def test_reward_std_is_population_std(self) -> None:
        g = RolloutGroup(
            prompt_id="p", model_version=0,
            rollouts=[make_rollout(1.0), make_rollout(0.0)],
        )
        assert g.reward_std == pytest.approx(0.5)

    def test_reward_std_of_single_rollout_is_zero(self) -> None:
        g = RolloutGroup(prompt_id="p", model_version=0, rollouts=[make_rollout(3.0)])
        assert g.reward_std == 0.0

    def test_advantages_are_group_relative_without_std_normalization(self) -> None:
        g = RolloutGroup(
            prompt_id="p", model_version=0,
            rollouts=[make_rollout(1.0), make_rollout(0.0), make_rollout(0.5)],
        )
        assert g.advantages == pytest.approx((0.5, -0.5, 0.0))
        assert isinstance(g.advantages, tuple)

    def test_pass_rate_default_predicate_is_reward_positive(self) -> None:
        g = RolloutGroup(
            prompt_id="p", model_version=0,
            rollouts=[make_rollout(1.0), make_rollout(0.0), make_rollout(-1.0), make_rollout(0.3)],
        )
        assert g.pass_rate == pytest.approx(0.5)
        assert g.n_success == 2

    def test_pass_rate_custom_predicate(self) -> None:
        g = RolloutGroup(
            prompt_id="p", model_version=0,
            rollouts=[make_rollout(0.9), make_rollout(0.4)],
            is_success=lambda r: r.reward >= 0.5,
        )
        assert g.pass_rate == pytest.approx(0.5)

    def test_is_success_does_not_affect_equality(self) -> None:
        rs = [make_rollout(1.0)]
        a = RolloutGroup(prompt_id="p", model_version=0, rollouts=rs)
        b = RolloutGroup(prompt_id="p", model_version=0, rollouts=rs, is_success=lambda r: False)
        assert a == b

    def test_all_pass_and_all_fail(self) -> None:
        all_pass = RolloutGroup(prompt_id="p", model_version=0, rollouts=[make_rollout(1.0)] * 4)
        all_fail = RolloutGroup(prompt_id="p", model_version=0, rollouts=[make_rollout(0.0)] * 4)
        assert all_pass.pass_rate == 1.0
        assert all_fail.pass_rate == 0.0

    def test_statistics_are_finite_at_reward_magnitude_limit(self) -> None:
        g = RolloutGroup(
            prompt_id="p", model_version=0,
            rollouts=[make_rollout(MAX_ABS_REWARD), make_rollout(-MAX_ABS_REWARD)],
        )
        assert g.mean_reward == 0.0
        assert math.isfinite(g.reward_std) and g.reward_std == MAX_ABS_REWARD
        assert all(math.isfinite(a) for a in g.advantages)

    @given(
        rewards=st.lists(
            st.floats(
                min_value=-MAX_ABS_REWARD, max_value=MAX_ABS_REWARD,
                allow_nan=False, allow_infinity=False,
            ),
            min_size=1, max_size=32,
        )
    )
    def test_statistics_are_consistent(self, rewards: list[float]) -> None:
        g = RolloutGroup(
            prompt_id="p", model_version=0, rollouts=[make_rollout(r) for r in rewards],
        )
        assert g.size == len(rewards)
        assert math.isfinite(g.mean_reward)
        assert math.isfinite(g.reward_std) and g.reward_std >= 0.0
        assert all(math.isfinite(a) for a in g.advantages)
        assert 0.0 <= g.pass_rate <= 1.0
        assert g.pass_rate == pytest.approx(g.n_success / g.size)
        assert len(g.advantages) == len(rewards)
        scale = max(1.0, max(abs(r) for r in rewards))
        assert math.fsum(g.advantages) / scale == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Content identity (attested on every insert record)
# ---------------------------------------------------------------------------

def _reference_digest(prompt_id: str, tokens, reward: float) -> str:
    """The digest definition from docs/design.md §10, written out independently."""
    canonical = json.dumps(
        {"prompt_id": prompt_id, "reward": float(reward).hex(), "tokens": list(tokens)},
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.blake2b(canonical, digest_size=32, person=b"rollout-content\x00").hexdigest()


class TestContentDigest:
    def test_is_64_lowercase_hex(self) -> None:
        d = content_digest_of("p", (1, 2), 1.0)
        assert len(d) == 64
        assert d == d.lower()
        int(d, 16)

    def test_matches_reference_definition(self) -> None:
        assert content_digest_of("p", (1, 2, 3), 0.5) == _reference_digest("p", (1, 2, 3), 0.5)

    @given(
        prompt_id=st.text(min_size=1, max_size=20),
        tokens=st.lists(st.integers(min_value=0, max_value=200_000), min_size=1, max_size=20),
        reward=st.floats(allow_nan=False, allow_infinity=False, width=64, min_value=-1e6, max_value=1e6),
    )
    def test_matches_reference_on_arbitrary_inputs(self, prompt_id, tokens, reward) -> None:
        assert content_digest_of(prompt_id, tokens, reward) == _reference_digest(prompt_id, tokens, reward)

    def test_ignores_logprobs_and_metadata(self) -> None:
        a = Rollout(tokens=[1, 2], logprobs=[-0.1, -0.2], reward=1.0, metadata={"row": 1})
        b = Rollout(tokens=[1, 2], logprobs=[-0.9, -0.8], reward=1.0, metadata={"row": 2})
        g = RolloutGroup(prompt_id="p", model_version=0, rollouts=[a, b])
        assert g.content_digests[0] == g.content_digests[1]

    def test_sensitive_to_tokens_reward_and_prompt(self) -> None:
        base = content_digest_of("p", (1, 2), 1.0)
        assert content_digest_of("p", (1, 3), 1.0) != base
        assert content_digest_of("p", (1, 2), 1.0000001) != base
        assert content_digest_of("q", (1, 2), 1.0) != base

    def test_negative_zero_differs_from_zero(self) -> None:
        # float.hex() distinguishes the two; documented in design.md §10.
        assert content_digest_of("p", (1,), -0.0) != content_digest_of("p", (1,), 0.0)

    # Known-answer vectors, computed once and pinned. They catch a change to
    # the canonical form (key order, escaping, personalisation) that a test
    # sharing code with the implementation would miss, and they are what
    # the checker's independent implementation is tested against too.
    GOLDEN = {
        ("p", (1, 2, 3), 0.5): "2a837616faed26935aa920b0fd1357cdd528667c469e6682e554e3b85079578f",
        ("\u00e9\U0001F600", (0,), -1.0): "724b08aaababdc38a1730a9e9167652c6c66ee419e90193be6e509b812ebd15c",
    }

    def test_known_answer_vectors(self) -> None:
        for (prompt_id, tokens, reward), expected in self.GOLDEN.items():
            assert content_digest_of(prompt_id, tokens, reward) == expected, (prompt_id, tokens, reward)

    def test_non_ascii_prompt_is_escaped_not_raw(self) -> None:
        # ensure_ascii=True is part of the definition: a checker that wrote
        # raw UTF-8 would disagree on every non-ASCII prompt id.
        raw = json.dumps({"prompt_id": "\u00e9", "reward": (1.0).hex(), "tokens": [1]},
                         sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        raw_digest = hashlib.blake2b(raw, digest_size=32, person=b"rollout-content\x00").hexdigest()
        assert content_digest_of("\u00e9", (1,), 1.0) != raw_digest

    def test_numpy_and_int_inputs_normalise(self) -> None:
        assert content_digest_of("p", np.array([1, 2], dtype=np.int64), 1) == content_digest_of("p", (1, 2), 1.0)

    @pytest.mark.parametrize("args", [
        ("", (1,), 1.0), (7, (1,), 1.0), ("p", (), 1.0), ("p", (True,), 1.0), ("p", (-1,), 1.0),
        ("p", (1,), float("nan")), ("p", (1,), float("inf")), ("p", (1,), 10 ** 400),
    ])
    def test_malformed_inputs_rejected(self, args) -> None:
        with pytest.raises(ValueError):
            content_digest_of(*args)

    def test_group_digests_align_with_rollouts(self) -> None:
        g = RolloutGroup(
            prompt_id="p", model_version=0,
            rollouts=[make_rollout(reward=0.0, n=2), make_rollout(reward=1.0, n=3)],
        )
        assert len(g.content_digests) == 2
        assert g.content_digests[1] == content_digest_of("p", (0, 1, 2), 1.0)


class TestRolloutGroupSource:
    def test_default_is_none(self) -> None:
        g = RolloutGroup(prompt_id="p", model_version=0, rollouts=[make_rollout()])
        assert g.source is None

    def test_source_is_kept(self) -> None:
        g = RolloutGroup(prompt_id="p", model_version=0, rollouts=[make_rollout()], source="gsm8k")
        assert g.source == "gsm8k"

    def test_source_participates_in_equality(self) -> None:
        a = RolloutGroup(prompt_id="p", model_version=0, rollouts=[make_rollout()], source="a")
        b = RolloutGroup(prompt_id="p", model_version=0, rollouts=[make_rollout()], source="b")
        assert a != b

    @pytest.mark.parametrize("bad", ["", "x" * 257, "a\nb", "a\tb", "a\x00b", "a\u2028b", "a\u200bb",
                                     "   ", 7, b"bytes"])
    def test_bad_source_rejected(self, bad: object) -> None:
        with pytest.raises(ValueError, match="source"):
            RolloutGroup(prompt_id="p", model_version=0, rollouts=[make_rollout()], source=bad)

    def test_maximum_length_accepted(self) -> None:
        g = RolloutGroup(prompt_id="p", model_version=0, rollouts=[make_rollout()], source="x" * MAX_SOURCE_LENGTH)
        assert len(g.source) == MAX_SOURCE_LENGTH

    def test_repr_shows_source_only_when_set(self) -> None:
        without = RolloutGroup(prompt_id="p", model_version=0, rollouts=[make_rollout()])
        with_source = RolloutGroup(prompt_id="p", model_version=0, rollouts=[make_rollout()], source="s")
        assert "source" not in repr(without)
        assert "source='s'" in repr(with_source)

    def test_unicode_source_accepted(self) -> None:
        g = RolloutGroup(prompt_id="p", model_version=0, rollouts=[make_rollout()], source="données/été")
        assert g.source == "données/été"
