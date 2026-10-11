"""Field whitelists and degenerate records.

Every record type the library writes has a fixed set of fields; a field
the checker does not read is provenance it never validated, so a record
carrying one is rejected. An empty ``sample`` and a ``batch`` of zero rows
are refused with their own reasons rather than failing later on a
confusing one.
"""

from __future__ import annotations

import copy
from fractions import Fraction

import pytest

from reservoir.attest import AttestationLog, _digest_record
from reservoir.rollout import Rollout
from reservoir.rollout_buffer import RolloutBuffer
from reservoir_checker.verify import CheckerError, verify_chain


def rechain(records, start=0):
    out = copy.deepcopy(records)
    prev = out[start - 1]["digest"] if start > 0 else "genesis"
    for rec in out[start:]:
        rec["prev_digest"] = prev
        rec["digest"] = _digest_record(rec)
        prev = rec["digest"]
    return out


def full_log():
    """Every op the library writes except telemetry (which has its own whitelist): config, inserts,
    updates, stale and capacity evicts, advances, a rebase, samples and witnesses."""
    buf = RolloutBuffer(capacity=8, half_life=1, max_policy_age=2, seed=4, attest=AttestationLog())
    for v in range(8):
        buf.add_group(f"g{v}", v, [Rollout(tokens=[v + 1, 2], logprobs=[-0.1, -0.2], reward=r)
                                   for r in (1.0, 0.0, 0.5)], source="s")
        batch = buf.sample(2, current_version=v)
        buf.witness_batch(batch, step=v, batch_rows=8, rows=[6, 7], tensor_digest="ab" * 32)
        buf.update_priorities(batch.indices[:1], [0.4])
    records = [dict(r) for r in buf.attestation_log.records]
    verify_chain(records)
    return records


def first(records, op, **match):
    return next(i for i, r in enumerate(records) if r["op"] == op and all(r.get(k) == v for k, v in match.items()))


def edited(records, idx, edit):
    m = copy.deepcopy(records)
    edit(m[idx])
    return rechain(m, idx)


@pytest.fixture(scope="module")
def base():
    return full_log()


class TestUnknownFields:
    @pytest.mark.parametrize("op, match", [
        ("decay_config", {}), ("insert", {}), ("update", {}), ("evict", {"reason": "stale"}),
        ("evict", {"reason": "capacity"}), ("advance_version", {}), ("rebase", {}), ("sample", {}), ("batch", {}),
    ])
    def test_extra_field_rejected(self, base, op, match):
        i = first(base, op, **match)
        with pytest.raises(CheckerError, match=r"unknown field.*'foo'"):
            verify_chain(edited(base, i, lambda r: r.update(foo="1")))

    @pytest.mark.parametrize("name", ["source", "content_digest"])
    def test_provenance_fields_on_a_sample_rejected(self, base, name):
        i = first(base, "sample")
        with pytest.raises(CheckerError, match="unknown field"):
            verify_chain(edited(base, i, lambda r: r.update({name: "ab" * 32})))

    def test_extra_field_in_a_draw_rejected(self, base):
        i = first(base, "sample")
        with pytest.raises(CheckerError, match=r"sample 0.*unknown field.*'source'"):
            verify_chain(edited(base, i, lambda r: r["samples"][0].update(source="s")))

    def test_extra_field_in_a_witness_row_rejected(self, base):
        i = first(base, "batch")
        with pytest.raises(CheckerError, match=r"replaced 0.*unknown field.*'note'"):
            verify_chain(edited(base, i, lambda r: r["replaced"][0].update(note="x")))

    def test_honest_logs_still_verify(self, base):
        verify_chain(base)


class TestDegenerateRecords:
    def test_empty_sample_rejected(self, base):
        i = first(base, "sample")
        with pytest.raises(CheckerError, match="sample record draws nothing"):
            verify_chain(edited(base, i, lambda r: r.update(samples=[])))

    def test_batch_rows_zero_rejected(self, base):
        i = first(base, "batch")
        with pytest.raises(CheckerError, match="batch_rows must be at least 1"):
            verify_chain(edited(base, i, lambda r: r.update(batch_rows=0)))


class TestLegacyAndUpcomingFields:
    def test_legacy_log_without_decay_config_verifies(self):
        from reservoir.attest import make_sample_entry

        log = AttestationLog()
        log.append_mutation("insert", 0, 0, 4, 1)
        log.append_mutation("insert", 1, 0, 4, 2)
        log.append_sample(2, 8, [make_sample_entry(0, 3, 4, 8, Fraction(1))])
        verify_chain([dict(r) for r in log.records], capacity=2)

    def test_sample_min_entry_age_and_draw_entry_version_are_known_fields(self):
        from campaigns.mutation_entry_age import build_entry_age_chain

        verify_chain(build_entry_age_chain())
