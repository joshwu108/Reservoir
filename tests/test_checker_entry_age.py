"""The declared minimum entry age of a sample record.

A sample that declares ``min_entry_age`` must draw only slots at least that
many versions old at the buffer's version, and every draw's
``entry_version`` must be the version the replay holds for its slot. Logs
written before the fields existed carry neither and still verify. Until the
library writes the fields, the honest logs here are library logs annotated
with exactly what the replay says (``campaigns.mutation_entry_age``).
"""

from __future__ import annotations

import copy

import pytest

from campaigns.mutation_entry_age import (
    _rechain,
    _run,
    build_entry_age_chain,
    build_same_step_chain,
    entry_age_mutants,
    with_entry_age,
)
from reservoir_checker.verify import CheckerError, verify_chain


def samples_of(records):
    return [i for i, r in enumerate(records) if r["op"] == "sample"]


def edited(records, idx, edit):
    m = copy.deepcopy(records)
    edit(m[idx])
    return _rechain(m, idx)


class TestAccepts:
    def test_old_enough_draws_verify(self):
        verify_chain(build_entry_age_chain())

    def test_logs_without_the_fields_still_verify(self):
        verify_chain(_run(sample_ahead=False))
        verify_chain(_run(sample_ahead=True))

    def test_minimum_zero_admits_same_step_draws(self):
        verify_chain(with_entry_age(_run(sample_ahead=False), 0))

    def test_entry_version_without_a_minimum_is_still_checked(self):
        records = build_entry_age_chain()
        stripped = copy.deepcopy(records)
        for i in samples_of(stripped):
            stripped[i].pop("min_entry_age")
        verify_chain(_rechain(stripped, 0))
        j = samples_of(stripped)[-1]
        k = next(k for k, d in enumerate(stripped[j]["samples"]) if d["entry_version"] != "0")
        with pytest.raises(CheckerError, match="replay holds version"):
            verify_chain(edited(_rechain(stripped, 0), j, lambda r: r["samples"][k].update(entry_version="0")))


class TestRejects:
    def test_same_step_draw_below_the_declared_minimum(self):
        with pytest.raises(CheckerError, match="below the declared min_entry_age 1"):
            verify_chain(build_same_step_chain())

    def test_forged_entry_version(self):
        records = build_entry_age_chain()
        j = samples_of(records)[-1]
        k = next(k for k, d in enumerate(records[j]["samples"]) if d["entry_version"] != "0")
        with pytest.raises(CheckerError, match="replay holds version"):
            verify_chain(edited(records, j, lambda r: r["samples"][k].update(entry_version="0")))

    def test_draw_without_its_version(self):
        records = build_entry_age_chain()
        i = samples_of(records)[0]
        with pytest.raises(CheckerError, match="needs entry_version on every draw"):
            verify_chain(edited(records, i, lambda r: r["samples"][0].pop("entry_version")))

    def test_minimum_dropped_from_a_later_sample(self):
        records = build_entry_age_chain()
        j = samples_of(records)[-1]
        with pytest.raises(CheckerError, match="cannot leave it out"):
            verify_chain(edited(records, j, lambda r: r.pop("min_entry_age")))

    def test_minimum_changed_between_samples(self):
        records = build_entry_age_chain()
        j = samples_of(records)[-1]
        with pytest.raises(CheckerError, match="fixed for a run"):
            verify_chain(edited(records, j, lambda r: r.update(min_entry_age="0")))

    def test_minimum_first_declared_after_the_first_sample(self):
        records = build_entry_age_chain()
        i = samples_of(records)[0]
        with pytest.raises(CheckerError, match="from its first sample"):
            verify_chain(edited(records, i, lambda r: r.pop("min_entry_age")))

    @pytest.mark.parametrize("bad", ["-1", "1.5", True, None, " 1", "01", 1])
    def test_minimum_malformed(self, bad):
        records = build_entry_age_chain()
        i = samples_of(records)[0]
        with pytest.raises(CheckerError, match="min_entry_age"):
            verify_chain(edited(records, i, lambda r: r.update(min_entry_age=bad)))

    def test_every_campaign_mutant_is_rejected(self):
        for records, name in entry_age_mutants(build_entry_age_chain(), build_same_step_chain()):
            with pytest.raises(CheckerError):
                verify_chain(records, 16)
