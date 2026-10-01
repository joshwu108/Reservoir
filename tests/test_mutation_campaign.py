"""
Test that the T4 mutation campaign achieves 100% rejection with >= 60 mutants.
"""

import pytest
from campaigns.mutation import run_mutation_campaign


class TestMutationCampaign:
    def test_100_percent_rejection(self):
        results = run_mutation_campaign()
        assert len(results["survived"]) == 0, (
            f"Surviving mutants (checker bugs): {results['survived']}"
        )

    def test_at_least_60_mutants(self):
        results = run_mutation_campaign()
        assert results["total_mutants"] >= 60, (
            f"Only {results['total_mutants']} mutants generated, need >= 60"
        )

    def test_decay_category_is_substantial(self):
        results = run_mutation_campaign()
        decay = next(d for d in results["details"] if d[0] == "decay")
        assert decay[1] >= 20, f"only {decay[1]} decay mutants"
        assert decay[1] == decay[2], "a decay mutant survived"

    def test_all_categories_present(self):
        results = run_mutation_campaign()
        category_names = {d[0] for d in results["details"]}
        required = {
            "digest_bitflip",
            "off_by_one_draw",
            "swapped_indices",
            "not_reduced_form",
            "deleted_records",
            "reordered_records",
            "stale_suffix_replay",
            "decay",
        }
        assert required.issubset(category_names), (
            f"Missing categories: {required - category_names}"
        )
