"""
Test that the T4 mutation campaign achieves 100% rejection with >= 60 mutants.
"""

import pytest
from campaigns.mutation import run_mutation_campaign


@pytest.fixture(scope="module")
def results() -> dict:
    """The campaign runs once per module; every test reads the same results."""
    return run_mutation_campaign()


class TestMutationCampaign:
    def test_100_percent_rejection(self, results):
        assert len(results["survived"]) == 0, (
            f"Surviving mutants (checker bugs): {results['survived']}"
        )

    def test_at_least_60_mutants(self, results):
        assert results["total_mutants"] >= 60, (
            f"Only {results['total_mutants']} mutants generated, need >= 60"
        )

    def test_decay_category_is_substantial(self, results):
        decay = next(d for d in results["details"] if d[0] == "decay")
        assert decay[1] >= 20, f"only {decay[1]} decay mutants"
        assert decay[1] == decay[2], "a decay mutant survived"

    def test_all_categories_present(self, results):
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
            "content",
            "content_manifest",
            "draw",
            "batch",
            "telemetry",
            "provenance",
            "replay_manifest",
            "staleness",
        }
        assert required.issubset(category_names), (
            f"Missing categories: {required - category_names}"
        )


class TestContentCampaign:
    """Content-commitment forgeries: what the checker catches with and without the manifest."""

    def test_content_categories_present_and_fully_rejected(self, results):
        by_name = {d[0]: d for d in results["details"]}
        assert "content" in by_name and "content_manifest" in by_name
        assert by_name["content"][1] >= 8 and by_name["content"][1] == by_name["content"][2]
        assert by_name["content_manifest"][1] >= 8 and by_name["content_manifest"][1] == by_name["content_manifest"][2]

    def test_limit_without_manifest_is_measured_and_stated(self, results):
        # A chain-consistent swap of a content digest or source cannot be
        # detected from the log alone: the log commits, the manifest opens.
        limit = results["content_limit"]
        assert limit["undetectable_without_manifest"] == 3
        assert set(limit["mutants"]) == {"insert_digest_swapped_chain_consistent",
                                         "insert_source_changed_chain_consistent",
                                         "insert_source_dropped_chain_consistent"}
        assert limit["detected_with_manifest"] == limit["undetectable_without_manifest"]
        assert limit["undetectable_without_manifest"] not in (None, 0)
        # Those mutants are not counted as survivors: survival there is the documented limit.
        assert results["survived"] == []


class TestDrawCampaign:
    def test_draw_category_fully_rejected(self, results):
        draw = next(d for d in results["details"] if d[0] == "draw")
        assert draw[1] >= 15 and draw[1] == draw[2]

    def test_draw_limit_is_measured(self, results):
        limit = results["draw_limit"]
        assert set(limit["rejected_with_draw_config"]) == set(limit["mutants"]) == {"draw_moved_within_leaf", "is_weight_replaced"}
        # Without the configuration, the reweighted sample always survives; the moved
        # draw survives unless it happened to cross a leaf boundary.
        assert "is_weight_replaced" in limit["survive_without_draw_config"]


class TestBatchCampaign:
    def test_batch_category_fully_rejected(self, results):
        batch = next(d for d in results["details"] if d[0] == "batch")
        assert batch[1] >= 12 and batch[1] == batch[2]


class TestTelemetryCampaign:
    def test_telemetry_category_fully_rejected(self, results):
        telemetry = next(d for d in results["details"] if d[0] == "telemetry")
        assert telemetry[1] >= 12 and telemetry[1] == telemetry[2]


class TestProvenanceCampaign:
    def test_provenance_category_fully_rejected(self, results):
        prov = next(d for d in results["details"] if d[0] == "provenance")
        assert prov[1] >= 15 and prov[1] == prov[2]

    def test_limit_is_measured_and_stated(self, results):
        # The log does not commit to the predicate text or to the manifest's
        # per-reward-function values: chain-consistent changes to them survive.
        limit = results["provenance_limit"]
        assert set(limit["survive"]) == {"quarantine_predicate_text_changed_chain_consistent",
                                         "quarantine_note_changed_chain_consistent",
                                         "manifest_rewards_value_changed",
                                         "manifest_rewards_dropped"}
        assert limit["statement"]
        assert results["survived"] == []


class TestReplayCampaign:
    def test_replay_manifest_category_fully_rejected(self, results):
        cat = next(d for d in results["details"] if d[0] == "replay_manifest")
        assert cat == ("replay_manifest", 16, 16)

    def test_reward_provenance_changes_are_measured_not_counted(self, results):
        # The log does not commit to the per-reward-function values: a tampered value is not
        # refused, the replay reports it, and nothing else in the output moves. That is a
        # measurement beside the category, not a rejection.
        split = results["replay_manifest"]
        assert set(split["changed"]) == {"manifest_rewards_changed_on_replayed_row",
                                         "manifest_rewards_dropped_on_replayed_row",
                                         "manifest_rewards_changed_on_generated_example"}
        assert len(split["broke"]) == 16 and not set(split["broke"]) & set(split["changed"])
        assert split["statement"]
        assert results["survived"] == []


class TestStalenessCampaign:
    def test_staleness_category_fully_rejected(self, results):
        cat = next(d for d in results["details"] if d[0] == "staleness")
        assert cat[1] >= 30 and cat[1] == cat[2]

    def test_consistent_parameter_changes_are_measured_not_counted(self, results):
        # The checker verifies that the decisions follow the declared policy; a parameter changed on every
        # record so that no decision differs passes, and the campaign states it.
        limit = results["staleness_limit"]
        assert set(limit["pass"]) == set(limit["mutants"]) == {"staleness_policy_decline_cap_added_never_binding",
                                                               "staleness_policy_ess_floor_loosened_nothing_declined_for_ess"}
        assert limit["statement"]
        assert results["survived"] == []
