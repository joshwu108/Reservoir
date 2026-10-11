"""
campaigns.mutation_reasons — Why each mutant must be rejected.

A mutant counts as rejected only when the checker's error names the rule
the mutant was built to break: ``EXPECTED`` maps a mutant's description to
a substring of that error, or a tuple of substrings that must all appear (``PREFIXES`` covers the families whose
descriptions carry an index). A mutant rejected for another reason is
listed under ``wrong_reason`` and fails the campaign just as a survivor
does, so a mutant that dies on an unrelated check cannot pass as coverage
of the rule its name claims. Every mutant must have an entry.
"""

from __future__ import annotations

from typing import Callable, Optional, Union

from reservoir_checker.decay_replay import CheckerError

TREE = "but tree has"                         # a mutation's old leaf disagrees with the replayed tree
TOTAL = "but replayed tree total"             # a sample's root_total disagrees with the replayed tree
PENDING = "have not been evicted"             # a record between an advance and the stale evicts it requires
LEAF = "but recomputed leaf is"
LOCATE = "maps to position"
REDUCED = "is not in reduced form"
KEYED = "is not the keyed draw"
FORMULA = "is not the declared formula's value"
RECOMPUTE = "does not recompute from its prompt, tokens and reward"
LINE_COUNT = "insert records with content digests"
SLOT = "(op_counter, index)"
SOURCE = ": source "
UNKNOWN = "unknown field"

Reason = Union[str, tuple[str, ...]]

PREFIXES: tuple[tuple[str, Reason], ...] = (
    ("digest_bitflip_record_", "digest mismatch"),
    ("digest_lastbyte_flip_record_", "digest mismatch"),
    ("boundary_draw_wrong_leaf_", LOCATE),
    ("before_boundary_wrong_leaf_", LOCATE),
    ("swapped_indices_", LOCATE),
    ("wrong_leaf_index_", LOCATE),
    ("prob_not_reduced_", REDUCED),
    ("is_weight_not_reduced_", REDUCED),
    ("deleted_mutation_record_", TOTAL),
    ("swap_update_", ("(update at pos=", TREE)),
    ("adjacent_swap_", ("(update at pos=", TREE)),
    ("sample_moved_before_inserts_", TOTAL),
    ("stale_suffix_replay_from_", ("(insert at pos=", "old_priority_int=0", TREE)),
    ("insert_leaf_plus_one_", LEAF),
    ("insert_q_plus_one_", LEAF),
    ("insert_index_", "index must be an int"),
    ("entry_version_encoded_as_", "entry_version must be a non-negative integer encoded as a decimal string"),
    ("insert_digest_malformed_", "content_digest must be 64 lowercase hex characters"),
    ("insert_source_malformed_", "source must be a printable string"),
    ("draw_moved_record", KEYED),
    ("is_weight_replaced_record", FORMULA),
    ("extra_field_", UNKNOWN),
)

EXPECTED: dict[str, Reason] = {
    # legacy chain
    "deleted_mutation_record_0": "prev_digest mismatch",   # not re-chained: the deletion breaks the link
    "deleted_first_record": TOTAL,
    "duplicate_last_record": ("(update at pos=", TREE),
    # decay protocol
    "decay_config_not_first": "requires a decay_config record",
    "decay_config_removed": "requires a decay_config record",
    "decay_config_half_life_changed": LEAF,
    "update_leaf_minus_one": LEAF,
    "update_q_plus_seven": LEAF,
    "insert_wrong_base_epoch": "but replay base_epoch is",
    "insert_version_newer_than_current": "is newer than current version",
    "update_version_changed_without_reset": "update must keep entry_version",
    "advance_version_backwards": "goes backwards",
    "advance_version_old_mismatch": "advance_version old_version=",
    "expired_entry_never_evicted": "rebase before expired entries",   # the next record after the gap is the rebase
    "stale_evict_relabelled_capacity": ("evict at position", PENDING),
    "live_entry_evicted_as_stale": "stale evict of a live entry",
    "evict_reason_missing": "evict reason must be one of",
    "rebase_before_last_stale_evict": "rebase before expired entries",
    "rebase_non_canonical_epoch": "is not the canonical base epoch",
    "rebase_total_after_wrong": "root_total_after",
    "rebase_total_before_wrong": "root_total_before",
    "rebase_record_removed": ("(evict at pos=", TREE),     # the next evict names the unshifted leaf
    "sample_total_ignores_rebase": TOTAL,
    "rebase_noop": "is not the canonical base epoch",
    "reason_on_insert": "reason is only valid on evict records",
    "config_half_life_hostile": "exceeds 1024",
    "config_bit_budget_violated": "64-bit budget",
    # content, from the log alone and with the manifest
    "insert_source_without_digest": "source requires content_digest",
    "evict_with_content_digest": "content_digest is only valid on insert records",
    "update_with_content_digest": "content_digest is only valid on insert records",
    "insert_without_digest_in_content_log": "insert without content_digest",
    "manifest_tokens_changed": RECOMPUTE,
    "manifest_reward_changed": RECOMPUTE,
    "manifest_prompt_changed": RECOMPUTE,
    "manifest_source_changed": SOURCE,
    "manifest_source_dropped": SOURCE,
    "manifest_entry_version_changed": ": entry_version ",
    "manifest_line_missing": LINE_COUNT,
    "manifest_line_extra": LINE_COUNT,
    "manifest_line_duplicated": LINE_COUNT,
    "manifest_lines_reordered": SLOT,
    "manifest_line_self_consistent_but_different_example": "content_digest disagrees with insert record",
    "manifest_op_counter_changed": SLOT,
    "insert_digest_swapped_chain_consistent": "content_digest disagrees with insert record",
    "insert_source_changed_chain_consistent": SOURCE,
    "insert_source_dropped_chain_consistent": SOURCE,
    # draws and weights
    "config_seed_changed": KEYED,
    "config_buffer_id_changed": KEYED,
    "config_beta_changed": FORMULA,
    "config_seed_missing": "needs all of",
    "config_alpha_not_hex": "alpha must be the canonical float.hex()",
    "config_beta_negative": "beta >= 0",
    "config_beta_overflows_weights": "makes the IS weight unrepresentable",
    "config_beta_huge": "makes the IS weight unrepresentable",
    "sample_record_deleted": KEYED,
    # batch witnesses
    "witness_rows_point_at_each_others_draws": "claims example",
    "witness_digest_changed": "claims example",
    "witness_draw_placed_twice": "is placed twice",
    "witness_row_replaced_twice": "is replaced twice",
    "witness_row_outside_batch": "is outside a batch",
    "witness_nonexistent_draw": "does not exist in this sample",
    "witness_missing_a_draw": "witness places",
    "witness_names_an_older_sample": "but the latest sample is",
    "witness_names_a_nonexistent_sample": "but the latest sample is",
    "witness_tensor_digest_malformed": "tensor_digest must be",
    "witness_batch_rows_zero": "batch_rows must be at least 1",
    "batch_step_forged": "is not the buffer version",
    "witness_duplicated": "already has a batch witness",
    "witness_between_advance_and_pending_evicts": ("batch while expired entries", PENDING),
    "witness_in_a_log_without_content_digests": "needs content digests",
    # telemetry
    "telemetry_ess_changed": "but the sample's weights give",
    "telemetry_ess_denominator_changed": "but the sample's weights give",
    "telemetry_staleness_max_changed": "staleness max/sum",
    "telemetry_staleness_sum_changed": "staleness max/sum",
    "telemetry_replaced_rows_disagree_with_draws": "rows but sample",
    "telemetry_declined_rows_disagree_with_draws": "rows but sample",
    "telemetry_counters_disagree_with_witness": "telemetry counters disagree with the batch witness",
    "telemetry_batch_rows_disagree_with_witness": "telemetry counters disagree with the batch witness",
    "telemetry_step_disagrees_with_witness": "telemetry counters disagree with the batch witness",
    "telemetry_reported_duplicated": "sorted, unique",
    "telemetry_batch_rows_too_small": "more replaced and declined rows",
    "telemetry_names_an_older_sample": "not the latest sample record",
    "telemetry_unlisted_reported_field": "does not match the extra fields",
    "telemetry_reported_not_canonical": "canonical float.hex()",
    "telemetry_reported_nan": "must be finite",
    "telemetry_reported_list_dropped": "does not match the extra fields",
    "telemetry_in_format_1_log": "telemetry records need a format-2 log",
    "telemetry_before_witness": "has no batch witness",
    "telemetry_and_witness_step_forged": "batch witness step 99 is not the buffer version",
    "telemetry_step_forged_without_witness": "telemetry step 99 is not the buffer version",
    "telemetry_in_a_log_without_content_digests": "telemetry of a replayed step needs content digests",
    # quarantine records and reward provenance
    "quarantine_without_predicate": "needs predicate",
    "quarantine_predicate_empty": "needs predicate",
    "quarantine_predicate_not_text": "needs predicate",
    "quarantine_predicate_too_long": "needs predicate",
    "quarantine_without_note": "needs note",
    "quarantine_note_multiline": "needs note",
    "quarantine_note_too_long": "needs note",
    "quarantine_reason_misspelled": "only valid on an evict with reason 'quarantine'",
    "quarantine_texts_on_an_explicit_evict": "only valid on an evict with reason 'quarantine'",
    "quarantine_texts_on_a_capacity_evict": "only valid on an evict with reason 'quarantine'",
    "quarantine_texts_on_an_insert": "only valid on an evict with reason 'quarantine'",
    "quarantine_in_a_format_2_log": "need a format-3 log",
    "quarantine_of_an_empty_slot": "with no live entry",
    "quarantine_keeps_the_leaf": "evict must set the leaf to 0",
    "quarantine_duplicated": ("(evict at pos=", TREE),
    "quarantine_between_advance_and_pending_evicts": ("evict at position", PENDING),
    "manifest_rewards_text_value": "must be a finite number",
    "manifest_rewards_bool_value": "must be a finite number",
    "manifest_rewards_nan": "must be a finite number",
    "manifest_rewards_empty_name": "rewards names",
    "manifest_rewards_not_an_object": "rewards must be an object",
    "manifest_text_field_beside_rewards": "exactly the manifest keys",
    # manifest tamperings against the offline replay
    "manifest_token_changed": RECOMPUTE,
    "manifest_token_appended": RECOMPUTE,
    "manifest_token_bool": "tokens must be non-negative integers",
    "manifest_reward_non_canonical_spelling": "reward_hex must be the canonical float.hex() spelling",
    "manifest_tokens_changed_digest_recomputed": "content_digest disagrees with insert record",
    "manifest_slot_changed": SLOT,
    "manifest_line_dropped": LINE_COUNT,
    "manifest_lines_swapped": SLOT,
    "manifest_line_appended": LINE_COUNT,
    # staleness decisions
    "staleness_kept_row_marked_declined": "declared age but the policy gives kept",
    "staleness_declined_row_marked_kept": "declared kept but the policy gives drift",
    "staleness_decline_reason_changed": "declared ess but the policy gives drift",
    "staleness_scale_removed": "declared scale 1 but the policy gives",
    "staleness_scale_not_reduced": "scale must be a reduced fraction",
    "staleness_scale_above_one": "scale must be a reduced fraction",
    "staleness_declined_row_keeps_a_scale": "a declined draw carries scale",
    "staleness_log_ratio_changed": "but the log-ratios give",
    "staleness_log_ratio_not_canonical": "log_ratios[0] must be the canonical float.hex()",
    "staleness_log_ratio_list_short": "one hex float per draw",
    "staleness_policy_mass_cap_changed": "but the policy gives 1",
    "staleness_policy_ess_floor_raised": "differs from the one recorded earlier",
    "staleness_policy_max_age_tightened": "differs from the one recorded earlier",
    "staleness_policy_max_age_loosened_past_a_decline": "differs from the one recorded earlier",
    "staleness_policy_decline_cap_violated": "exceed the policy's max_declines_per_step",
    "staleness_policy_group_size_changed": "of size 1, not 2",
    "staleness_policy_extra_field": "policy must be an object with exactly the fields",
    "staleness_policy_inactive_but_recorded": "an inactive policy is not recorded",
    "staleness_policy_dropped_declines_kept": "declined but no policy is recorded",
    "staleness_decisions_dropped": "policy and decisions are recorded together",
    "staleness_log_ratios_dropped": "policy and decisions require log_ratios",
    "staleness_group_label_changed": "not 9",
    "staleness_rows_swapped_against_witness": "the decisions disagree with the batch witness",
    "staleness_row_outside_batch": "a decision names a row outside the batch",
    "staleness_decisions_out_of_draw_order": "draw must be 0",
    "staleness_reported_mean_disagrees_with_ratios": "reported log_ratio_mean_abs",
    "staleness_reported_max_disagrees_with_ratios": "reported log_ratio_max_abs",
    "staleness_counters_disagree_with_decisions": "telemetry counters disagree with the batch witness",
    "staleness_evict_reason_ess_is_not_a_reason": "evict reason must be one of",
    "staleness_fields_dropped_from_a_later_record": "a later one cannot leave them out",
    "staleness_policy_dropped_from_a_later_record": "a later one cannot leave it out",
    "staleness_policy_changed_between_records": "differs from the one recorded earlier",
    "staleness_scale_unicode_digit": "scale_num must be a canonical decimal integer",
    "staleness_scale_leading_zero": "scale_den must be a canonical decimal integer",
    # record shapes
    "source_on_a_sample": UNKNOWN,
    "content_digest_on_a_sample": UNKNOWN,
    "sample_draws_nothing": "draws nothing",
    # declared minimum entry age
    "sample_draws_same_step_slot": "below the declared min_entry_age",
    "sample_draw_entry_version_forged": "but the replay holds version",
    "sample_draw_entry_version_missing": "needs entry_version on every draw",
    "min_entry_age_dropped_from_a_later_sample": "a later one cannot leave it out",
    "min_entry_age_changed_between_samples": "the minimum is fixed for a run",
    "min_entry_age_malformed": "min_entry_age must be a non-negative integer",
    "min_entry_age_in_a_log_without_decay_fields": "need a log with decay fields",
}


def expected_reason(description: str) -> Reason:
    """The substring(s) the rejection of ``description`` must contain; KeyError if the mutant has none."""
    if description in EXPECTED:
        return EXPECTED[description]
    for prefix, reason in PREFIXES:
        if description.startswith(prefix):
            return reason
    raise KeyError(f"mutant {description!r} has no expected rejection reason in campaigns.mutation_reasons")


def rejection(check: Callable[[], object]) -> Optional[str]:
    """The checker's error message, or None when ``check`` accepts."""
    try:
        check()
    except CheckerError as exc:
        return str(exc)
    return None


def tally(results: dict, category: str, outcomes: list[tuple[str, Optional[str]]]) -> None:
    """Count ``(description, message)`` outcomes into ``results`` under ``category``.

    A None message is a survivor; a message without the expected substring
    is a wrong-reason rejection; only the rest count as rejected.
    """
    count = 0
    for description, message in outcomes:
        results["total_mutants"] += 1
        results.setdefault("messages", {})[description] = message
        if message is None:
            results["survived"].append(description)
            continue
        expected = expected_reason(description)
        if not all(part in message for part in ((expected,) if isinstance(expected, str) else expected)):
            results.setdefault("wrong_reason", []).append(f"{description}: expected {expected!r} in {message!r}")
            continue
        results["rejected"] += 1
        count += 1
    results["details"].append((category, len(outcomes), count))
