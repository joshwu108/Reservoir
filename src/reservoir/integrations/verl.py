"""Reservoir-backed replay for verl's ``RayPPOTrainer`` under GRPO.

Usage::

    from reservoir.integrations.verl import ReservoirRayPPOTrainer, ReservoirReplay

    trainer = ReservoirRayPPOTrainer(
        config=config, tokenizer=tokenizer, role_worker_mapping=..., resource_pool_manager=...,
        train_dataset=..., val_dataset=..., collate_fn=..., train_sampler=...,
        replay_buffer=ReservoirReplay(capacity=50_000, half_life=4, max_policy_age=16,
                                      seed=0, attest="run-01/attest.jsonl"),
    )
    trainer.init_workers()
    trainer.fit()

``ReservoirRayPPOTrainer`` takes every argument ``RayPPOTrainer`` takes plus
``replay_buffer``; ``benchmarks/modal/verl_replay_real.py`` shows the task
runner that builds it from a Hydra config (``trainer.use_v1=false``).

What it does
------------
The same thing ``reservoir.integrations.trl`` does for TRL: a prompt
whose ``n`` responses all got the same reward is a "dead" group whose
GRPO advantages are exactly zero (``compute_grpo_outcome_advantage``
divides a zero numerator), so it contributes no gradient. The trainer
keeps the live rollouts in a ``RolloutBuffer`` and, whenever a training
step produces dead groups, fills their rows with rollouts replayed from
the buffer: sampled by exact, age-decayed priority with keyed
deterministic draws, carrying their behavior logprobs so the actor loss
applies a real off-policy ratio, and with every insertion, draw and
eviction written to the attestation log that ``reservoir-verify`` checks.

Where it plugs in
-----------------
``RayPPOTrainer.fit`` runs on the Ray driver. Per step it generates,
scores, computes ``old_log_probs`` and the advantages on the whole batch
(a ``DataProto`` on the driver) and then calls ``self._update_actor(batch)``,
which converts the batch to verl's no-padding TensorDict and ships it to
the actor workers. ``ReservoirReplayMixin`` overrides that one method: it
hands the batch to ``ReservoirReplay.mix`` and passes the result to the
original. No verl method body is copied, and since the driver holds the
whole batch there is no rank-ownership question: the driver owns the
buffer and the log.

The batch keys the adapter reads and rewrites are listed in
``_verl_rows``; rows of one prompt are found by ``non_tensor_batch["uid"]``
(``balance_batch`` reorders rows, so groups are not contiguous). The mixed
``DataProto`` carries the rewritten tensors, ``uid`` (a replayed row takes
the uid of the group it came from) and the original ``meta_info`` with
``global_token_num`` recomputed from the final ``attention_mask``. The
other ``non_tensor_batch`` columns (``data_source``, ``reward_model``,
``extra_info``, ...) describe the prompts that were generated and would be
stale for a replayed row, so they are not carried; the actor reads none
of them. ``rollout_log_probs`` (vLLM's sampling logprobs, attached by
default) is dropped from the mixed batch when rollout correction is off,
the only case in which nothing downstream reads it; with
``algorithm.rollout_correction`` on, the batch is refused. ``fit`` keeps
its own reference to the generated batch, so verl's data metrics
(``critic/score/*``, ``response_length/*``) describe the generated batch
while the actor metrics describe the replayed one.

Per training step:

1. Behavior logprobs are verl's ``old_log_probs``, which ``fit`` computes
   on every batch before the update, so no extra forward is needed.
2. Store: every row of a live group becomes a ``Rollout`` (``reward`` =
   the row's advantage, the value the loss consumes) and each group one
   ``add_group(prompt_id, model_version=global_steps, ...)``.
3. Replay: with ``d`` dead rows, ``sample(d, current_version=global_steps)``
   draws rollouts and writes them into the dead rows, each advantage
   multiplied by its importance-sampling weight (exact for every loss
   mode that is positively homogeneous in the advantage: ``vanilla``,
   ``gspo``, ``gpg``, ``geo_mean``; ``clip_cov`` and ``kl_cov`` rank
   tokens by covariance with the advantage and are refused with
   ``beta > 0``). The batch is padded if a replayed sequence is longer
   than the current width.
4. Every replaced row is re-read and checked against the sampled rollout
   (``verify_written_rows``), a batch witness naming which row holds
   which draw is written, replay health is measured (one
   ``_compute_old_log_prob`` call over the replayed rows for the
   log-ratios) and logged under ``reservoir/*`` next to ``actor/*``, and
   the optional drift gate declines rows whose log-ratio is too large.

A step with no dead groups, or an empty buffer, hands the original
``_update_actor`` the very ``DataProto`` ``fit`` built. Versions,
half-life and ``max_policy_age`` are counted in ``trainer.global_steps``
(1 during the first step).

Checkpoints: ``_save_checkpoint`` also snapshots a durable buffer under
``step-<global_steps>`` and prunes buffer checkpoints to the
``global_step_N`` directories the trainer kept; ``_load_checkpoint`` at a
non-zero step rewinds the buffer to that snapshot, so a resumed run
continues the draw counter and the attestation chain from the checkpoint.

Scope and guards
----------------
``algorithm.adv_estimator=grpo`` only; text models with 2-D position ids;
single-turn prefix response masks; no critic (``values``), no rollout
correction (``rollout_is_weights``), no KL-in-reward, no multimodal
inputs, no MoE routing or distillation tensors. Each is refused by name
before anything is stored. Tested against the verl release in
``_verl_compat`` with a fake trainer and one real run on one GPU; that
release's default trainer is the TransferQueue-based "V1" loop, and this
adapter targets the ``DataProto`` trainer (``trainer.use_v1=false``),
which verl marks deprecated. The hook-ran check raises on the second
training step's logprob computation, at every save and at the end of
``fit`` if ``_update_actor`` never ran through the override, so a verl
rename cannot silently disable replay. No training-quality claim is made
for replay; see ``docs/nonclaims.md``.

``ReservoirReplay`` and ``ReservoirReplayMixin`` import nothing from verl
and are tested with a fake trainer. ``ReservoirRayPPOTrainer`` is built
on first access through ``build_trainer_class``, which imports verl.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path
from typing import Any, Final, Optional, Union

import numpy as np

from reservoir.durable_rollout import DurableRolloutBuffer
from reservoir.integrations._trl_lifecycle import checkpoint_tag, model_binding, resume_from_checkpoint
from reservoir.integrations._trl_staleness import RowDecision, StalenessPolicy, decide, evictions_for, place, resolve_policy
from reservoir.integrations._trl_telemetry import StepTelemetry, summarize
from reservoir.integrations._verl_compat import require_verl
from reservoir.integrations._verl_rows import (
    global_token_num,
    rows_to_groups,
    tensor_digest,
    verify_written_rows,
    write_rows,
)
from reservoir.integrations.trl import StoredAdvantagePriority, _validated_gate
from reservoir.priorities import PriorityStrategy, ReplaySignal, validated_rescore
from reservoir.rollout_attest import AttestTarget
from reservoir.rollout_buffer import RolloutBatch, RolloutBuffer

UNSUPPORTED_BATCH_KEYS: Final[tuple[str, ...]] = (
    "values", "rollout_is_weights", "routed_experts", "teacher_logprobs", "teacher_ids", "sum_pi_squared",
)
"""Batch tensors that would need per-row replacement logic this adapter does not have."""

UNSUPPORTED_NON_TENSOR_KEYS: Final[tuple[str, ...]] = ("multi_modal_inputs", "multi_modal_data")
"""Vision inputs; a replayed row has none to put there. Refused when any row's entry is non-empty."""

ROLLOUT_LOGPROBS_KEY: Final[str] = "rollout_log_probs"
"""vLLM's sampling logprobs; dropped when nothing downstream reads them, refused otherwise."""

ROLLOUT_CORRECTION_KEYS: Final[tuple[str, ...]] = ("bypass_mode", "rollout_is", "rollout_rs")
"""``algorithm.rollout_correction.*`` settings under which the actor reads ``rollout_log_probs``."""

SUPPORTED_ADV_ESTIMATORS: Final[tuple[str, ...]] = ("grpo",)

NON_HOMOGENEOUS_LOSS_MODES: Final[tuple[str, ...]] = ("clip_cov", "kl_cov")
"""Loss modes whose token selection depends on the advantage scale; refused with ``beta > 0``."""

STAT_NAMES: Final[tuple[str, ...]] = (
    "hook_calls", "ingested_rows", "ingested_groups", "dead_groups", "skipped_rows", "clamped_logprobs",
    "replaced_rows", "near_dead_groups", "declined_rows", "telemetry_forwards", "rescored_rows", "rescaled_rows",
    "dropped_rollout_logprobs",
)

CHECKPOINT_DIR_PREFIX: Final[str] = "global_step_"
"""verl writes ``<default_local_dir>/global_step_<n>`` per saved step."""

LOGPROB_FORWARD_KEYS: Final[tuple[str, ...]] = (
    "input_ids", "attention_mask", "position_ids", "response_mask", "prompts", "responses",
)
"""What ``_compute_old_log_prob`` needs from the replayed rows (the no-padding converter asserts the first four)."""


def config_value(config: Any, path: str, default: Any = None) -> Any:
    """``config.a.b.c`` through attribute or item access (OmegaConf or namespaces); ``default`` if absent or None."""
    node = config
    for part in path.split("."):
        if node is None:
            return default
        if hasattr(node, part):
            node = getattr(node, part)
            continue
        try:
            node = node[part]
        except (KeyError, TypeError, IndexError):
            return default
    return default if node is None else node


_MISSING = object()


def config_setting(config: Any, node_path: str, key: str, absent_node: Any) -> Any:
    """``key`` under the config node at ``node_path``.

    A missing node means a verl without that feature and yields
    ``absent_node``; a node that exists but lacks ``key`` means verl renamed
    a setting the adapter's guards read, and raises rather than reading
    "off". A key whose value is None yields None.
    """
    node = config_value(config, node_path, _MISSING)
    if node is _MISSING:
        return absent_node
    if hasattr(node, key):
        return getattr(node, key)
    try:
        return node[key]
    except (KeyError, TypeError, IndexError):
        raise ValueError(
            f"the config has {node_path} but no {node_path}.{key}; the installed verl renamed a setting the "
            "adapter checks, so it cannot tell whether that feature is on"
        ) from None


class ReservoirReplay:
    """The replay buffer handed to ``ReservoirRayPPOTrainer(replay_buffer=...)``.

    Parameters mirror ``reservoir.integrations.trl.ReservoirReplay``:
    ``capacity``, ``half_life`` and ``max_policy_age`` (in training
    steps), ``beta`` (``0.0`` disables importance weighting), ``seed``,
    ``attest``, ``directory`` (crash-atomic buffer on disk), ``source``
    (provenance tag on every insert), ``manifest`` (requires ``attest``),
    ``telemetry`` (replay health every step; the log-ratios cost one
    ``_compute_old_log_prob`` call over the replayed rows),
    ``staleness_policy`` (a ``StalenessPolicy`` or preset name; its
    decisions go to the telemetry record), ``max_log_ratio`` and
    ``max_declines_per_step`` (the legacy drift gate, off
    by default).

    Attributes
    ----------
    buffer
        The underlying ``RolloutBuffer`` or ``DurableRolloutBuffer``.
    is_owner : bool
        Always True: the Ray driver holds the whole batch and the buffer.
    stats : dict[str, int]
        Counters named in ``STAT_NAMES``.
    last_replay : RolloutBatch | None
        The sampled batch of the most recent step that replayed rows.
    last_telemetry : StepTelemetry | None
        The most recent step's measurements (None with telemetry off).
    """

    is_owner: bool = True

    def __init__(
        self,
        capacity: int,
        priority: Optional[PriorityStrategy] = None,
        half_life: int = 4,
        max_policy_age: int = 16,
        beta: float = 0.4,
        seed: int = 0,
        attest: AttestTarget = None,
        directory: Union[str, Path, None] = None,
        source: Optional[str] = None,
        manifest: Union[str, Path, None] = None,
        telemetry: bool = True,
        staleness_policy: Union[StalenessPolicy, str, None] = None,
        max_log_ratio: Optional[float] = None,
        max_declines_per_step: Optional[int] = None,
        **rollout_buffer_kwargs: Any,
    ) -> None:
        if manifest is not None and attest is None:
            raise ValueError("manifest requires attestation: pass attest=<path or log> as well")
        kwargs = dict(
            capacity=capacity, priority=priority if priority is not None else StoredAdvantagePriority(),
            half_life=half_life, max_policy_age=max_policy_age, beta=beta, seed=seed, attest=attest,
            manifest=manifest, **rollout_buffer_kwargs,
        )
        self.buffer: Union[RolloutBuffer, DurableRolloutBuffer] = (
            RolloutBuffer(**kwargs) if directory is None else DurableRolloutBuffer(directory, **kwargs)
        )
        self.source = source
        self.beta = float(beta)
        self.telemetry = bool(telemetry)
        self.policy: StalenessPolicy = resolve_policy(staleness_policy, _validated_gate(max_log_ratio),
                                                      max_declines_per_step, self.telemetry)
        self.max_log_ratio = self.policy.max_log_ratio
        self.max_declines_per_step = self.policy.max_declines_per_step
        self.last_telemetry: Optional[StepTelemetry] = None
        self.last_replay: Optional[RolloutBatch] = None
        self.stats: dict[str, int] = {name: 0 for name in STAT_NAMES}

    # -- the hook ------------------------------------------------------------

    def mix(self, data: Any, trainer: Any) -> Any:
        """Store the live rows of ``data`` and replace its dead rows with replayed ones.

        ``data`` is the ``DataProto`` ``fit`` hands to ``_update_actor``;
        ``trainer`` is the ``RayPPOTrainer`` (or a stand-in exposing
        ``global_steps``, ``config``, ``tokenizer`` and
        ``_compute_old_log_prob``). Returns ``data`` itself when nothing was
        replaced, otherwise a new ``DataProto``; the tensors of ``data`` are
        never modified. Order matters: groups are stored, the buffer is
        advanced to ``global_steps`` (evicting entries older than
        ``max_policy_age``), and only then are dead rows replaced from
        whatever is still live. If the buffer raises while groups are
        being stored, the groups stored before it stay stored and the
        counters are not updated.
        """
        step = int(trainer.global_steps)
        self._check_batch(data, trainer)
        self._check_version(step)
        tensors = {key: data.batch[key] for key in data.batch.keys() if key != ROLLOUT_LOGPROBS_KEY}
        uids = [str(u) for u in data.non_tensor_batch["uid"]]
        conversion = rows_to_groups(tensors, uids, step=step)
        self.last_replay = None
        self.stats["hook_calls"] += 1
        for group in conversion.groups:
            self.buffer.add_group(group.prompt_id, step, group.rollouts, source=self.source)
        self.stats["ingested_rows"] += sum(len(g.rollouts) for g in conversion.groups)
        self.stats["ingested_groups"] += len(conversion.groups)
        for name in ("dead_groups", "near_dead_groups", "skipped_rows", "clamped_logprobs"):
            self.stats[name] += getattr(conversion, name)
        if step > self.buffer.current_version:
            self.buffer.advance(step)
        if not conversion.dead_rows or self.buffer.size == 0 or self.buffer.total == 0:
            self._telemetry(step, None, [], [], len(uids), conversion, 0)
            return data
        return self._replay(data, tensors, uids, trainer, step, conversion)

    def _check_version(self, step: int) -> None:
        if step < self.buffer.current_version:
            raise ValueError(
                f"global_steps {step} is below the buffer's current version {self.buffer.current_version}; "
                "versions only move forward. A resumed run must reopen the buffer at or behind the step it "
                "resumes from"
            )

    def _check_batch(self, data: Any, trainer: Any) -> None:
        """Refuse configurations the adapter does not handle, before touching anything."""
        if not hasattr(type(data), "from_dict"):
            raise TypeError(f"{type(data).__name__} has no from_dict; is this a verl DataProto?")
        if "uid" not in data.non_tensor_batch:
            raise ValueError("the batch has no non_tensor_batch['uid']; rows cannot be grouped by prompt")
        keys = set(data.batch.keys())
        for key in UNSUPPORTED_BATCH_KEYS:
            if key in keys:
                raise NotImplementedError(
                    f"ReservoirReplay cannot replay batches with {key!r} (a critic, rollout correction, "
                    "MoE routing and distillation tensors are unsupported)"
                )
        for key in UNSUPPORTED_NON_TENSOR_KEYS:
            # The agent loop attaches the column to every batch; a text model's rows hold None or {}.
            if key in data.non_tensor_batch and any(_has_content(v) for v in data.non_tensor_batch[key]):
                raise NotImplementedError(f"ReservoirReplay cannot replay batches with {key!r} (vision inputs are unsupported)")
        config = getattr(trainer, "config", None)
        estimator = config_value(config, "algorithm.adv_estimator")
        if estimator not in SUPPORTED_ADV_ESTIMATORS:
            raise NotImplementedError(
                f"algorithm.adv_estimator={estimator!r}; the adapter's dead-group criterion is defined for "
                f"{', '.join(SUPPORTED_ADV_ESTIMATORS)} only"
            )
        on = [k for k in ROLLOUT_CORRECTION_KEYS
              if config_setting(config, "algorithm.rollout_correction", k, absent_node=None) not in (None, False)]
        if on:
            raise NotImplementedError(
                f"algorithm.rollout_correction.{on[0]} is set, so the actor reads {ROLLOUT_LOGPROBS_KEY!r}, which "
                "replayed rows cannot supply; rollout correction is unsupported"
            )
        loss_mode = config_setting(config, "actor_rollout_ref.actor.policy_loss", "loss_mode", absent_node="vanilla")
        if self.beta > 0.0 and loss_mode in NON_HOMOGENEOUS_LOSS_MODES:
            raise ValueError(
                f"policy_loss.loss_mode {loss_mode!r} selects tokens by covariance with the advantage, so "
                "importance weights cannot be folded into it; use ReservoirReplay(beta=0.0) with it"
            )

    def _replay(self, data: Any, tensors: dict, uids: list[str], trainer: Any, step: int, conversion) -> Any:
        """Sample one rollout per dead row, write them in, gate, verify, witness, measure."""
        dead_rows = list(conversion.dead_rows)
        batch = self.buffer.sample(len(dead_rows), current_version=step)
        rewards = [r.reward for r in batch.rollouts]
        unit = place(rewards, batch.is_weights, ()).weighted           # importance-weighted, before the policy
        pad = _pad_token_id(trainer)
        provisional = write_rows(tensors, dead_rows, batch.rollouts, unit, pad)
        ratios = self._log_ratios(data, provisional, dead_rows, trainer)
        # verl rows of one prompt share a uid and may be reordered (balance_batch), so the policy's groups are
        # declared per decision (uids in order of first appearance among the dead rows) and the record's
        # group_size is None.
        labels = {uid: g for g, uid in enumerate(dict.fromkeys(uids[r] for r in dead_rows))}
        decisions = self._decide(batch, ratios, dead_rows, [labels[uids[r]] for r in dead_rows])
        kept, declined, weighted, rescaled = place(rewards, batch.is_weights, decisions)
        if declined or rescaled:
            # Declined draws leave their dead rows dead; rescaled draws carry the capped advantage; the witness
            # digest is over what is returned.
            new = (write_rows(tensors, [dead_rows[k] for k in kept], [batch.rollouts[k] for k in kept],
                              weighted, pad) if kept else dict(tensors))
        else:
            new = provisional
        rows = [dead_rows[k] for k in kept]
        verify_written_rows(new, rows, [batch.rollouts[k] for k in kept], weighted)
        self.buffer.witness_batch(batch, step=step, batch_rows=len(uids), rows=rows,
                                  tensor_digest=tensor_digest(new), declined=declined)
        # Telemetry describes the sampled batch and is recomputed by the checker from the live
        # slots, so it is written before any declined entry is evicted.
        self._telemetry(step, batch, ratios, kept, len(uids), conversion, len(declined), decisions)
        for slot, reason in evictions_for(batch.indices, decisions):
            self.buffer.evict(slot, reason)
        self._rescore(batch, kept, weighted, ratios, step)
        self.last_replay = batch
        self.stats["replaced_rows"] += len(kept)
        self.stats["declined_rows"] += len(declined)
        self.stats["rescaled_rows"] += rescaled
        if ROLLOUT_LOGPROBS_KEY in data.batch.keys():
            self.stats["dropped_rollout_logprobs"] += 1
        new_uids = list(uids)
        for k in kept:
            new_uids[dead_rows[k]] = str(batch.rollouts[k].metadata.get("uid", batch.groups[k].prompt_id))
        meta_info = dict(data.meta_info)
        meta_info["global_token_num"] = global_token_num(new["attention_mask"])
        return type(data).from_dict(
            tensors=new, non_tensors={"uid": np.array(new_uids, dtype=object)}, meta_info=meta_info,
        )

    def _decide(self, batch: RolloutBatch, ratios: list[float], rows: list[int], groups: list[int]) -> tuple[RowDecision, ...]:
        """The staleness policy's decision for every draw; ``()`` when the policy is off."""
        if not self.policy.active:
            return ()
        ages = [self.buffer.current_version - v for v in batch.model_versions]
        return decide(self.policy, ratios=ratios, is_weights=batch.is_weights, ages=ages, rows=rows, groups=groups)

    def _log_ratios(self, data: Any, tensors: dict, rows: list[int], trainer: Any) -> list[float]:
        """Sequence log-ratios ``sum(current - behavior)`` of the replayed rows, via ``_compute_old_log_prob``."""
        if not rows or (not self.telemetry and not self.policy.active):
            return []
        import torch

        # The worker group splits the batch evenly across its data-parallel ranks and each rank
        # chunks its share into fixed micro-batches, and both splits assert exactness, so the
        # subset is padded with repeats of its first row to a multiple of both; the padded rows'
        # results are discarded.
        multiple = logprob_batch_multiple(trainer)
        padded_rows = rows + [rows[0]] * ((-len(rows)) % multiple)
        idx = torch.tensor(padded_rows)
        meta_info = dict(data.meta_info)
        meta_info["global_token_num"] = global_token_num(tensors["attention_mask"][idx])
        subset = type(data).from_dict(
            tensors={key: tensors[key][idx] for key in LOGPROB_FORWARD_KEYS}, meta_info=meta_info,
        )
        self.stats["telemetry_forwards"] += 1
        result = trainer._compute_old_log_prob(subset)
        proto = result[0] if isinstance(result, tuple) else result  # 0.9 returns (DataProto, mfu)
        current = proto.batch["old_log_probs"][: len(rows)]
        idx = idx[: len(rows)]
        behavior = tensors["old_log_probs"][idx]
        if tuple(current.shape) != tuple(behavior.shape):
            raise ValueError(
                f"_compute_old_log_prob returned old_log_probs of shape {tuple(proto.batch['old_log_probs'].shape)} "
                f"for {len(padded_rows)} rows, expected a {tuple(behavior.shape)} prefix"
            )
        mask = tensors["response_mask"][idx].bool()
        diff = current.to(behavior.dtype).to(behavior.device) - behavior
        ratios = torch.where(mask, diff, torch.zeros_like(diff)).sum(dim=1)   # masked NaNs must not poison the row
        return [float(x) for x in ratios.detach().cpu().tolist()]

    def _rescore(self, batch: RolloutBatch, kept: list[int], weighted: list[float], ratios: list[float], step: int) -> None:
        """Ask the strategy for new priorities of the placed rollouts; a slot drawn twice is rescored once."""
        indices: list[int] = []
        scores: list[float] = []
        seen: set[int] = set()
        for position, k in enumerate(kept):
            slot = batch.indices[k]
            if slot in seen:
                continue
            seen.add(slot)
            ratio = ratios[k] if ratios else None
            signal = ReplaySignal(
                advantage=weighted[position], is_weight=float(batch.is_weights[k]),
                log_ratio=ratio if ratio is not None and math.isfinite(ratio) else None,
                age=self.buffer.current_version - batch.model_versions[k], step=step,
            )
            value = validated_rescore(self.buffer.priority, batch.rollouts[k], batch.groups[k], signal)
            if value is not None:
                indices.append(slot)
                scores.append(value)
        if indices:
            self.buffer.update_priorities(indices, scores)
            self.stats["rescored_rows"] += len(indices)

    def _telemetry(self, step: int, batch: Optional[RolloutBatch], ratios: list[float], kept: list[int],
                   batch_rows: int, conversion, declined: int, decisions: tuple[RowDecision, ...] = ()) -> None:
        if not self.telemetry:
            self.last_telemetry = None
            return
        point = summarize(batch, self.buffer.current_version, ratios, kept, batch_rows,
                          conversion.dead_groups, conversion.near_dead_groups, declined, decisions, self.policy)
        self.last_telemetry = point
        self.buffer.record_telemetry(
            step, point.counts(), batch, point.reported(),
            log_ratios=list(ratios) if batch is not None else None,
            policy=self.policy.to_record(None) if decisions else None,
            decisions=[d.to_record() for d in decisions] if decisions else None,
        )

    def close(self) -> None:
        """Close the attestation file, if one was opened."""
        self.buffer.close()

    def __repr__(self) -> str:
        source = f", source={self.source!r}" if self.source is not None else ""
        return f"ReservoirReplay({self.buffer!r}, beta={self.beta}{source})"


def _has_content(value: Any) -> bool:
    """Whether a per-row multimodal entry carries anything (None and empty containers do not)."""
    if value is None:
        return False
    try:
        return len(value) > 0
    except TypeError:
        return True


def logprob_batch_multiple(trainer: Any) -> int:
    """Row count the actor's ``compute_log_prob`` requires a batch to be a multiple of.

    The data-parallel world size of the actor worker group times, unless
    the log-prob forward uses dynamic batching
    (``rollout.log_prob_use_dynamic_bsz``), its micro-batch size
    (``rollout.log_prob_micro_batch_size_per_gpu``). ``fit``'s own batches
    satisfy this by configuration; the telemetry subset has to be padded.
    """
    world_size = int(getattr(getattr(trainer, "actor_rollout_wg", None), "world_size", 1) or 1)
    rollout = config_value(getattr(trainer, "config", None), "actor_rollout_ref.rollout")
    if config_value(rollout, "log_prob_use_dynamic_bsz", False):
        return world_size
    micro = config_value(rollout, "log_prob_micro_batch_size_per_gpu", 1)
    return world_size * max(1, int(micro))


def _pad_token_id(trainer: Any) -> int:
    pad = getattr(getattr(trainer, "tokenizer", None), "pad_token_id", None)
    if pad is None:
        raise ValueError("trainer.tokenizer.pad_token_id is None; replayed rows cannot be padded into place")
    return int(pad)


def assert_hook_ran(replay: ReservoirReplay, global_steps: int, resumed_from: int = 0) -> None:
    """Raise if a training step has completed in this process without the replay hook ever running.

    ``global_steps`` is 1 during the first step, so a value of 2 or more
    means at least one ``_update_actor`` call is behind us. A resumed run
    starts at ``resumed_from`` (the loaded checkpoint's step), which does
    not count: only steps trained by this process can have run the hook.
    """
    if int(global_steps) - int(resumed_from) >= 2 and replay.stats["hook_calls"] == 0:
        raise RuntimeError(
            "ReservoirRayPPOTrainer completed a training step but its replay hook never ran; the "
            "installed verl no longer routes the batch through RayPPOTrainer._update_actor the way "
            "this adapter expects"
        )


def append_metrics(actor_output: Any, point: Optional[StepTelemetry]) -> None:
    """Put the step's ``reservoir/*`` metrics into the actor output ``fit`` reduces and logs.

    ``_update_actor`` returns a ``DataProto`` whose ``meta_info["metrics"]``
    maps ``actor/*`` names to values (or lists of values); ``fit`` runs
    ``reduce_metrics`` over it and logs the result. With telemetry off there
    is nothing to add; otherwise an output without that dict is an error,
    since the telemetry would vanish without one.
    """
    if point is None:
        return
    metrics = getattr(getattr(actor_output, "meta_info", None), "get", lambda *_: None)("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError(
            "RayPPOTrainer._update_actor returned no meta_info['metrics'] dict, so the reservoir/* telemetry "
            "cannot be logged; the installed verl changed the actor output this adapter relies on"
        )
    for name, value in point.metrics().items():
        metrics[name] = value


# ---------------------------------------------------------------------------
# Checkpoint binding
# ---------------------------------------------------------------------------

def trainer_checkpoint_steps(local_dir) -> Optional[set[int]]:
    """Steps of the ``global_step_N`` directories under ``local_dir``, or None if it cannot be listed."""
    if local_dir is None:
        return None
    root = Path(local_dir)
    if not root.is_dir():
        return None
    steps = set()
    for entry in root.iterdir():
        suffix = entry.name[len(CHECKPOINT_DIR_PREFIX):]
        if entry.is_dir() and entry.name.startswith(CHECKPOINT_DIR_PREFIX) and suffix.isdigit():
            steps.add(int(suffix))
    return steps


def verl_checkpoint_dir(local_dir, global_steps: int) -> Optional[Path]:
    """``<local_dir>/global_step_<n>`` when ``local_dir`` is known, else None (it need not exist)."""
    if local_dir is None:
        return None
    return Path(local_dir) / f"{CHECKPOINT_DIR_PREFIX}{int(global_steps)}"


def bind_checkpoint(replay: ReservoirReplay, global_steps: int, local_dir=None) -> None:
    """When the trainer saves ``global_step_<n>``, snapshot a durable buffer as ``step-<n>``.

    With ``local_dir`` the digest of ``global_step_<n>`` is recorded in the
    buffer checkpoint (a resume refuses a different model checkpoint) and
    the buffer's checkpoints are pruned to the steps the trainer still has
    (it removes old ones under ``max_actor_ckpt_to_keep`` before this is
    called). A buffer without a directory has nothing to bind; the call is
    a no-op.
    """
    if not isinstance(replay.buffer, DurableRolloutBuffer):
        return
    binding = model_binding(verl_checkpoint_dir(local_dir, global_steps))
    replay.buffer.checkpoint(checkpoint_tag(global_steps), binding=binding)
    steps = trainer_checkpoint_steps(local_dir)
    if steps is not None:
        replay.buffer.prune_checkpoints({checkpoint_tag(n) for n in steps | {int(global_steps)}})


# ---------------------------------------------------------------------------
# The mixin and the trainer class
# ---------------------------------------------------------------------------

class ReservoirReplayMixin:
    """Routes ``_update_actor`` through the replay and binds the buffer to verl's checkpoints.

    Place it before ``RayPPOTrainer`` in the base list. The class using it
    must set ``self.replay_buffer`` to a ``ReservoirReplay``.
    """

    replay_buffer: ReservoirReplay
    _reservoir_resumed_from: int = 0

    def _check_hook_ran(self) -> None:
        assert_hook_ran(self.replay_buffer, getattr(self, "global_steps", 0), self._reservoir_resumed_from)

    def _update_actor(self, batch: Any, *args: Any, **kwargs: Any):  # type: ignore[override]
        mixed = self.replay_buffer.mix(batch, self)
        output = super()._update_actor(mixed, *args, **kwargs)  # type: ignore[misc]
        append_metrics(output, self.replay_buffer.last_telemetry)
        return output

    def _compute_old_log_prob(self, batch: Any, *args: Any, **kwargs: Any):  # type: ignore[override]
        # Runs once per training step before _update_actor: the earliest point to notice that
        # the previous step's update bypassed the override.
        self._check_hook_ran()
        return super()._compute_old_log_prob(batch, *args, **kwargs)  # type: ignore[misc]

    def _save_checkpoint(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        result = super()._save_checkpoint(*args, **kwargs)  # type: ignore[misc]
        self._check_hook_ran()
        step = int(self.global_steps)  # type: ignore[attr-defined]
        bind_checkpoint(self.replay_buffer, step, config_value(self.config, "trainer.default_local_dir"))  # type: ignore[attr-defined]
        return result

    def _load_checkpoint(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        result = super()._load_checkpoint(*args, **kwargs)  # type: ignore[misc]
        self._reservoir_resumed_from = int(self.global_steps)  # type: ignore[attr-defined]
        local_dir = config_value(self.config, "trainer.default_local_dir")  # type: ignore[attr-defined]
        resume_from_checkpoint(self.replay_buffer, self._reservoir_resumed_from,
                               verl_checkpoint_dir(local_dir, self._reservoir_resumed_from))
        return result

    def fit(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        result = super().fit(*args, **kwargs)  # type: ignore[misc]
        self._check_hook_ran()
        return result


@functools.lru_cache(maxsize=None)
def build_trainer_class() -> type:
    """Import verl (checked by ``require_verl``) and build ``ReservoirRayPPOTrainer``."""
    support = require_verl()

    class ReservoirRayPPOTrainer(ReservoirReplayMixin, support.ray_trainer):  # type: ignore[misc,valid-type]
        """``RayPPOTrainer`` with Reservoir replay of dead GRPO groups. See the module docstring."""

        def __init__(self, *args: Any, replay_buffer: ReservoirReplay, **kwargs: Any) -> None:
            if not isinstance(replay_buffer, ReservoirReplay):
                raise TypeError(f"replay_buffer must be a ReservoirReplay, got {type(replay_buffer).__name__}")
            self.replay_buffer = replay_buffer
            super().__init__(*args, **kwargs)

    ReservoirRayPPOTrainer.__qualname__ = "ReservoirRayPPOTrainer"
    ReservoirRayPPOTrainer.__module__ = __name__
    return ReservoirRayPPOTrainer


def __getattr__(name: str) -> Any:
    """Build ``ReservoirRayPPOTrainer`` on first access so importing this module needs no verl."""
    if name == "ReservoirRayPPOTrainer":
        return build_trainer_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | {"ReservoirRayPPOTrainer"})


# ``ReservoirRayPPOTrainer`` is deliberately absent: a star import must not require verl.
__all__ = [
    "CHECKPOINT_DIR_PREFIX",
    "NON_HOMOGENEOUS_LOSS_MODES",
    "ROLLOUT_CORRECTION_KEYS",
    "ROLLOUT_LOGPROBS_KEY",
    "ReservoirReplay",
    "ReservoirReplayMixin",
    "STAT_NAMES",
    "SUPPORTED_ADV_ESTIMATORS",
    "UNSUPPORTED_BATCH_KEYS",
    "UNSUPPORTED_NON_TENSOR_KEYS",
    "append_metrics",
    "assert_hook_ran",
    "bind_checkpoint",
    "verl_checkpoint_dir",
    "build_trainer_class",
    "config_setting",
    "config_value",
    "logprob_batch_multiple",
    "trainer_checkpoint_steps",
]
