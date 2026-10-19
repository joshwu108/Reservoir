"""Reservoir-backed replay for TRL's ``GRPOTrainer``.

Usage::

    from trl import GRPOConfig
    from reservoir.integrations.trl import ReservoirGRPOTrainer, ReservoirReplay

    trainer = ReservoirGRPOTrainer(
        model=model,
        args=GRPOConfig(...),
        train_dataset=dataset,
        reward_funcs=[...],
        replay_buffer=ReservoirReplay(capacity=50_000, half_life=4, max_policy_age=16,
                                      seed=0, attest="run-01/attest.jsonl"),
    )
    trainer.train()

What it does
------------
GRPO throws every rollout away after one update, and a prompt whose
completions all got the same reward (a "dead" group) contributes no
gradient at all. ``ReservoirGRPOTrainer`` keeps the useful rollouts in a
``RolloutBuffer`` and, whenever a generation step produces dead groups,
fills their rows with rollouts replayed from the buffer: sampled by
exact, age-decayed priority with keyed deterministic draws, carrying
their behavior logprobs so the loss applies a real off-policy ratio, and
with every insertion, draw and eviction written to the attestation log
that ``python -m checker.verify`` checks.

Where it plugs in
-----------------
``GRPOTrainer._prepare_inputs`` calls ``_generate_and_score_completions``
once per generation step and feeds the dict it returns, split into
micro-batches, to ``_compute_loss``. ``ReservoirReplayMixin`` overrides
that one method: it calls the original and hands the result to
``ReservoirReplay.mix``. No TRL method body is copied; the adapter only
reads and rewrites the keys ``_compute_loss`` consumes:

================================  ==========================================
key                               shape, meaning
================================  ==========================================
``prompt_ids``, ``prompt_mask``   ``(B, Lp)`` long, left-padded
``completion_ids``,
``completion_mask``               ``(B, Lc)`` long, right-padded prefix mask
``advantages``                    ``(B,)`` float32, group-centred; a dead
                                  group is ``0.0`` in every row
``old_per_token_logps``           ``(B, Lc)`` float, optional
``ref_per_token_logps``           ``(B, Lc)`` float, present iff ``beta != 0``
``num_items_in_batch``            0-d tensor, loss normaliser
================================  ==========================================

Rows ``[g * G, (g + 1) * G)`` are the ``G = num_generations`` completions
of one prompt (the trainer shuffles rows only after this hook runs).

Per generation step, in train mode only:

1. Behavior logprobs: TRL computes ``old_per_token_logps`` only when
   generation and optimizer steps are misaligned or vLLM importance
   correction is on. When the key is absent the adapter runs one
   no-grad forward through ``trainer._get_per_token_logps_and_entropies``
   for the fresh rows, because a replayed rollout needs them and
   ``Rollout`` requires them.
2. Replay: with ``d`` dead groups, ``sample(d * G, current_version=
   global_step)`` draws rollouts stored at earlier steps (possibly from
   different prompts; the loss does not need group contiguity) and writes
   them into the dead rows. Each replayed advantage is multiplied by its importance-sampling
   weight, which is exact for every loss type that is positively
   homogeneous in the advantage (all of them except ``vespo``). The batch
   is padded if a replayed sequence is longer than the current width,
   ``old_per_token_logps`` is attached for all rows, and
   ``num_items_in_batch`` is recomputed from the final mask.
3. Staleness policy (``staleness_policy=``, off by default): from each
   replayed row's sequence log-ratio, exact importance weight and age, the
   policy declines rows (age bound, ESS floor, legacy ``max_log_ratio``)
   or rescales a group's advantages by an exact fraction (group-mass
   cap), in that order; see ``_trl_staleness``. A declined row's dead
   slot stays dead. Every decision is written to the telemetry record and
   replayed by the checker; the batch witness digests the rescaled
   advantages.
4. Store: every row of a live group becomes a ``Rollout`` (``reward`` =
   TRL's advantage, the value the loss consumes) and each group is one
   ``add_group(prompt_id, model_version=global_step, ...)``.

A batch with no dead groups, or an empty buffer, is returned as the very
dict TRL produced. When TRL did not compute ``old_per_token_logps``, one
extra no-grad forward has run by then, which consumes RNG state under
dropout; apart from that a step without replay matches a plain
``GRPOTrainer`` step. Dead groups are replaced before the step's live
groups are stored and the buffer is built with ``min_entry_age=1``, so a
replayed row is never a copy of a fresh row of the same batch; every
sample record declares that floor and each drawn slot's entry version.

Versions, half-life and ``max_policy_age`` are counted in optimizer
steps (``trainer.state.global_step``).

More than one process
---------------------
Under ``accelerate`` with ``N > 1`` processes TRL hands each rank a
contiguous, rank-ordered slice of the generation batch, so a prompt group
may straddle ranks. Rank 0 owns the buffer and the single log writer; the
other ranks' ``ReservoirReplay`` never builds a buffer (``is_owner`` is
False and ``.buffer`` raises). Each rank computes behavior logprobs for its
own rows, the slices are gathered to rank 0, rank 0 runs the steps above on
the global batch, and the result is broadcast and sliced back, every rank
receiving its own rows padded to the global width and the recomputed
global ``num_items_in_batch``. ``ReservoirGRPOTrainer`` attaches the
accelerator at construction; the mixin attaches lazily on the first hook
call otherwise. See ``_trl_distributed``.

Scope and guards
----------------
Text-only, tested against TRL 1.13.0 (see ``_trl_compat``) in one process
and against a fake accelerator of two and four. Outputs carrying tool
masks, vLLM importance-sampling ratios or vision inputs raise
``NotImplementedError`` naming the key. One vLLM key is tolerated: TRL attaches
``sampling_per_token_logps`` (the logprobs vLLM reported while sampling)
to every vLLM batch, but with ``vllm_importance_sampling_correction=False``
and no ``off_policy_mask_threshold`` the loss never reads it. In that
case the hook drops the key from the batch (counted in
``stats["dropped_sampling_logprobs"]``) rather than refusing a batch TRL
would have trained on identically; replayed rows have no vLLM logprobs
to put there. With the correction on, the key is refused as before. ``ReservoirReplayCallback`` raises at
the end of the first optimizer step if the hook never ran, so a TRL
rename of the overridden method cannot silently disable replay.

What is attested: which rollouts were stored (each insert record carries
the content digest of prompt, completion and advantage, and the
``source`` tag given to ``ReservoirReplay``), which were drawn and with
what weight, and which were evicted. With ``manifest=`` the opening of
every digest is written next to the log, so ``python -m checker.verify
<log> --manifest <manifest>`` can confirm the log commits to exactly the
rows that were stored and ``python -m checker.transcript`` can report
exposure per example and per source. Generation, reward functions and
TRL's row shuffling are outside the log. No training-quality claim is
made for replay; see ``docs/nonclaims.md``.

``ReservoirReplay`` and ``ReservoirReplayMixin`` import nothing from TRL
and are tested with a fake trainer. ``ReservoirGRPOTrainer`` is built on
first access through ``build_trainer_class``, which imports TRL.
"""

from __future__ import annotations

import functools
import warnings
from pathlib import Path
from typing import Any, Final, Optional, Sequence, Union

import torch

from reservoir.durable_rollout import DurableRolloutBuffer
from reservoir.integrations._trl_compat import require_trl
from reservoir.integrations._trl_distributed import OWNER_RANK, owner_step, Communicator, communicator_for, mix_distributed
from reservoir.integrations._trl_guards import (  # re-exported: the refusals and the keys they name
    SAMPLING_LOGPROBS_KEY,
    SHARDED_STRATEGIES,
    UNSUPPORTED_OUTPUT_KEYS,
    check_batch,
    dropout_without_old_logprobs,
    drop_unused_sampling_logprobs,
    refuse_sharded,
)
from reservoir.integrations._trl_lifecycle import (
    RANK_ENV_VARS,
    bind_checkpoint,
    checkpoint_tag,
    resume_model_checkpoint,
    trainer_checkpoint_dir,
    env_rank,
    resume_from_checkpoint,
    trainer_checkpoint_steps,
)
from reservoir.integrations._replay_common import (  # re-exported: the adapters' shared priority and validators
    StoredAdvantagePriority,
    _validated_decline_cap,
    _validated_floor,
    _validated_gate,
    rescore_placed,
)
from reservoir.integrations._trl_rows import (  # tensor_digest and its keys re-exported
    PER_TOKEN_LOGPROB_KEYS,
    TENSOR_DIGEST_KEYS,
    RowConversion,
    rows_to_groups,
    tensor_digest,
    verify_written_rows,
    write_rows,
)
from reservoir.integrations._trl_staleness import (
    RowDecision,
    StalenessPolicy,
    decide,
    evictions_for,
    place,
    resolve_policy,
)
from reservoir.integrations._trl_telemetry import (
    StepTelemetry,
    log_metrics,
    sequence_log_ratios,
    summarize,
)
from reservoir.priorities import PriorityStrategy
from reservoir.rollout_attest import AttestTarget
from reservoir.rollout_buffer import RolloutBatch, RolloutBuffer

STAT_NAMES: Final[tuple[str, ...]] = (
    "hook_calls", "ingested_rows", "ingested_groups", "dead_groups", "skipped_rows",
    "clamped_logprobs", "replaced_rows", "logprob_forwards", "dropped_sampling_logprobs",
    "near_dead_groups", "declined_rows", "telemetry_forwards", "rescored_rows", "rescaled_rows",
    "age_floor_skips", "negligible_rows",
)

class ReservoirReplay:
    """The replay buffer handed to ``ReservoirGRPOTrainer(replay_buffer=...)``.

    Parameters
    ----------
    capacity, half_life, max_policy_age, beta, seed, attest, **kwargs
        Passed to ``RolloutBuffer`` (or ``DurableRolloutBuffer`` when
        ``directory`` is given). ``half_life`` and ``max_policy_age`` are
        in optimizer steps. ``beta=0.0`` disables importance weighting.
    priority : PriorityStrategy, optional
        Default ``StoredAdvantagePriority()``.
    directory : path, optional
        Make the buffer crash-atomic on disk.
    source : str, optional
        Provenance tag written on every stored row's insert record
        (``RolloutGroup.source``), for per-source exposure and quota
        reporting. One tag per adapter: TRL hands the hook token ids, not
        dataset rows, so a finer tag would have to come from the caller.
    manifest : path, optional
        Also write the opening of every stored row's content digest here.
        Requires ``attest``.
    telemetry : bool
        Measure replay health every step (default True). The log-ratio
        statistics cost one no-grad forward over the replayed rows.
    staleness_policy : StalenessPolicy or str, optional
        Which replayed rows to train on and with what weight: a
        ``StalenessPolicy`` or a preset name (``"conservative"``,
        ``"async"``, ``"off"``). Off by default. Its decisions are written
        to the telemetry record, so it needs ``telemetry=True``.
    max_log_ratio : float, optional
        The 0.6.0 drift gate, now the policy's last stage: decline
        replayed rows whose absolute sequence log-ratio exceeds this.
        Declines are never silent.
    max_declines_per_step : int, optional
        Raise if more rows than this would be declined in one step.
    min_entry_age : int
        Passed to the buffer (default 1; 0 or 1, since a floor of 2 would
        empty every sample): a replayed rollout was stored at
        least this many steps ago. The hook samples before it stores the
        step's live groups, so with 1 no replayed row duplicates a row of
        the batch; while an entry from an earlier generation of the same
        step is live the sample is empty and ``stats["age_floor_skips"]``
        counts it. 0 allows replaying such same-step entries.

    Attributes
    ----------
    buffer
        The underlying buffer; read it for ``size``, ``live_positions()``,
        ``attestation_log`` and so on. Built at construction on rank 0 (or
        in a single process); on any other rank it is never built and
        reading it raises, so no second process can open the log.
    is_owner : bool
        Whether this process holds the buffer: rank 0 once ``attach`` has
        run, before that whatever ``RANK``/``LOCAL_RANK`` in the
        environment say (absent means yes).
    stats : dict[str, int]
        Counters: ``hook_calls``, ``ingested_rows``, ``ingested_groups``,
        ``dead_groups``, ``skipped_rows`` (live rows with an empty
        completion), ``clamped_logprobs``, ``replaced_rows``,
        ``logprob_forwards`` (extra forwards run to obtain behavior
        logprobs).
    last_replay : RolloutBatch | None
        The sampled batch of the most recent hook call that replayed
        rows, with its buffer indices and importance weights.
    """

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
        min_entry_age: int = 1,
        **rollout_buffer_kwargs: Any,
    ) -> None:
        self._buffer_kwargs = dict(
            capacity=capacity,
            priority=priority if priority is not None else StoredAdvantagePriority(),
            half_life=half_life,
            max_policy_age=max_policy_age,
            beta=beta,
            seed=seed,
            attest=attest,
            manifest=manifest,
            min_entry_age=_validated_floor(min_entry_age),
            **rollout_buffer_kwargs,
        )
        self._directory = directory
        if manifest is not None and attest is None:
            raise ValueError("manifest requires attestation: pass attest=<path or log> as well")
        self._has_targets = attest is not None or manifest is not None or directory is not None
        self._buffer: Union[RolloutBuffer, DurableRolloutBuffer, None] = None
        self._owner: Optional[bool] = None
        self._comm: Optional[Communicator] = None
        self.model_checkpoint: Union[str, Path, None] = None
        """Set before ``train(resume_from_checkpoint=...)`` to the directory the model restarts from when it
        is not ``output_dir/checkpoint-N``; the buffer checkpoint's recorded digest is checked against it."""
        self._env_rank = env_rank()
        self.rank: Optional[int] = self._env_rank
        # In-memory state is harmless to build now and discard later; a log, manifest or
        # directory waits until the rank is known (the launcher's RANK, or attach).
        if self.is_owner and not self._has_targets:
            self._buffer = self._build_buffer()
        self.source = source
        self.beta = float(beta)
        self.telemetry = bool(telemetry)
        self.policy: StalenessPolicy = resolve_policy(
            staleness_policy, _validated_gate(max_log_ratio), _validated_decline_cap(max_declines_per_step), self.telemetry,
        )
        self._pending_rewards: Optional[tuple[tuple[str, ...], torch.Tensor]] = None  # from _calculate_rewards
        self.last_telemetry: Optional[StepTelemetry] = None
        self.stats: dict[str, int] = {name: 0 for name in STAT_NAMES}
        self.last_replay: Optional[RolloutBatch] = None
        self._dropout_warned = False

    @property
    def max_log_ratio(self) -> Optional[float]:
        """The legacy drift gate threshold: stage 4 of ``policy``."""
        return self.policy.max_log_ratio

    @property
    def max_declines_per_step(self) -> Optional[int]:
        return self.policy.max_declines_per_step

    # -- ownership -----------------------------------------------------------

    def _build_buffer(self) -> Union[RolloutBuffer, DurableRolloutBuffer]:
        if self._directory is None:
            return RolloutBuffer(**self._buffer_kwargs)
        return DurableRolloutBuffer(self._directory, **self._buffer_kwargs)

    @property
    def is_owner(self) -> bool:
        if self._owner is not None:
            return self._owner
        return self._env_rank in (None, OWNER_RANK)

    @property
    def buffer(self) -> Union[RolloutBuffer, DurableRolloutBuffer]:
        if self._buffer is not None:
            return self._buffer
        if self.is_owner:
            self._buffer = self._build_buffer()
            return self._buffer
        if self._owner is False:
            raise RuntimeError(
                f"rank {self.rank} does not own the replay buffer; only rank {OWNER_RANK} holds it "
                "and writes the log. Read buffer state on rank 0."
            )
        raise RuntimeError(
            f"RANK/LOCAL_RANK={self._env_rank} in the environment: this process is not rank {OWNER_RANK}, "
            "so it holds no replay buffer; the buffer lives on rank 0"
        )

    def attach(self, accelerator: Any) -> None:
        """Bind this replay to the launcher's process group; decides which rank owns the buffer.

        ``ReservoirGRPOTrainer`` calls this at construction; ``mix`` calls
        it on the first hook call otherwise. With more than one process
        this is a collective: every rank reports whether it already holds a
        buffer with a log, manifest or directory, and if any rank but 0
        does, every rank raises (that buffer may have touched a file rank 0
        owns) and nothing is attached. A non-owner's in-memory buffer is
        closed and dropped.
        """
        comm = communicator_for(accelerator)
        rank = int(comm.process_index)
        refuse_sharded(accelerator, comm.num_processes)
        if comm.num_processes > 1:
            self._refuse_foreign_file_state(comm, rank)
        self._comm = comm
        self.rank = rank
        self._owner = rank == OWNER_RANK

        def build() -> None:
            if self._buffer is None:
                self._buffer = self._build_buffer()
        # Building the buffer can fail (a bad directory, a mismatched snapshot); every rank
        # hears about it rather than waiting for rank 0 in the first gather.
        owner_step(comm, self._owner, build)
        if not self._owner and self._buffer is not None:
            self._buffer.close()
            self._buffer = None

    def _refuse_foreign_file_state(self, comm: Communicator, rank: int) -> None:
        """Collective check that only rank 0 has opened a log, manifest or directory."""
        mine = self._buffer is not None and self._has_targets
        reports = comm.gather_object({"rank": rank, "file_state": mine})
        offenders = sorted(r["rank"] for r in reports if r["file_state"] and r["rank"] != OWNER_RANK)
        if offenders:
            if mine and rank in offenders:
                self._buffer.close()
                self._buffer = None
            raise RuntimeError(
                f"rank(s) {offenders} built a replay buffer with a log, manifest or directory before the "
                f"process group was attached; only rank {OWNER_RANK} may own one. Construct "
                "ReservoirReplay under the launcher (RANK set) or attach the accelerator before "
                "using the buffer"
            )

    # -- the hook ------------------------------------------------------------

    def mix(self, output: dict, trainer: Any) -> dict:
        """Store the live rows of ``output`` and replace its dead rows with replayed ones.

        ``trainer`` is the ``GRPOTrainer`` (or a stand-in exposing the
        same members). Returns ``output`` itself when nothing was
        replaced, otherwise a new dict; TRL's tensors are never modified.
        With more than one process the work is routed through
        ``_trl_distributed.mix_distributed`` and ``output`` is this rank's
        slice of the generation batch.
        """
        if self._comm is None and int(trainer.accelerator.num_processes) > 1:
            self.attach(trainer.accelerator)
        if self._comm is not None and self._comm.num_processes > 1:
            return mix_distributed(self, output, trainer, self._comm)
        return self.mix_local(output, trainer)

    def prepare_local(self, output: dict, trainer: Any) -> dict:
        """Drop what the loss would not read and refuse what the adapter cannot handle.

        Runs on every rank before any collective, so it must not touch the
        buffer; the version check is the owner's, in ``mix_local``.
        """
        kept = drop_unused_sampling_logprobs(output, trainer)
        if kept is not output:
            self.stats["dropped_sampling_logprobs"] += 1
        check_batch(kept, trainer, self.beta)
        if not self._dropout_warned and dropout_without_old_logprobs(kept, trainer):
            warnings.warn(
                "model dropout is active without TRL old_per_token_logps; replay can change fresh-row loss ratios",
                RuntimeWarning, stacklevel=2,
            )
            self._dropout_warned = True
        return kept

    def mix_local(self, output: dict, trainer: Any) -> dict:
        """The single-process hook on a batch with whole prompt groups.

        Order matters: the buffer is advanced to ``global_step`` (evicting
        entries older than ``max_policy_age``), dead rows are replaced from
        whatever is still live, and only then are this step's live groups
        stored, so a replayed row is never one the batch already holds.
        If the buffer raises while groups are being stored, the groups
        stored before it stay stored and the counters are not updated.
        """
        step = int(trainer.state.global_step)
        output = self.prepare_local(output, trainer)
        self._check_version(step)
        self.last_replay = None
        self.stats["hook_calls"] += 1
        logprobs = output.get("old_per_token_logps")
        if logprobs is None:
            logprobs = self.behavior_logprobs(output, trainer)
        conversion = self._convert(output, trainer, step, logprobs)
        if step > self.buffer.current_version:
            self.buffer.advance(step)
        new = None
        if conversion.dead_rows and self.buffer.size and self.buffer.total:
            new = self._replay(output, trainer, step, logprobs, conversion)
        if new is None:
            self._telemetry(trainer, step, None, [], [], output["advantages"].size(0), conversion, 0)
        self._ingest(conversion, step)
        return output if new is None else new

    def _check_version(self, step: int) -> None:
        """Refuse a step behind the buffer (owner only: it reads the buffer)."""
        if step < self.buffer.current_version:
            raise ValueError(
                f"global_step {step} is below the buffer's current version "
                f"{self.buffer.current_version}; versions only move forward. A resumed run "
                "must reopen the buffer at or behind the step it resumes from"
            )

    def behavior_logprobs(self, output: dict, trainer: Any) -> torch.Tensor:
        """One no-grad forward to get the logprobs TRL did not compute (for this rank's rows)."""
        input_ids = torch.cat([output["prompt_ids"], output["completion_ids"]], dim=1)
        attention_mask = torch.cat([output["prompt_mask"], output["completion_mask"]], dim=1)
        logits_to_keep = output["completion_ids"].size(1)
        with torch.no_grad():
            logprobs, _, _ = trainer._get_per_token_logps_and_entropies(
                trainer.model,
                input_ids,
                attention_mask,
                logits_to_keep,
                batch_size=trainer.args.per_device_train_batch_size,
            )
        self.stats["logprob_forwards"] += 1
        return logprobs

    def note_rewards_per_func(self, rewards_per_func: Any, reward_names: Any) -> None:
        """Remember the per-reward-function values TRL computed for the batch about to be ingested.

        ``ReservoirReplayMixin`` calls this from ``_calculate_rewards``; TRL
        gathers the tensor across processes in rank order, which is the
        order the owner's global batch is assembled in. Anything that is
        not a 2-D tensor with matching names is ignored, so a trainer that
        computes rewards differently simply records no provenance.
        """
        if (isinstance(rewards_per_func, torch.Tensor) and rewards_per_func.dim() == 2 and reward_names
                and len(tuple(reward_names)) == rewards_per_func.size(1)):
            self._pending_rewards = (tuple(str(n) for n in reward_names), rewards_per_func.detach())
        else:
            self._pending_rewards = None

    def _take_rewards_per_func(self, batch_rows: int) -> tuple[Optional[tuple[str, ...]], Optional[torch.Tensor]]:
        """The pending per-function rewards if they describe this batch, else none; always consumed."""
        pending, self._pending_rewards = self._pending_rewards, None
        if pending is None:
            return None, None
        names, values = pending
        if values.size(0) != batch_rows:
            raise ValueError(
                f"_calculate_rewards produced {values.size(0)} rows of rewards but the batch has {batch_rows}; "
                "reward provenance cannot be attributed to rows"
            )
        return names, values.to("cpu", torch.float64)

    def _convert(self, output: dict, trainer: Any, step: int, logprobs: torch.Tensor) -> RowConversion:
        """The batch as live groups to store and dead rows to replace."""
        names, values = self._take_rewards_per_func(int(output["advantages"].size(0)))
        aggregation = getattr(trainer, "multi_objective_aggregation", "sum_then_normalize")
        return rows_to_groups(
            output, num_generations=trainer.num_generations, step=step, logprobs=logprobs,
            reward_names=names, rewards_per_func=values,
            equal_rewards_are_dead=aggregation == "sum_then_normalize",
        )

    def _ingest(self, conversion: RowConversion, step: int) -> None:
        """Add every live group of the step to the buffer, after its dead rows were replaced."""
        for group in conversion.groups:
            self.buffer.add_group(group.prompt_id, step, group.rollouts, source=self.source)
        self.stats["ingested_rows"] += sum(len(group.rollouts) for group in conversion.groups)
        self.stats["ingested_groups"] += len(conversion.groups)
        self.stats["dead_groups"] += conversion.dead_groups
        self.stats["near_dead_groups"] += conversion.near_dead_groups
        self.stats["skipped_rows"] += conversion.skipped_rows
        self.stats["clamped_logprobs"] += conversion.clamped_logprobs
        self.stats["negligible_rows"] += conversion.negligible_rows

    def _replay(
        self, output: dict, trainer: Any, step: int, logprobs: torch.Tensor, conversion: RowConversion
    ) -> Optional[dict]:
        """Sample one rollout per dead row, write them in, gate, verify, witness, measure.

        Returns None when the sample is empty (a live entry is younger than
        the buffer's ``min_entry_age``). ``write_rows`` recomputes ``num_items_in_batch`` from the final
        mask; this runs on the whole generation batch (rank 0 assembles it
        under more than one process), so that is the value TRL would gather.
        """
        dead_rows = conversion.dead_rows
        batch = self.buffer.sample(len(dead_rows), current_version=step)
        if not batch.indices:
            self.stats["age_floor_skips"] += 1
            return None
        rewards = [r.reward for r in batch.rollouts]
        unit = place(rewards, batch.is_weights, ()).weighted           # importance-weighted, before the policy
        with_logprobs = output if "old_per_token_logps" in output else {**output, "old_per_token_logps": logprobs}
        pad = trainer._tokenizer.pad_token_id
        provisional = write_rows(with_logprobs, dead_rows, batch.rollouts, unit, pad)
        ratios = self._log_ratios(provisional, list(dead_rows), trainer)
        group_size = int(trainer.num_generations)
        decisions = self._decide(batch, ratios, dead_rows, [r // group_size for r in dead_rows])
        kept, declined, weighted, rescaled = place(rewards, batch.is_weights, decisions)
        if declined or rescaled:
            # Declined draws leave their dead rows dead; rescaled draws carry the capped advantage. The batch
            # goes back with the behavior logprobs attached and the witness digest below is over what is returned.
            new = write_rows(with_logprobs, [dead_rows[k] for k in kept], [batch.rollouts[k] for k in kept],
                             weighted, pad) if kept else with_logprobs
        else:
            new = provisional
        rows = [dead_rows[k] for k in kept]
        verify_written_rows(new, rows, [batch.rollouts[k] for k in kept], weighted)
        self.buffer.witness_batch(batch, step=step, batch_rows=new["advantages"].size(0), rows=rows,
                                  tensor_digest=tensor_digest(new), declined=declined)
        # Telemetry describes the sampled batch and is recomputed by the checker from the live
        # slots, so it is written before any declined entry is evicted.
        self._telemetry(trainer, step, batch, ratios, kept, new["advantages"].size(0), conversion, len(declined),
                        decisions, group_size)
        for slot, reason in evictions_for(batch.indices, decisions):
            self.buffer.evict(slot, reason)
        self._rescore(batch, kept, weighted, ratios, step)
        self.last_replay = batch
        self.stats["replaced_rows"] += len(kept)
        self.stats["declined_rows"] += len(declined)
        self.stats["rescaled_rows"] += rescaled
        return new

    def _decide(self, batch: RolloutBatch, ratios: list[float], rows: Sequence[int],
                groups: Sequence[int]) -> tuple[RowDecision, ...]:
        """The staleness policy's decision for every draw; ``()`` when the policy is off."""
        if not self.policy.active:
            return ()
        ages = [self.buffer.current_version - v for v in batch.model_versions]
        return decide(self.policy, ratios=ratios, is_weights=batch.is_weights, ages=ages, rows=list(rows),
                      groups=list(groups))

    def _rescore(self, batch: RolloutBatch, kept: list[int], weighted: list[float], ratios: list[float],
                 step: int) -> None:
        """New priorities for the placed rollouts; see ``rescore_placed``."""
        self.stats["rescored_rows"] += rescore_placed(self.buffer, batch, kept, weighted, ratios, step)

    def _log_ratios(self, output: dict, rows: list[int], trainer: Any) -> list[float]:
        """Sequence log-ratios of the replayed rows, when telemetry or the policy needs them."""
        if not rows or (not self.telemetry and not self.policy.active):
            return []
        self.stats["telemetry_forwards"] += 1
        return sequence_log_ratios(output, rows, trainer)

    def _telemetry(self, trainer: Any, step: int, batch: Optional[RolloutBatch], ratios: list[float],
                   kept: list[int], batch_rows: int, conversion: RowConversion, declined: int,
                   decisions: tuple[RowDecision, ...] = (), group_size: Optional[int] = None) -> None:
        """Summarise the step, log it to the trainer and write the telemetry record.

        For a replayed step the record carries every draw's log-ratio and,
        with a policy on, the policy and its decisions, so the checker can
        replay them.
        """
        if not self.telemetry:
            return
        point = summarize(batch, self.buffer.current_version, ratios, kept, batch_rows,
                          conversion.dead_groups, conversion.near_dead_groups, declined, decisions, self.policy)
        self.last_telemetry = point
        log_metrics(trainer, point)
        self.buffer.record_telemetry(
            step, point.counts(), batch, point.reported(),
            log_ratios=list(ratios) if batch is not None else None,
            policy=self.policy.to_record(group_size) if decisions else None,
            decisions=[d.to_record() for d in decisions] if decisions else None,
        )

    def close(self) -> None:
        """Close the attestation file, if one was opened; nothing to do off the owner rank."""
        if self._buffer is not None:
            self._buffer.close()

    def __repr__(self) -> str:
        source = f", source={self.source!r}" if self.source is not None else ""
        inner = repr(self._buffer) if self._buffer is not None else f"<no buffer on rank {self.rank}>"
        return f"ReservoirReplay({inner}, beta={self.beta}{source})"


class ReservoirReplayMixin:
    """Overrides ``_generate_and_score_completions`` to route its output through ``replay_buffer``.

    Place it before ``GRPOTrainer`` in the base list. The class using it
    must set ``self.replay_buffer`` to a ``ReservoirReplay``.
    """

    replay_buffer: ReservoirReplay

    def _calculate_rewards(self, *args, **kwargs):  # type: ignore[override]
        rewards_per_func = super()._calculate_rewards(*args, **kwargs)  # type: ignore[misc]
        self.replay_buffer.note_rewards_per_func(rewards_per_func, getattr(self, "reward_func_names", None))
        return rewards_per_func

    def _generate_and_score_completions(self, inputs):  # type: ignore[override]
        output = super()._generate_and_score_completions(inputs)  # type: ignore[misc]
        if not self.model.training:  # type: ignore[attr-defined]
            self.replay_buffer.note_rewards_per_func(None, None)
            return output
        return self.replay_buffer.mix(output, self)


def assert_hook_ran(replay: ReservoirReplay, global_step: int) -> None:
    """Raise if training has completed a step without the replay hook ever running."""
    if global_step >= 1 and replay.stats["hook_calls"] == 0:
        raise RuntimeError(
            "ReservoirGRPOTrainer completed an optimizer step but its replay hook never ran; "
            "the installed TRL no longer calls _generate_and_score_completions the way "
            "this adapter expects"
        )


@functools.lru_cache(maxsize=None)
def build_trainer_class() -> type:
    """Import TRL (checked by ``require_trl``) and build ``ReservoirGRPOTrainer``."""
    support = require_trl()

    class ReservoirReplayCallback(support.trainer_callback):  # type: ignore[misc,valid-type]
        """Fails the run early if the replay hook is not being called; binds a durable
        buffer to the trainer's checkpoints (see ``bind_checkpoint`` and ``resume_from_checkpoint``)."""

        def __init__(self, replay: ReservoirReplay) -> None:
            self.replay = replay

        def on_step_end(self, args, state, control, **kwargs):
            assert_hook_ran(self.replay, state.global_step)
            return control

        def on_save(self, args, state, control, **kwargs):
            bind_checkpoint(self.replay, state.global_step, getattr(args, "output_dir", None))
            return control

        def on_train_begin(self, args, state, control, **kwargs):
            model_checkpoint = resume_model_checkpoint(self.replay, args, state.global_step)
            resume_from_checkpoint(self.replay, state.global_step, model_checkpoint)
            return control

    class ReservoirGRPOTrainer(ReservoirReplayMixin, support.grpo_trainer):  # type: ignore[misc,valid-type]
        """``GRPOTrainer`` with Reservoir replay of dead groups. See the module docstring."""

        replay_callback_class = ReservoirReplayCallback

        def __init__(self, *args: Any, replay_buffer: ReservoirReplay, **kwargs: Any) -> None:
            if not isinstance(replay_buffer, ReservoirReplay):
                raise TypeError(
                    f"replay_buffer must be a ReservoirReplay, got {type(replay_buffer).__name__}"
                )
            super().__init__(*args, **kwargs)
            check_batch({}, self, replay_buffer.beta)
            self.replay_buffer = replay_buffer
            accelerator = getattr(self, "accelerator", None)
            if accelerator is not None:
                replay_buffer.attach(accelerator)
            self.add_callback(ReservoirReplayCallback(replay_buffer))

    ReservoirGRPOTrainer.Callback = ReservoirReplayCallback  # type: ignore[attr-defined]
    # Module-level qualnames let pickle find both classes through __getattr__.
    ReservoirReplayCallback.__qualname__ = "ReservoirGRPOTrainer.Callback"
    ReservoirGRPOTrainer.__qualname__ = "ReservoirGRPOTrainer"
    for cls in (ReservoirReplayCallback, ReservoirGRPOTrainer):
        cls.__module__ = __name__
    return ReservoirGRPOTrainer


def __getattr__(name: str) -> Any:
    """Build ``ReservoirGRPOTrainer`` on first access so importing this module needs no TRL."""
    if name == "ReservoirGRPOTrainer":
        return build_trainer_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | {"ReservoirGRPOTrainer"})


# ``ReservoirGRPOTrainer`` is deliberately absent: a star import must not
# require TRL. Import it by name.
__all__ = [
    "RANK_ENV_VARS",
    "ReservoirReplay",
    "ReservoirReplayMixin",
    "StalenessPolicy",
    "StoredAdvantagePriority",
    "PER_TOKEN_LOGPROB_KEYS",
    "TENSOR_DIGEST_KEYS",
    "tensor_digest",
    "UNSUPPORTED_OUTPUT_KEYS",
    "assert_hook_ran",
    "bind_checkpoint",
    "trainer_checkpoint_dir",
    "build_trainer_class",
    "checkpoint_tag",
    "env_rank",
    "resume_from_checkpoint",
    "trainer_checkpoint_steps",
]

from reservoir.integrations import _trl_presets  # noqa: E402,F401
