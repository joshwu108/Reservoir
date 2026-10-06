"""Reservoir-backed replay for verl's ``RayPPOTrainer`` under GRPO.

Usage::

    from reservoir.integrations.verl import ReservoirRayPPOTrainer, ReservoirReplay

    trainer = ReservoirRayPPOTrainer(
        config=config, tokenizer=tokenizer, role_worker_mapping=..., resource_pool_manager=...,
        reward_fn=..., val_reward_fn=..., train_dataset=..., val_dataset=...,
        replay_buffer=ReservoirReplay(capacity=50_000, half_life=4, max_policy_age=16,
                                      seed=0, attest="run-01/attest.jsonl"),
    )
    trainer.init_workers()
    trainer.fit()

What it does
------------
The same thing ``reservoir.integrations.trl`` does for TRL: a prompt
whose ``n`` responses all got the same reward is a "dead" group whose
GRPO advantages are exactly zero, so it contributes no gradient.
``ReservoirRayPPOTrainer`` keeps the live rollouts in a ``RolloutBuffer``
and, whenever a training step produces dead groups, fills their rows
with rollouts replayed from the buffer: sampled by exact, age-decayed
priority with keyed deterministic draws, carrying their behavior logprobs
so the actor loss applies a real off-policy ratio, and with every
insertion, draw and eviction written to the attestation log that
``reservoir-verify`` checks.

Where it plugs in
-----------------
``RayPPOTrainer.fit`` runs on the Ray driver. Per step it generates, scores,
computes ``old_log_probs`` and the advantages on the whole batch (a
``DataProto`` on the driver), then calls
``self.actor_rollout_wg.update_actor(batch)``. ``ReservoirReplayMixin``
overrides ``init_workers`` to wrap that worker group in ``ReplayWorkerGroup``,
a proxy that forwards every call unchanged except ``update_actor``, whose
batch it first hands to ``ReservoirReplay.mix``. No verl method body is
copied. Because the driver holds the whole batch, there is no rank
ownership question: the driver owns the buffer and the log.

The batch keys the adapter reads and rewrites are listed in
``_verl_rows``; rows of one prompt are found by ``non_tensor_batch["uid"]``
(``balance_batch`` reorders rows, so groups are not contiguous). The mixed
``DataProto`` carries the rewritten tensors, ``uid`` (a replayed row takes
the uid of the group it came from) and the original ``meta_info`` with
``global_token_num`` recomputed from the final ``attention_mask``. The
other ``non_tensor_batch`` columns (``data_source``, ``reward_model``,
``extra_info``, ...) describe the prompts that were generated and would be
stale for a replayed row, so they are not carried; the stock FSDP and
Megatron actors do not read them. ``fit`` keeps its own reference to the
generated batch, so verl's data metrics (``critic/score/*``,
``response_length/*``) describe the generated batch while the actor
metrics describe the replayed one.

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
   ``compute_log_prob`` call over the replayed rows for the log-ratios)
   and logged under ``reservoir/*`` through the actor's metrics, and the
   optional drift gate declines rows whose log-ratio is too large.

A step with no dead groups, or an empty buffer, hands ``update_actor`` the
very ``DataProto`` ``fit`` built. Versions, half-life and
``max_policy_age`` are counted in ``trainer.global_steps``.

Checkpoints: ``_save_checkpoint`` also snapshots a durable buffer under
``step-<global_steps>`` and prunes buffer checkpoints to the
``global_step_N`` directories the trainer kept; ``_load_checkpoint`` at a
non-zero step rewinds the buffer to that snapshot, so a resumed run
continues the draw counter and the attestation chain from the checkpoint.

Scope and guards
----------------
``algorithm.adv_estimator=grpo`` only; text models with 2-D position ids;
single-turn prefix response masks; no critic (``values``), no rollout
importance-sampling logprobs (``rollout_log_probs``), no KL-in-reward, no
multimodal inputs. Each is refused by name before anything is stored.
Tested against the verl release in ``_verl_compat`` with a fake trainer
and one real run on one GPU. The hook-ran check raises on the second
training step's generation, at every save and at the end of ``fit`` if
``update_actor`` never passed through the proxy, so a verl change that
bypasses the worker group cannot silently disable replay. No
training-quality claim is made for replay; see ``docs/nonclaims.md``.

``ReservoirReplay``, ``ReservoirReplayMixin`` and ``ReplayWorkerGroup``
import nothing from verl and are tested with a fake trainer.
``ReservoirRayPPOTrainer`` is built on first access through
``build_trainer_class``, which imports verl.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path
from typing import Any, Final, Optional, Sequence, Union

import numpy as np

from reservoir.durable_rollout import DurableRolloutBuffer
from reservoir.integrations._trl_lifecycle import checkpoint_tag, resume_from_checkpoint
from reservoir.integrations._trl_telemetry import StepTelemetry, choose_declines, summarize
from reservoir.integrations._verl_compat import require_verl
from reservoir.integrations._verl_rows import (
    REF_KEY,
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

UNSUPPORTED_BATCH_KEYS: Final[tuple[str, ...]] = ("values", "rollout_log_probs", "rollout_is_weights")
"""Batch tensors that would need per-row replacement logic this adapter does not have."""

UNSUPPORTED_NON_TENSOR_KEYS: Final[tuple[str, ...]] = ("multi_modal_inputs", "multi_modal_data")
"""Vision inputs; a replayed row has none to put there."""

SUPPORTED_ADV_ESTIMATORS: Final[tuple[str, ...]] = ("grpo",)

NON_HOMOGENEOUS_LOSS_MODES: Final[tuple[str, ...]] = ("clip_cov", "kl_cov")
"""Loss modes whose token selection depends on the advantage scale; refused with ``beta > 0``."""

STAT_NAMES: Final[tuple[str, ...]] = (
    "hook_calls", "ingested_rows", "ingested_groups", "dead_groups", "skipped_rows", "clamped_logprobs",
    "replaced_rows", "near_dead_groups", "declined_rows", "telemetry_forwards", "rescored_rows",
)

CHECKPOINT_DIR_PREFIX: Final[str] = "global_step_"
"""verl writes ``<default_local_dir>/global_step_<n>`` per saved step."""

LOGPROB_FORWARD_KEYS: Final[tuple[str, ...]] = (
    "input_ids", "attention_mask", "position_ids", "prompts", "responses", "response_mask",
)
"""What ``compute_log_prob`` needs from the replayed rows."""


def config_value(config: Any, path: str, default: Any = None) -> Any:
    """``config.a.b.c`` through attribute or item access (OmegaConf or namespaces); ``default`` if absent."""
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


class ReservoirReplay:
    """The replay buffer handed to ``ReservoirRayPPOTrainer(replay_buffer=...)``.

    Parameters mirror ``reservoir.integrations.trl.ReservoirReplay``:
    ``capacity``, ``half_life`` and ``max_policy_age`` (in training
    steps), ``beta`` (``0.0`` disables importance weighting), ``seed``,
    ``attest``, ``directory`` (crash-atomic buffer on disk), ``source``
    (provenance tag on every insert), ``manifest`` (requires ``attest``),
    ``telemetry`` (replay health every step; the log-ratios cost one
    ``compute_log_prob`` call over the replayed rows), ``max_log_ratio``
    and ``max_declines_per_step`` (the drift gate, off by default).

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
        The most recent step's measurements.
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
        self.max_log_ratio = _validated_gate(max_log_ratio)
        self.max_declines_per_step = max_declines_per_step
        self.last_telemetry: Optional[StepTelemetry] = None
        self.last_replay: Optional[RolloutBatch] = None
        self.stats: dict[str, int] = {name: 0 for name in STAT_NAMES}

    # -- the hook ------------------------------------------------------------

    def mix(self, data: Any, trainer: Any) -> Any:
        """Store the live rows of ``data`` and replace its dead rows with replayed ones.

        ``data`` is the ``DataProto`` ``fit`` is about to hand to
        ``update_actor``; ``trainer`` is the ``RayPPOTrainer`` (or a stand-in
        exposing ``global_steps``, ``config``, ``tokenizer`` and
        ``actor_rollout_wg.compute_log_prob``). Returns ``data`` itself when
        nothing was replaced, otherwise a new ``DataProto``; the tensors of
        ``data`` are never modified. Order matters: groups are stored, the
        buffer is advanced to ``global_steps`` (evicting entries older than
        ``max_policy_age``), and only then are dead rows replaced from
        whatever is still live.
        """
        step = int(trainer.global_steps)
        tensors = {key: data.batch[key] for key in data.batch.keys()}
        self._check_batch(data, trainer)
        self._check_version(step)
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
        if "uid" not in data.non_tensor_batch:
            raise ValueError("the batch has no non_tensor_batch['uid']; rows cannot be grouped by prompt")
        for key in UNSUPPORTED_BATCH_KEYS:
            if key in data.batch.keys():
                raise NotImplementedError(
                    f"ReservoirReplay cannot replay batches with {key!r} (a critic, rollout importance "
                    "sampling and rollout logprobs are unsupported)"
                )
        for key in UNSUPPORTED_NON_TENSOR_KEYS:
            if key in data.non_tensor_batch:
                raise NotImplementedError(f"ReservoirReplay cannot replay batches with {key!r} (vision inputs are unsupported)")
        estimator = config_value(trainer.config, "algorithm.adv_estimator")
        if estimator not in SUPPORTED_ADV_ESTIMATORS:
            raise NotImplementedError(
                f"algorithm.adv_estimator={estimator!r}; the adapter's dead-group criterion is defined for "
                f"{', '.join(SUPPORTED_ADV_ESTIMATORS)} only"
            )
        loss_mode = config_value(trainer.config, "actor_rollout_ref.actor.policy_loss.loss_mode", "vanilla")
        if self.beta > 0.0 and loss_mode in NON_HOMOGENEOUS_LOSS_MODES:
            raise ValueError(
                f"policy_loss.loss_mode {loss_mode!r} selects tokens by covariance with the advantage, so "
                "importance weights cannot be folded into it; use ReservoirReplay(beta=0.0) with it"
            )
        if not hasattr(type(data), "from_dict"):
            raise TypeError(f"{type(data).__name__} has no from_dict; is this a verl DataProto?")

    def _replay(self, data: Any, tensors: dict, uids: list[str], trainer: Any, step: int, conversion) -> Any:
        """Sample one rollout per dead row, write them in, gate, verify, witness, measure."""
        dead_rows = list(conversion.dead_rows)
        batch = self.buffer.sample(len(dead_rows), current_version=step)
        weighted = [r.reward * float(w) for r, w in zip(batch.rollouts, batch.is_weights)]
        pad = _pad_token_id(trainer)
        provisional = write_rows(tensors, dead_rows, batch.rollouts, weighted, pad)
        ratios = self._log_ratios(data, provisional, dead_rows, trainer)
        declined = choose_declines(ratios, self.max_log_ratio, self.max_declines_per_step)
        declined_set = set(declined)
        kept = [k for k in range(len(dead_rows)) if k not in declined_set]
        if declined:
            # With every draw declined the dead rows stay dead; the witness digest is over what is returned.
            new = (write_rows(tensors, [dead_rows[k] for k in kept], [batch.rollouts[k] for k in kept],
                              [weighted[k] for k in kept], pad) if kept else dict(tensors))
        else:
            new = provisional
        rows = [dead_rows[k] for k in kept]
        verify_written_rows(new, rows, [batch.rollouts[k] for k in kept], [weighted[k] for k in kept])
        self.buffer.witness_batch(batch, step=step, batch_rows=len(uids), rows=rows,
                                  tensor_digest=tensor_digest(new), declined=declined)
        # Telemetry describes the sampled batch and is recomputed by the checker from the live
        # slots, so it is written before any declined entry is evicted.
        self._telemetry(step, batch, ratios, kept, len(uids), conversion, len(declined))
        for slot in sorted({batch.indices[k] for k in declined} - {batch.indices[k] for k in kept}):
            self.buffer.evict(slot, "drift")
        self._rescore(batch, kept, weighted, ratios, step)
        self.last_replay = batch
        self.stats["replaced_rows"] += len(kept)
        self.stats["declined_rows"] += len(declined)
        new_uids = list(uids)
        for k in kept:
            new_uids[dead_rows[k]] = str(batch.rollouts[k].metadata.get("uid", batch.groups[k].prompt_id))
        meta_info = dict(data.meta_info)
        meta_info["global_token_num"] = global_token_num(new["attention_mask"])
        return type(data).from_dict(
            tensors=new, non_tensors={"uid": np.array(new_uids, dtype=object)}, meta_info=meta_info,
        )

    def _log_ratios(self, data: Any, tensors: dict, rows: list[int], trainer: Any) -> list[float]:
        """Sequence log-ratios ``sum(current - behavior)`` of the replayed rows, via ``compute_log_prob``."""
        if not rows or (not self.telemetry and self.max_log_ratio is None):
            return []
        import torch

        idx = torch.tensor(rows)
        subset = type(data).from_dict(
            tensors={key: tensors[key][idx] for key in LOGPROB_FORWARD_KEYS}, meta_info=dict(data.meta_info),
        )
        self.stats["telemetry_forwards"] += 1
        current = trainer.actor_rollout_wg.compute_log_prob(subset).batch["old_log_probs"]
        behavior = tensors["old_log_probs"][idx]
        if tuple(current.shape) != tuple(behavior.shape):
            raise ValueError(
                f"compute_log_prob returned old_log_probs of shape {tuple(current.shape)}, expected {tuple(behavior.shape)}"
            )
        mask = tensors["response_mask"][idx].to(behavior.dtype)
        ratios = ((current.to(behavior.dtype).to(behavior.device) - behavior) * mask).sum(dim=1)
        return [float(x) for x in ratios.detach().cpu().tolist()]

    def _rescore(self, batch: RolloutBatch, kept: list[int], weighted: list[float], ratios: list[float], step: int) -> None:
        """Ask the strategy for new priorities of the placed rollouts; a slot drawn twice is rescored once."""
        indices: list[int] = []
        scores: list[float] = []
        seen: set[int] = set()
        for k in kept:
            slot = batch.indices[k]
            if slot in seen:
                continue
            seen.add(slot)
            ratio = ratios[k] if ratios else None
            signal = ReplaySignal(
                advantage=weighted[k], is_weight=float(batch.is_weights[k]),
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
                   batch_rows: int, conversion, declined: int) -> None:
        if not self.telemetry:
            self.last_telemetry = None
            return
        point = summarize(batch, self.buffer.current_version, ratios, kept, batch_rows,
                          conversion.dead_groups, conversion.near_dead_groups, declined)
        self.last_telemetry = point
        self.buffer.record_telemetry(step, point.counts(), batch, point.reported())

    def close(self) -> None:
        """Close the attestation file, if one was opened."""
        self.buffer.close()

    def __repr__(self) -> str:
        source = f", source={self.source!r}" if self.source is not None else ""
        return f"ReservoirReplay({self.buffer!r}, beta={self.beta}{source})"


def _pad_token_id(trainer: Any) -> int:
    pad = getattr(getattr(trainer, "tokenizer", None), "pad_token_id", None)
    if pad is None:
        raise ValueError("trainer.tokenizer.pad_token_id is None; replayed rows cannot be padded into place")
    return int(pad)


# ---------------------------------------------------------------------------
# The seam: a proxy over the actor worker group
# ---------------------------------------------------------------------------

def assert_hook_ran(replay: ReservoirReplay, global_steps: int) -> None:
    """Raise if a training step has completed without the replay hook ever running."""
    if int(global_steps) >= 2 and replay.stats["hook_calls"] == 0:
        raise RuntimeError(
            "ReservoirRayPPOTrainer completed a training step but its replay hook never ran; the "
            "installed verl no longer calls actor_rollout_wg.update_actor the way this adapter expects"
        )


class ReplayWorkerGroup:
    """Forwards every call to the actor worker group; ``update_actor`` goes through the replay first.

    ``fit`` holds ``self.actor_rollout_wg`` and calls ``generate_sequences``,
    ``compute_log_prob``, ``update_actor`` and the checkpoint methods on
    it. Wrapping the object, rather than copying ``fit``, keeps every other
    call untouched. The telemetry of the step is appended to the actor's
    returned metrics (``meta_info["metrics"]``, lists per key that ``fit``
    reduces and logs), so ``reservoir/*`` shows up next to ``actor/*``.
    """

    def __init__(self, inner: Any, replay: ReservoirReplay, trainer: Any) -> None:
        self._inner = inner
        self._replay = replay
        self._trainer = trainer

    @property
    def wrapped(self) -> Any:
        return self._inner

    def update_actor(self, data: Any, *args: Any, **kwargs: Any) -> Any:
        mixed = self._replay.mix(data, self._trainer)
        output = self._inner.update_actor(mixed, *args, **kwargs)
        self._append_metrics(output)
        return output

    def generate_sequences(self, *args: Any, **kwargs: Any) -> Any:
        assert_hook_ran(self._replay, getattr(self._trainer, "global_steps", 0))
        return self._inner.generate_sequences(*args, **kwargs)

    def _append_metrics(self, output: Any) -> None:
        point = self._replay.last_telemetry
        metrics = getattr(output, "meta_info", {}).get("metrics") if output is not None else None
        if point is None or not isinstance(metrics, dict):
            return
        for name, value in point.metrics().items():
            metrics[name] = [value]

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def __repr__(self) -> str:
        return f"ReplayWorkerGroup({self._inner!r})"


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


def bind_checkpoint(replay: ReservoirReplay, global_steps: int, local_dir=None) -> None:
    """When the trainer saves ``global_step_<n>``, snapshot a durable buffer as ``step-<n>``.

    With ``local_dir`` the buffer's checkpoints are pruned to the steps the
    trainer still has (it removes old ones under ``max_actor_ckpt_to_keep``
    before this is called). A buffer without a directory has nothing to
    bind; the call is a no-op.
    """
    if not isinstance(replay.buffer, DurableRolloutBuffer):
        return
    replay.buffer.checkpoint(checkpoint_tag(global_steps))
    steps = trainer_checkpoint_steps(local_dir)
    if steps is not None:
        replay.buffer.prune_checkpoints({checkpoint_tag(n) for n in steps | {int(global_steps)}})


# ---------------------------------------------------------------------------
# The mixin and the trainer class
# ---------------------------------------------------------------------------

class ReservoirReplayMixin:
    """Wraps the actor worker group and binds the buffer to verl's checkpoints.

    Place it before ``RayPPOTrainer`` in the base list. The class using it
    must set ``self.replay_buffer`` to a ``ReservoirReplay`` before
    ``init_workers`` runs.
    """

    replay_buffer: ReservoirReplay

    def init_workers(self):  # type: ignore[override]
        super().init_workers()  # type: ignore[misc]
        self.actor_rollout_wg = ReplayWorkerGroup(self.actor_rollout_wg, self.replay_buffer, self)  # type: ignore[attr-defined]

    def _save_checkpoint(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        result = super()._save_checkpoint(*args, **kwargs)  # type: ignore[misc]
        step = int(self.global_steps)  # type: ignore[attr-defined]
        assert_hook_ran(self.replay_buffer, step)
        bind_checkpoint(self.replay_buffer, step, config_value(self.config, "trainer.default_local_dir"))  # type: ignore[attr-defined]
        return result

    def _load_checkpoint(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        result = super()._load_checkpoint(*args, **kwargs)  # type: ignore[misc]
        resume_from_checkpoint(self.replay_buffer, int(self.global_steps))  # type: ignore[attr-defined]
        return result

    def fit(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        result = super().fit(*args, **kwargs)  # type: ignore[misc]
        assert_hook_ran(self.replay_buffer, int(self.global_steps))  # type: ignore[attr-defined]
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
    "ReplayWorkerGroup",
    "ReservoirReplay",
    "ReservoirReplayMixin",
    "STAT_NAMES",
    "SUPPORTED_ADV_ESTIMATORS",
    "UNSUPPORTED_BATCH_KEYS",
    "UNSUPPORTED_NON_TENSOR_KEYS",
    "assert_hook_ran",
    "bind_checkpoint",
    "build_trainer_class",
    "config_value",
    "trainer_checkpoint_steps",
]
