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
2. Store: every row of a live group becomes a ``Rollout`` (``reward`` =
   TRL's advantage, the value the loss consumes) and each group is one
   ``add_group(prompt_id, model_version=global_step, ...)``.
3. Replay: with ``d`` dead groups, ``sample(d * G, current_version=
   global_step)`` draws rollouts (possibly from different prompts; the
   loss does not need group contiguity) and writes them into the dead
   rows. Each replayed advantage is multiplied by its importance-sampling
   weight, which is exact for every loss type that is positively
   homogeneous in the advantage (all of them except ``vespo``). The batch
   is padded if a replayed sequence is longer than the current width,
   ``old_per_token_logps`` is attached for all rows, and
   ``num_items_in_batch`` is recomputed from the final mask.

A batch with no dead groups, or an empty buffer, is returned as the very
dict TRL produced. When TRL did not compute ``old_per_token_logps``, one
extra no-grad forward has run by then, which consumes RNG state under
dropout; apart from that a step without replay matches a plain
``GRPOTrainer`` step. A live group is stored before the same step's dead
groups are replaced, so a replayed row can be a copy of a fresh row from
the same batch; that is how TRL's own buffer behaved too.

Versions, half-life and ``max_policy_age`` are counted in optimizer
steps (``trainer.state.global_step``).

Scope and guards
----------------
Text-only, single process, tested against TRL 1.13.0 (see
``_trl_compat``). Outputs carrying tool masks, vLLM importance-sampling
ratios or vision inputs raise ``NotImplementedError`` naming the key;
more than one process raises too. ``ReservoirReplayCallback`` raises at
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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Optional, Union

import torch

from reservoir.durable_rollout import DurableRolloutBuffer
from reservoir.integrations._trl_compat import require_trl
from reservoir.integrations._trl_rows import RowConversion, rows_to_groups, write_rows
from reservoir.priorities import DEFAULT_EPSILON, PriorityStrategy, _require_epsilon
from reservoir.rollout import Rollout, RolloutGroup
from reservoir.rollout_attest import AttestTarget
from reservoir.rollout_buffer import RolloutBatch, RolloutBuffer

UNSUPPORTED_OUTPUT_KEYS: Final[tuple[str, ...]] = (
    "tool_mask",
    "importance_sampling_ratio",
    "sampling_per_token_logps",
    "pixel_values",
    "image_grid_thw",
    "pixel_attention_mask",
    "image_sizes",
    "spatial_shapes",
    "num_tiles",
    "token_type_ids",
    "mm_token_type_ids",
    "image_position_ids",
    "num_images",
)
"""Output keys that would need per-row replacement logic this adapter does not have."""

STAT_NAMES: Final[tuple[str, ...]] = (
    "hook_calls", "ingested_rows", "ingested_groups", "dead_groups", "skipped_rows",
    "clamped_logprobs", "replaced_rows", "logprob_forwards",
)


@dataclass(frozen=True)
class StoredAdvantagePriority(PriorityStrategy):
    """``|reward| + epsilon`` where ``reward`` holds TRL's advantage.

    TRL centres advantages within each group before the adapter sees
    them, so ``|advantage|`` is already the group-relative magnitude
    that advantage-prioritized replay samples on. ``AdvantagePriority``
    would re-centre on the mean of the rows that survived truncation,
    which is not the same quantity.
    """

    epsilon: float = DEFAULT_EPSILON

    def __post_init__(self) -> None:
        object.__setattr__(self, "epsilon", _require_epsilon(self.epsilon))

    def score(self, rollout: Rollout, group: RolloutGroup) -> float:
        return abs(rollout.reward) + self.epsilon


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

    Attributes
    ----------
    buffer
        The underlying buffer; read it for ``size``, ``live_positions()``,
        ``attestation_log`` and so on.
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
        **rollout_buffer_kwargs: Any,
    ) -> None:
        kwargs = dict(
            capacity=capacity,
            priority=priority if priority is not None else StoredAdvantagePriority(),
            half_life=half_life,
            max_policy_age=max_policy_age,
            beta=beta,
            seed=seed,
            **rollout_buffer_kwargs,
        )
        if directory is None:
            self.buffer: Union[RolloutBuffer, DurableRolloutBuffer] = RolloutBuffer(
                attest=attest, manifest=manifest, **kwargs
            )
        else:
            self.buffer = DurableRolloutBuffer(directory, attest=attest, manifest=manifest, **kwargs)
        self.source = source
        self.beta = float(beta)
        self.stats: dict[str, int] = {name: 0 for name in STAT_NAMES}
        self.last_replay: Optional[RolloutBatch] = None

    # -- the hook ------------------------------------------------------------

    def mix(self, output: dict, trainer: Any) -> dict:
        """Store the live rows of ``output`` and replace its dead rows with replayed ones.

        ``trainer`` is the ``GRPOTrainer`` (or a stand-in exposing the
        same members). Returns ``output`` itself when nothing was
        replaced, otherwise a new dict; TRL's tensors are never modified.

        Order matters: groups are stored, then the buffer is advanced to
        ``global_step`` (evicting entries older than ``max_policy_age``),
        and only then are dead rows replaced from whatever is still live.
        If the buffer raises while groups are being stored, the groups
        stored before it stay stored and the counters are not updated.
        """
        step = int(trainer.state.global_step)
        self._check(output, trainer, step)
        self.last_replay = None
        self.stats["hook_calls"] += 1
        logprobs = output.get("old_per_token_logps")
        if logprobs is None:
            logprobs = self._behavior_logprobs(output, trainer)
        conversion = self._ingest(output, trainer, step, logprobs)
        if step > self.buffer.current_version:
            self.buffer.advance(step)
        if not conversion.dead_rows or self.buffer.size == 0 or self.buffer.total == 0:
            return output
        return self._replay(output, trainer, step, logprobs, conversion.dead_rows)

    def _check(self, output: dict, trainer: Any, step: int) -> None:
        """Refuse configurations the adapter does not handle, before touching anything."""
        if step < self.buffer.current_version:
            raise ValueError(
                f"global_step {step} is below the buffer's current version "
                f"{self.buffer.current_version}; versions only move forward. A resumed run "
                "must reopen the buffer at or behind the step it resumes from"
            )
        if trainer.accelerator.num_processes > 1:
            raise NotImplementedError(
                "ReservoirReplay supports a single process; "
                f"the trainer runs {trainer.accelerator.num_processes}"
            )
        if self.beta > 0.0 and getattr(trainer, "loss_type", None) == "vespo":
            raise ValueError(
                "loss_type 'vespo' is not linear in the advantage, so importance weights "
                "cannot be folded into it; use ReservoirReplay(beta=0.0) with vespo"
            )
        for key in UNSUPPORTED_OUTPUT_KEYS:
            if key in output:
                raise NotImplementedError(
                    f"ReservoirReplay cannot replay batches with {key!r} "
                    "(tools, vLLM importance sampling and vision inputs are unsupported)"
                )

    def _behavior_logprobs(self, output: dict, trainer: Any) -> torch.Tensor:
        """One no-grad forward to get the logprobs TRL did not compute."""
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

    def _ingest(self, output: dict, trainer: Any, step: int, logprobs: torch.Tensor) -> RowConversion:
        """Convert the batch and add every live group to the buffer."""
        conversion = rows_to_groups(
            output, num_generations=trainer.num_generations, step=step, logprobs=logprobs
        )
        for group in conversion.groups:
            self.buffer.add_group(group.prompt_id, step, group.rollouts, source=self.source)
        self.stats["ingested_rows"] += sum(len(group.rollouts) for group in conversion.groups)
        self.stats["ingested_groups"] += len(conversion.groups)
        self.stats["dead_groups"] += conversion.dead_groups
        self.stats["skipped_rows"] += conversion.skipped_rows
        self.stats["clamped_logprobs"] += conversion.clamped_logprobs
        return conversion

    def _replay(
        self, output: dict, trainer: Any, step: int, logprobs: torch.Tensor, dead_rows: tuple[int, ...]
    ) -> dict:
        """Sample one rollout per dead row and write them into the batch.

        ``write_rows`` recomputes ``num_items_in_batch`` from the final
        mask; with a single process that is the value TRL would gather.
        """
        batch = self.buffer.sample(len(dead_rows), current_version=step)
        advantages = [
            rollout.reward * float(weight) for rollout, weight in zip(batch.rollouts, batch.is_weights)
        ]
        with_logprobs = output if "old_per_token_logps" in output else {**output, "old_per_token_logps": logprobs}
        new = write_rows(
            with_logprobs, dead_rows, batch.rollouts, advantages, trainer._tokenizer.pad_token_id
        )
        self.last_replay = batch
        self.stats["replaced_rows"] += len(dead_rows)
        return new

    def close(self) -> None:
        """Close the attestation file, if one was opened."""
        self.buffer.close()

    def __repr__(self) -> str:
        source = f", source={self.source!r}" if self.source is not None else ""
        return f"ReservoirReplay({self.buffer!r}, beta={self.beta}{source})"


class ReservoirReplayMixin:
    """Overrides ``_generate_and_score_completions`` to route its output through ``replay_buffer``.

    Place it before ``GRPOTrainer`` in the base list. The class using it
    must set ``self.replay_buffer`` to a ``ReservoirReplay``.
    """

    replay_buffer: ReservoirReplay

    def _generate_and_score_completions(self, inputs):  # type: ignore[override]
        output = super()._generate_and_score_completions(inputs)  # type: ignore[misc]
        if not self.model.training:  # type: ignore[attr-defined]
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
        """Fails the run early if the replay hook is not being called."""

        def __init__(self, replay: ReservoirReplay) -> None:
            self.replay = replay

        def on_step_end(self, args, state, control, **kwargs):
            assert_hook_ran(self.replay, state.global_step)
            return control

    class ReservoirGRPOTrainer(ReservoirReplayMixin, support.grpo_trainer):  # type: ignore[misc,valid-type]
        """``GRPOTrainer`` with Reservoir replay of dead groups. See the module docstring."""

        def __init__(self, *args: Any, replay_buffer: ReservoirReplay, **kwargs: Any) -> None:
            if not isinstance(replay_buffer, ReservoirReplay):
                raise TypeError(
                    f"replay_buffer must be a ReservoirReplay, got {type(replay_buffer).__name__}"
                )
            super().__init__(*args, **kwargs)
            self.replay_buffer = replay_buffer
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
    "ReservoirReplay",
    "ReservoirReplayMixin",
    "StoredAdvantagePriority",
    "UNSUPPORTED_OUTPUT_KEYS",
    "assert_hook_ran",
    "build_trainer_class",
]
