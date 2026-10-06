"""Conversion between TRL's padded batch tensors and ``Rollout`` objects.

``GRPOTrainer._generate_and_score_completions`` returns one dict per
generation step whose tensors describe ``B`` completion rows:

- ``prompt_ids`` / ``prompt_mask``: ``(B, Lp)``, **left**-padded, so the
  mask is a suffix mask (zeros, then ones).
- ``completion_ids`` / ``completion_mask``: ``(B, Lc)``, **right**-padded;
  the mask is a prefix mask (ones up to and including the first EOS, then
  zeros; a truncated completion may be all zeros).
- ``advantages``: ``(B,)`` float, group-centred; every row of a
  zero-variance group is exactly ``0.0``.
- ``old_per_token_logps`` / ``ref_per_token_logps``: ``(B, Lc)`` float,
  optional, aligned with ``completion_ids``.

Rows ``[g * G, (g + 1) * G)`` belong to prompt ``g`` (``G`` =
``num_generations``). This module turns such a dict into ``Rollout``
objects grouped by prompt (``rows_to_groups``) and writes sampled
rollouts back into chosen rows of such a dict (``write_rows``), padding
the batch when a replayed sequence is longer than the current one
(``pad_batch``). It knows nothing about the replay buffer or TRL
itself; it imports torch and ``reservoir.rollout`` only.

Nothing here modifies an input tensor in place: the trainer splits and
reuses the dict it returned across several optimizer steps.
``write_rows`` returns fresh tensors for every key it touches;
``pad_batch`` returns the input dict itself when nothing changes.

Field mapping for one stored row ``r``::

    Rollout.tokens    = completion_ids[r][:n]      n = prefix length of completion_mask[r]
    Rollout.logprobs  = logprobs[r][:n]            behavior logprobs, clamped (see below)
    Rollout.reward    = advantages[r]              TRL's advantage, the value the loss consumes
    Rollout.metadata  = {"prompt_ids": [...unpadded prompt...], "row": r,
                         "global_step": step, "ref_logprobs": [...],  # only if the batch has them
                         "rewards": {name: value, ...}}             # only with reward provenance

Reward provenance: ``rows_to_groups(..., reward_names=trainer.reward_func_names,
rewards_per_func=<(B, F) tensor>)`` records each row's per-reward-function
values under ``metadata["rewards"]`` (``rollout_manifest.REWARDS_KEY``), so
the buffer's manifest carries them next to the example. A ``NaN`` is TRL's
"this function abstained on this row" and is left out of that row's
mapping (a row where every function abstained gets ``{}``, which is not
the same as no provenance); an infinity is a data error and raises. The
values are numeric only and outside the content digest.

Metadata values are lists, not tuples, so the durable buffer's JSON
round-trip check accepts them.

Logprob sanity: a log-probability is ``<= 0`` by definition, but
``x - logsumexp(x)`` can land a hair above zero in floating point. Values
in ``(0, MAX_POSITIVE_LOGPROB]`` are clamped to ``0.0`` and counted;
anything larger, or non-finite, is a data error and raises.
"""

from __future__ import annotations

import hashlib
import math
from typing import Final, NamedTuple, Optional, Sequence

import torch
import torch.nn.functional as F

from reservoir.rollout import MAX_SOURCE_LENGTH, Rollout

MAX_POSITIVE_LOGPROB: Final[float] = 1e-3
"""Largest positive logprob treated as rounding noise and clamped to 0."""

PROMPT_ID_BYTES: Final[int] = 8
"""Digest size of the content-addressed prompt id (16 hex characters)."""

_CORE_KEYS: Final[tuple[str, ...]] = (
    "prompt_ids", "prompt_mask", "completion_ids", "completion_mask", "advantages",
)


class GroupSpec(NamedTuple):
    """The stored rollouts of one prompt group and the batch rows they came from."""

    prompt_id: str
    rollouts: tuple[Rollout, ...]
    rows: tuple[int, ...]


NEAR_DEAD_ADVANTAGE: Final[float] = 1e-3
"""A live group whose every |advantage| is below this is counted as near-dead.

A group is replaced only when its advantages are exactly zero, which is
exactly when it contributes no gradient. TRL computes advantages in
float32, and for a non-power-of-two group size with equal non-integer
rewards the mean can carry a rounding residue that TRL then trains on;
such a group is live for the loss and is therefore kept, but it is
counted so the condition is visible in the adapter's statistics.
"""


class RowConversion(NamedTuple):
    """Result of ``rows_to_groups``: the groups to store plus what was left out."""

    groups: tuple[GroupSpec, ...]
    dead_rows: tuple[int, ...]   # rows of groups whose advantages are all zero
    dead_groups: int
    skipped_rows: int            # rows of live groups with an empty completion mask
    clamped_logprobs: int        # tiny positive logprobs clamped to 0
    near_dead_groups: int = 0    # live groups with every |advantage| below NEAR_DEAD_ADVANTAGE


# ---------------------------------------------------------------------------
# Masks, ids, logprobs
# ---------------------------------------------------------------------------

def _mask_lengths(mask: torch.Tensor, prefix: bool, name: str) -> tuple[int, ...]:
    """Length of the ones-run in each row; raise if a row is not a pure run."""
    m = mask.bool()
    lengths = m.sum(dim=1)
    positions = torch.arange(m.size(1), device=m.device).unsqueeze(0)
    if prefix:
        expected = positions < lengths.unsqueeze(1)
    else:
        expected = positions >= (m.size(1) - lengths).unsqueeze(1)
    bad = (m != expected).any(dim=1)
    if bool(bad.any()):
        row = int(bad.nonzero()[0, 0])
        kind = "prefix (ones then zeros)" if prefix else "suffix (zeros then ones)"
        raise ValueError(f"{name} row {row} is not a {kind} mask")
    return tuple(int(x) for x in lengths.tolist())


def prefix_mask_lengths(mask: torch.Tensor) -> tuple[int, ...]:
    """Number of leading ones per row of a right-padded mask; ``ValueError`` otherwise."""
    return _mask_lengths(mask, prefix=True, name="completion_mask")


def suffix_mask_lengths(mask: torch.Tensor) -> tuple[int, ...]:
    """Number of trailing ones per row of a left-padded mask; ``ValueError`` otherwise."""
    return _mask_lengths(mask, prefix=False, name="prompt_mask")


def prompt_id_of(tokens: Sequence[int]) -> str:
    """Content-addressed id of a prompt: BLAKE2b over its token ids."""
    encoded = ",".join(str(int(t)) for t in tokens).encode("ascii")
    return hashlib.blake2b(encoded, digest_size=PROMPT_ID_BYTES).hexdigest()


def _finite_floats(values: Sequence[float], row: int, name: str) -> list[float]:
    """Python floats of ``values``; a NaN or infinity raises naming the row."""
    out = [float(v) for v in values]
    for i, value in enumerate(out):
        if not math.isfinite(value):
            raise ValueError(f"row {row}, token {i}: {name} {value!r} is not finite")
    return out


def clamp_logprobs(values: Sequence[float], row: int) -> tuple[tuple[float, ...], int]:
    """Return ``(logprobs, n_clamped)``; raise on non-finite or clearly positive values."""
    out: list[float] = []
    clamped = 0
    for i, raw in enumerate(values):
        value = float(raw)
        if not math.isfinite(value) or value > MAX_POSITIVE_LOGPROB:
            raise ValueError(
                f"row {row}, token {i}: behavior logprob {value!r} is not a finite value <= 0"
            )
        if value > 0.0:
            value = 0.0
            clamped += 1
        out.append(value)
    return tuple(out), clamped


# ---------------------------------------------------------------------------
# Batch -> rollouts
# ---------------------------------------------------------------------------

class _Columns(NamedTuple):
    """The batch as Python lists, one entry per row, ready for slicing."""

    prompts: list[list[int]]          # unpadded prompt ids
    completions: list[list[int]]      # padded completion ids
    lengths: tuple[int, ...]          # completion prefix lengths
    advantages: list[float]
    logprobs: list[list[float]]
    refs: Optional[list[list[float]]]
    reward_names: Optional[tuple[str, ...]] = None    # reward provenance, both or neither
    rewards: Optional[list[list[float]]] = None


def _reward_provenance(
    names: Optional[Sequence[str]], values: Optional[torch.Tensor], batch: int
) -> tuple[Optional[tuple[str, ...]], Optional[list[list[float]]]]:
    """Validate the optional per-reward-function inputs of ``rows_to_groups``."""
    if (names is None) != (values is None):
        raise ValueError("reward_names and rewards_per_func must be given together")
    if names is None or values is None:
        return None, None
    names_t = tuple(names)
    if any(not isinstance(n, str) or not n or len(n) > MAX_SOURCE_LENGTH or not n.isprintable() or not n.strip()
           for n in names_t):
        raise ValueError(
            f"reward_names must be non-empty printable strings of at most {MAX_SOURCE_LENGTH} characters, "
            f"got {list(names_t)}"
        )
    if len(set(names_t)) != len(names_t):
        raise ValueError(f"reward_names must be distinct, got {list(names_t)}")
    if values.dim() != 2:
        raise ValueError(f"rewards_per_func must have shape (rows, functions), got {tuple(values.shape)}")
    if values.size(0) != batch:
        raise ValueError(f"rewards_per_func has {values.size(0)} rows for a batch of {batch}")
    if values.size(1) != len(names_t):
        raise ValueError(f"rewards_per_func has {values.size(1)} columns for {len(names_t)} reward_names")
    return names_t, values.tolist()


def _row_rewards(names: tuple[str, ...], values: Sequence[float], row: int) -> dict[str, float]:
    """``{name: value}`` for one row, leaving out the functions that abstained (NaN)."""
    out: dict[str, float] = {}
    for name, value in zip(names, values):
        value = float(value)
        if math.isnan(value):
            continue
        if not math.isfinite(value):
            raise ValueError(f"row {row}: reward {name!r} is {value!r}, not a finite value")
        out[name] = value
    return out


def _columns(
    output: dict, logprobs: torch.Tensor,
    reward_names: Optional[Sequence[str]] = None, rewards_per_func: Optional[torch.Tensor] = None,
) -> _Columns:
    """Validate masks and shapes once and move the batch to Python lists."""
    if tuple(logprobs.shape) != tuple(output["completion_ids"].shape):
        raise ValueError(
            f"logprobs must have the completion shape {tuple(output['completion_ids'].shape)}, "
            f"got {tuple(logprobs.shape)}"
        )
    prompt_lengths = suffix_mask_lengths(output["prompt_mask"])
    width = output["prompt_ids"].size(1)
    prompt_rows = output["prompt_ids"].tolist()
    advantages = output["advantages"].tolist()
    for r, value in enumerate(advantages):
        if not math.isfinite(value):
            raise ValueError(f"row {r}: advantage {value!r} is not finite")
    ref = output.get("ref_per_token_logps")
    names, rewards = _reward_provenance(reward_names, rewards_per_func, len(advantages))
    return _Columns(
        prompts=[row[width - n:] if n else [] for row, n in zip(prompt_rows, prompt_lengths)],
        completions=output["completion_ids"].tolist(),
        lengths=prefix_mask_lengths(output["completion_mask"]),
        advantages=advantages,
        logprobs=logprobs.tolist(),
        refs=ref.tolist() if ref is not None else None,
        reward_names=names,
        rewards=rewards,
    )


def _group_rollouts(
    cols: _Columns, rows: range, group_index: int, step: int
) -> tuple[list[Rollout], list[int], int, int]:
    """Rollouts of one live group: ``(rollouts, kept_rows, skipped, clamped)``."""
    rollouts: list[Rollout] = []
    kept: list[int] = []
    prompt = cols.prompts[rows.start]
    skipped = clamped = 0
    for r in rows:
        if cols.prompts[r] != prompt:
            raise ValueError(
                f"group {group_index}: rows {rows.start}-{rows.stop - 1} do not share one prompt; "
                "check num_generations"
            )
        n = cols.lengths[r]
        if n == 0:
            skipped += 1
            continue
        lp, n_clamped = clamp_logprobs(cols.logprobs[r][:n], row=r)
        clamped += n_clamped
        metadata: dict = {"prompt_ids": list(prompt), "row": r, "global_step": step}
        if cols.refs is not None:
            metadata["ref_logprobs"] = _finite_floats(cols.refs[r][:n], r, "ref logprob")
        if cols.reward_names is not None and cols.rewards is not None:
            metadata["rewards"] = _row_rewards(cols.reward_names, cols.rewards[r], r)
        rollouts.append(
            Rollout(tokens=cols.completions[r][:n], logprobs=lp, reward=cols.advantages[r], metadata=metadata)
        )
        kept.append(r)
    return rollouts, kept, skipped, clamped


def rows_to_groups(
    output: dict, *, num_generations: int, step: int, logprobs: torch.Tensor,
    reward_names: Optional[Sequence[str]] = None, rewards_per_func: Optional[torch.Tensor] = None,
) -> RowConversion:
    """Turn a TRL output dict into ``Rollout`` groups ready for ``add_group``.

    ``logprobs`` are the behavior logprobs for every row (TRL's
    ``old_per_token_logps`` or ones computed by the caller), shape
    ``(B, Lc)``. Zero-variance groups and rows with an empty completion
    mask are left out and counted; the rows of zero-variance groups are
    returned so the caller can replace them. ``reward_names`` and
    ``rewards_per_func`` (``(B, F)``, given together or not at all) attach
    each row's per-reward-function values to its metadata; see the module
    docstring.

    Raises
    ------
    ValueError
        ``B`` not a multiple of ``num_generations``, ``logprobs`` of the
        wrong shape, a malformed mask, a non-finite advantage, rows of one
        group with different prompts (a wrong ``num_generations``), a bad
        logprob, or misaligned or non-finite reward provenance. Every
        message names the row or group.
    """
    batch = output["advantages"].size(0)
    if num_generations < 1 or batch % num_generations:
        raise ValueError(
            f"batch of {batch} rows is not a multiple of num_generations={num_generations}"
        )
    cols = _columns(output, logprobs, reward_names, rewards_per_func)
    groups: list[GroupSpec] = []
    dead_rows: list[int] = []
    skipped = clamped = near_dead = 0
    for g in range(batch // num_generations):
        rows = range(g * num_generations, (g + 1) * num_generations)
        if all(cols.advantages[r] == 0.0 for r in rows):
            dead_rows.extend(rows)
            continue
        if all(abs(cols.advantages[r]) < NEAR_DEAD_ADVANTAGE for r in rows):
            near_dead += 1
        rollouts, kept, n_skipped, n_clamped = _group_rollouts(cols, rows, g, step)
        skipped += n_skipped
        clamped += n_clamped
        if rollouts:
            groups.append(GroupSpec(prompt_id_of(cols.prompts[rows.start]), tuple(rollouts), tuple(kept)))
    return RowConversion(
        tuple(groups), tuple(dead_rows), len(dead_rows) // num_generations, skipped, clamped, near_dead
    )


# ---------------------------------------------------------------------------
# Rollouts -> batch
# ---------------------------------------------------------------------------

def pad_batch(
    output: dict, target_prompt_len: int, target_completion_len: int, pad_token_id: int
) -> dict:
    """Grow a batch to the target widths: prompts on the left, completions on the right.

    Returns ``output`` itself when nothing changes, otherwise a new dict
    whose padded tensors are new and whose other entries are shared.
    Shrinking is a ``ValueError``.
    """
    lp = output["prompt_ids"].size(1)
    lc = output["completion_ids"].size(1)
    if target_prompt_len < lp or target_completion_len < lc:
        raise ValueError(
            f"cannot shrink a batch: targets ({target_prompt_len}, {target_completion_len}) "
            f"below current ({lp}, {lc})"
        )
    if target_prompt_len == lp and target_completion_len == lc:
        return output
    new = dict(output)
    dp = target_prompt_len - lp
    if dp:
        new["prompt_ids"] = F.pad(output["prompt_ids"], (dp, 0), value=pad_token_id)
        new["prompt_mask"] = F.pad(output["prompt_mask"], (dp, 0), value=0)
    dc = target_completion_len - lc
    if dc:
        fills = (
            ("completion_ids", pad_token_id), ("completion_mask", 0),
            ("old_per_token_logps", 0.0), ("ref_per_token_logps", 0.0),
        )
        for key, fill in fills:
            if key in output:
                new[key] = F.pad(output[key], (0, dc), value=fill)
    return new


def _check_write_args(
    output: dict, rows: tuple[int, ...], rollouts: tuple[Rollout, ...], advantages: tuple[float, ...]
) -> bool:
    """Validate ``write_rows`` inputs; return whether ref logprobs are in play."""
    if not (len(rows) == len(rollouts) == len(advantages)):
        raise ValueError(
            f"rows ({len(rows)}), rollouts ({len(rollouts)}) and advantages "
            f"({len(advantages)}) must have the same length"
        )
    batch = output["completion_ids"].size(0)
    for r in rows:
        if not 0 <= r < batch:
            raise ValueError(f"row {r} is outside the batch of {batch} rows")
    if len(set(rows)) != len(rows):
        raise ValueError(f"rows contains duplicates: {rows}")
    handled = set(_CORE_KEYS) | {"old_per_token_logps", "ref_per_token_logps", "num_items_in_batch"}
    for key, value in output.items():
        if key not in handled and isinstance(value, torch.Tensor) and value.dim() >= 2 and value.size(0) == batch:
            raise ValueError(
                f"batch entry {key!r} has one row per sample and write_rows does not know how to "
                "replace its rows"
            )
    if "old_per_token_logps" not in output:
        raise ValueError(
            "the batch has no old_per_token_logps; behavior logprobs for the fresh rows "
            "must be attached before replayed rows are written"
        )
    has_ref = "ref_per_token_logps" in output
    for r, rollout in zip(rows, rollouts):
        ref = rollout.metadata.get("ref_logprobs")
        if (ref is not None) != has_ref:
            raise ValueError(
                f"row {r}: the batch {'has' if has_ref else 'has no'} ref_per_token_logps but "
                f"the replayed rollout {'has none' if has_ref else 'carries them'}; "
                "the buffer was filled under a different KL setting (beta)"
            )
        if ref is not None and len(ref) != len(rollout):
            raise ValueError(f"row {r}: ref_logprobs length {len(ref)} != {len(rollout)} tokens")
    return has_ref


def write_rows(
    output: dict,
    rows: Sequence[int],
    rollouts: Sequence[Rollout],
    advantages: Sequence[float],
    pad_token_id: int,
) -> dict:
    """Return a copy of ``output`` with ``rows[k]`` replaced by ``rollouts[k]``.

    ``advantages[k]`` is written as the row's advantage (the caller has
    already applied any importance-sampling weight). The batch is padded
    first if a replayed prompt or completion is longer than the current
    width; shorter replayed rows are padded into place with
    ``pad_token_id`` / ``0`` / ``0.0``. Requires ``old_per_token_logps``
    in ``output``; ``ref_per_token_logps`` is written iff the batch has it,
    and every replayed rollout must agree. ``num_items_in_batch``, when
    present, is recomputed from the final mask (single-process value; a
    multi-process caller must gather it). Any other per-row tensor in the
    batch is a ``ValueError``, since its rows would go stale.
    """
    rows_t = tuple(int(r) for r in rows)
    rollouts_t = tuple(rollouts)
    adv_t = tuple(float(a) for a in advantages)
    has_ref = _check_write_args(output, rows_t, rollouts_t, adv_t)

    target_p = max([output["prompt_ids"].size(1)] + [len(r.metadata["prompt_ids"]) for r in rollouts_t])
    target_c = max([output["completion_ids"].size(1)] + [len(r) for r in rollouts_t])
    padded = pad_batch(output, target_p, target_c, pad_token_id)
    keys = _CORE_KEYS + ("old_per_token_logps",) + (("ref_per_token_logps",) if has_ref else ())
    new = dict(padded)
    for key in keys:
        new[key] = padded[key].clone()

    for r, rollout, adv in zip(rows_t, rollouts_t, adv_t):
        _write_row(new, r, rollout, adv, pad_token_id, has_ref)
    if "num_items_in_batch" in output:
        old = output["num_items_in_batch"]
        new["num_items_in_batch"] = new["completion_mask"].sum().to(old.dtype if isinstance(old, torch.Tensor) else torch.long)
    return new


def _write_row(new: dict, r: int, rollout: Rollout, adv: float, pad_token_id: int, has_ref: bool) -> None:
    """Overwrite row ``r`` of the (already cloned) tensors in ``new``."""
    prompt = rollout.metadata["prompt_ids"]
    lp_r, n = len(prompt), len(rollout)
    width = new["prompt_ids"].size(1)

    def as_row(values, like: torch.Tensor) -> torch.Tensor:
        return torch.tensor(values, dtype=like.dtype, device=like.device)

    new["prompt_ids"][r] = pad_token_id
    new["prompt_ids"][r, width - lp_r:] = as_row(prompt, new["prompt_ids"])
    new["prompt_mask"][r] = 0
    new["prompt_mask"][r, width - lp_r:] = 1
    new["completion_ids"][r] = pad_token_id
    new["completion_ids"][r, :n] = as_row(rollout.tokens, new["completion_ids"])
    new["completion_mask"][r] = 0
    new["completion_mask"][r, :n] = 1
    new["old_per_token_logps"][r] = 0.0
    new["old_per_token_logps"][r, :n] = as_row(rollout.logprobs, new["old_per_token_logps"])
    if has_ref:
        new["ref_per_token_logps"][r] = 0.0
        new["ref_per_token_logps"][r, :n] = as_row(
            rollout.metadata["ref_logprobs"], new["ref_per_token_logps"]
        )
    new["advantages"][r] = adv


def verify_written_rows(output: dict, rows: Sequence[int], rollouts: Sequence[Rollout],
                        advantages: Sequence[float]) -> None:
    """Re-read ``rows`` of ``output`` and confirm each holds exactly its rollout.

    Prompt ids under the prompt mask, completion ids and behavior logprobs
    under the completion mask, and the advantage must all equal what the
    rollout and the caller supplied. A padding or masking mistake in
    ``write_rows`` is a ``ValueError`` here, before any witness is written.
    """
    for r, rollout, adv in zip(rows, rollouts, advantages):
        pmask = output["prompt_mask"][r].bool()
        cmask = output["completion_mask"][r].bool()
        prompt = output["prompt_ids"][r][pmask].tolist()
        tokens = output["completion_ids"][r][cmask].tolist()
        logps = output["old_per_token_logps"][r][cmask].tolist()
        if prompt != list(rollout.metadata["prompt_ids"]):
            raise ValueError(f"row {r}: written prompt ids differ from the replayed rollout's")
        if tokens != list(rollout.tokens):
            raise ValueError(f"row {r}: written completion ids differ from the replayed rollout's tokens")
        if any(abs(a - b) > 0.0 for a, b in zip(logps, rollout.logprobs)) or len(logps) != len(rollout.logprobs):
            raise ValueError(f"row {r}: written behavior logprobs differ from the replayed rollout's")
        if float(output["advantages"][r]) != float(torch.tensor(adv, dtype=output["advantages"].dtype)):
            raise ValueError(f"row {r}: written advantage {float(output['advantages'][r])} is not {adv}")


__all__ = [
    "MAX_POSITIVE_LOGPROB",
    "GroupSpec",
    "RowConversion",
    "clamp_logprobs",
    "pad_batch",
    "prefix_mask_lengths",
    "prompt_id_of",
    "rows_to_groups",
    "suffix_mask_lengths",
    "verify_written_rows",
    "write_rows",
]
