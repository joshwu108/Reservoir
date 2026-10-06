"""Conversion between verl's ``DataProto`` batch tensors and ``Rollout`` objects.

``RayPPOTrainer.fit`` hands ``actor_rollout_wg.update_actor`` one
``DataProto`` per training step. Its ``batch`` (a TensorDict) describes
``B`` response rows; this module works on that TensorDict as a plain
mapping of tensors and knows nothing about Ray, the worker group or the
replay buffer. The keys it reads and rewrites:

- ``prompts`` ``(B, Lp)`` long, **left**-padded; ``responses`` ``(B, Lr)``
  long, **right**-padded.
- ``input_ids`` ``(B, Lp + Lr)`` = ``prompts`` followed by ``responses``;
  ``attention_mask`` likewise, whose last ``Lr`` columns are
  ``response_mask`` ``(B, Lr)`` (a prefix mask: ones up to and including
  the EOS, then zeros). A multi-turn ``loss_mask`` with tool output
  masked out in the middle is not a prefix mask and is refused.
- ``position_ids`` ``(B, Lp + Lr)``: ``clip(cumsum(mask) - 1, 0)`` over the
  prompt, then ``last prompt position + 1, 2, ...`` over the response
  (padding included), which is how verl's rollout builds them. A 3-D
  ``position_ids`` (the multimodal rope of vision models) is refused.
- ``old_log_probs`` ``(B, Lr)`` float, the behavior logprobs verl computes
  for every batch before the update; ``ref_log_prob`` ``(B, Lr)``,
  present iff the KL loss is on.
- ``advantages``, ``returns`` ``(B, Lr)`` float, per token. Under GRPO the
  advantage is one outcome-level value broadcast over the response
  tokens, and ``returns`` equals ``advantages``; a row whose advantages
  vary over its tokens is refused (a critic-style estimator), and a
  zero-variance group is exactly ``0.0`` in every row.
- ``token_level_scores``, ``token_level_rewards`` ``(B, Lr)`` float: the
  reward at the last response token, zero elsewhere, identical unless
  ``algorithm.use_kl_in_reward`` is on (refused: replayed rows have no
  per-token KL rewards to carry).

Rows of one prompt share a ``uid`` (``non_tensor_batch["uid"]``); they
need not be contiguous, since ``balance_batch`` reorders rows by length.

Field mapping for one stored row ``r``::

    Rollout.tokens    = responses[r][:n]           n = prefix length of response_mask[r]
    Rollout.logprobs  = old_log_probs[r][:n]       behavior logprobs, clamped (see _trl_rows)
    Rollout.reward    = advantages[r][0]           verl's advantage, the value the loss consumes
    Rollout.metadata  = {"prompt_ids": [...unpadded prompt...], "row": r, "global_step": step,
                         "uid": uid, "score": token_level_scores[r].sum(),
                         "ref_logprobs": [...]}                       # only if the batch has them

``write_rows`` writes sampled rollouts back into chosen rows: the prompt
right-aligned, the response left-aligned, ``input_ids``, both masks and
``position_ids`` rebuilt, logprobs filled, the advantage broadcast over
the response tokens into ``advantages`` and ``returns``, and the stored
score placed at the last response token of ``token_level_scores`` and
``token_level_rewards``. The batch is padded first when a replayed
sequence is longer than the current width. Nothing here modifies an
input tensor in place.
"""

from __future__ import annotations

import hashlib
import math
from typing import Final, Mapping, NamedTuple, Sequence

import torch
import torch.nn.functional as F

from reservoir.integrations._trl_rows import (
    NEAR_DEAD_ADVANTAGE,
    GroupSpec,
    RowConversion,
    _finite_floats,
    _mask_lengths,
    clamp_logprobs,
    prompt_id_of,
)
from reservoir.rollout import Rollout

Tensors = Mapping[str, torch.Tensor]

REQUIRED_KEYS: Final[tuple[str, ...]] = (
    "prompts", "responses", "input_ids", "attention_mask", "position_ids", "response_mask",
    "old_log_probs", "advantages", "returns", "token_level_scores", "token_level_rewards",
)
"""Every key ``update_actor`` receives from ``fit`` under GRPO; all are rewritten per row."""

REF_KEY: Final[str] = "ref_log_prob"
"""Present iff ``actor.use_kl_loss``; rewritten when present, and every replayed row must carry one."""

PER_TOKEN_KEYS: Final[tuple[str, ...]] = (
    "old_log_probs", "advantages", "returns", "token_level_scores", "token_level_rewards", REF_KEY,
)
"""``(B, Lr)`` float tensors aligned with ``responses``."""

HANDLED_KEYS: Final[frozenset[str]] = frozenset(REQUIRED_KEYS) | {REF_KEY}

TENSOR_DIGEST_KEYS: Final[tuple[str, ...]] = (
    "input_ids", "attention_mask", "position_ids", "response_mask", "advantages", "old_log_probs", REF_KEY,
)
"""Digested in this order when present; see ``tensor_digest``."""


class Layout(NamedTuple):
    """Widths of a checked batch."""

    rows: int
    prompt_len: int
    response_len: int


# ---------------------------------------------------------------------------
# Layout checks
# ---------------------------------------------------------------------------

def check_layout(t: Tensors) -> Layout:
    """Confirm the batch has every key in the shapes above; raise naming what is off.

    ``ValueError`` for a missing key or an inconsistent shape or mask;
    ``NotImplementedError`` for a layout the adapter does not handle
    (3-D ``position_ids``, KL-in-reward, per-token advantages).
    """
    missing = [k for k in REQUIRED_KEYS if k not in t]
    if missing:
        raise ValueError(f"the batch lacks {missing}; is this the DataProto fit hands to update_actor?")
    rows, lp = t["prompts"].shape
    lr = t["responses"].size(1)
    if t["responses"].size(0) != rows:
        raise ValueError(f"prompts has {rows} rows but responses has {t['responses'].size(0)}")
    for key in ("input_ids", "attention_mask"):
        if tuple(t[key].shape) != (rows, lp + lr):
            raise ValueError(f"{key} has shape {tuple(t[key].shape)}, expected {(rows, lp + lr)}")
    if t["position_ids"].dim() != 2:
        raise NotImplementedError(
            f"position_ids has {t['position_ids'].dim()} dimensions; only text models with 2-D "
            "position ids are supported (no multimodal rope)"
        )
    if tuple(t["position_ids"].shape) != (rows, lp + lr):
        raise ValueError(f"position_ids has shape {tuple(t['position_ids'].shape)}, expected {(rows, lp + lr)}")
    for key in PER_TOKEN_KEYS:
        if key in t and tuple(t[key].shape) != (rows, lr):
            raise ValueError(f"{key} has shape {tuple(t[key].shape)}, expected {(rows, lr)}")
    if tuple(t["response_mask"].shape) != (rows, lr):
        raise ValueError(f"response_mask has shape {tuple(t['response_mask'].shape)}, expected {(rows, lr)}")
    if not torch.equal(t["input_ids"], torch.cat([t["prompts"], t["responses"]], dim=1)):
        raise ValueError("input_ids is not prompts followed by responses")
    if not torch.equal(t["attention_mask"][:, lp:].bool(), t["response_mask"].bool()):
        raise ValueError("the response columns of attention_mask differ from response_mask")
    if not torch.equal(t["token_level_rewards"], t["token_level_scores"]):
        raise NotImplementedError(
            "token_level_rewards differs from token_level_scores (algorithm.use_kl_in_reward is on); "
            "replayed rows carry no per-token KL rewards, so this setting is not supported"
        )
    return Layout(rows, lp, lr)


def _row_advantage(advantages: torch.Tensor, r: int, n: int) -> float:
    """The outcome-level advantage of row ``r``; zero for an empty response."""
    if n == 0:
        return 0.0
    values = advantages[r, :n]
    first = float(values[0])
    if not math.isfinite(first):
        raise ValueError(f"row {r}: advantage {first!r} is not finite")
    if bool((values != values[0]).any()):
        raise NotImplementedError(
            f"row {r}: advantages vary across the response tokens; only outcome-level advantages "
            "(algorithm.adv_estimator=grpo) are supported"
        )
    return first


# ---------------------------------------------------------------------------
# Batch -> rollouts
# ---------------------------------------------------------------------------

def _group_rows(uids: Sequence[object]) -> dict[str, list[int]]:
    """Rows per uid, uids in order of first appearance."""
    groups: dict[str, list[int]] = {}
    for r, uid in enumerate(uids):
        groups.setdefault(str(uid), []).append(r)
    return groups


def rows_to_groups(t: Tensors, uids: Sequence[object], *, step: int) -> RowConversion:
    """Turn the batch into ``Rollout`` groups ready for ``add_group``; see the module docstring.

    ``uids`` has one entry per row (``non_tensor_batch["uid"]``). A group
    is dead when every row's advantage is exactly zero (a row with an
    empty response counts as zero); its rows are returned for replacement.
    Rows of live groups with an empty response are skipped and counted.

    Raises
    ------
    ValueError
        A malformed layout or mask, a non-finite value, rows of one uid
        with different prompts, a bad logprob, or ``uids`` of the wrong
        length. ``NotImplementedError`` for the layouts ``check_layout``
        refuses.
    """
    layout = check_layout(t)
    if len(uids) != layout.rows:
        raise ValueError(f"{len(uids)} uids for a batch of {layout.rows} rows")
    prompt_lengths = _mask_lengths(t["attention_mask"][:, :layout.prompt_len], prefix=False, name="prompt attention_mask")
    lengths = _mask_lengths(t["response_mask"], prefix=True, name="response_mask")
    prompts = t["prompts"].tolist()
    unpadded = [row[layout.prompt_len - n:] if n else [] for row, n in zip(prompts, prompt_lengths)]
    advantages = [_row_advantage(t["advantages"], r, n) for r, n in enumerate(lengths)]
    scores = [_finite_floats([s], r, "score")[0] for r, s in enumerate(t["token_level_scores"].sum(dim=1).tolist())]
    responses = t["responses"].tolist()
    logprobs = t["old_log_probs"].tolist()
    refs = t[REF_KEY].tolist() if REF_KEY in t else None

    groups: list[GroupSpec] = []
    dead_rows: list[int] = []
    dead = skipped = clamped = near_dead = 0
    for uid, rows in _group_rows(uids).items():
        if all(advantages[r] == 0.0 for r in rows):
            dead_rows.extend(rows)
            dead += 1
            continue
        if all(abs(advantages[r]) < NEAR_DEAD_ADVANTAGE for r in rows):
            near_dead += 1
        prompt = unpadded[rows[0]]
        rollouts: list[Rollout] = []
        kept: list[int] = []
        for r in rows:
            if unpadded[r] != prompt:
                raise ValueError(f"uid {uid!r}: rows {rows} do not share one prompt")
            n = lengths[r]
            if n == 0:
                skipped += 1
                continue
            lp, n_clamped = clamp_logprobs(logprobs[r][:n], row=r)
            clamped += n_clamped
            metadata: dict = {"prompt_ids": list(prompt), "row": r, "global_step": step, "uid": uid,
                              "score": scores[r]}
            if refs is not None:
                metadata["ref_logprobs"] = _finite_floats(refs[r][:n], r, "ref logprob")
            rollouts.append(Rollout(tokens=responses[r][:n], logprobs=lp, reward=advantages[r], metadata=metadata))
            kept.append(r)
        if rollouts:
            groups.append(GroupSpec(prompt_id_of(prompt), tuple(rollouts), tuple(kept)))
    return RowConversion(tuple(groups), tuple(sorted(dead_rows)), dead, skipped, clamped, near_dead)


# ---------------------------------------------------------------------------
# Rollouts -> batch
# ---------------------------------------------------------------------------

def _response_positions(last_prompt_position: torch.Tensor, response_len: int) -> torch.Tensor:
    """verl's rule: ``last prompt position + 1 .. response_len``, padding included."""
    delta = torch.arange(1, response_len + 1, device=last_prompt_position.device, dtype=last_prompt_position.dtype)
    return last_prompt_position.unsqueeze(-1) + delta


def pad_batch(t: Tensors, target_prompt_len: int, target_response_len: int, pad_token_id: int) -> dict[str, torch.Tensor]:
    """Grow the batch to the target widths: prompts on the left, responses on the right.

    Returns a dict sharing every untouched tensor (``t`` itself as a new
    dict when nothing changes). Shrinking is a ``ValueError``.
    """
    layout = check_layout(t)
    lp, lr = layout.prompt_len, layout.response_len
    if target_prompt_len < lp or target_response_len < lr:
        raise ValueError(
            f"cannot shrink a batch: targets ({target_prompt_len}, {target_response_len}) below current ({lp}, {lr})"
        )
    new = dict(t)
    dp, dr = target_prompt_len - lp, target_response_len - lr
    if dp == 0 and dr == 0:
        return new
    if dp:
        new["prompts"] = F.pad(t["prompts"], (dp, 0), value=pad_token_id)
    if dr:
        new["responses"] = F.pad(t["responses"], (0, dr), value=pad_token_id)
        new["response_mask"] = F.pad(t["response_mask"], (0, dr), value=0)
        for key in PER_TOKEN_KEYS:
            if key in t:
                new[key] = F.pad(t[key], (0, dr), value=0.0)
    prompt_mask = F.pad(t["attention_mask"][:, :lp], (dp, 0), value=0)
    new["attention_mask"] = torch.cat([prompt_mask, new["response_mask"]], dim=1)
    new["input_ids"] = torch.cat([new["prompts"], new["responses"]], dim=1)
    prompt_pos = F.pad(t["position_ids"][:, :lp], (dp, 0), value=0)
    response_pos = t["position_ids"][:, lp:]
    if dr:
        response_pos = torch.cat([response_pos, _response_positions(response_pos[:, -1], dr)], dim=1)
    new["position_ids"] = torch.cat([prompt_pos, response_pos], dim=1)
    return new


def _check_write_args(t: Tensors, rows: tuple[int, ...], rollouts: tuple[Rollout, ...], advantages: tuple[float, ...]) -> bool:
    """Validate ``write_rows`` inputs; return whether ref logprobs are in play."""
    if not (len(rows) == len(rollouts) == len(advantages)):
        raise ValueError(
            f"rows ({len(rows)}), rollouts ({len(rollouts)}) and advantages ({len(advantages)}) must have the same length"
        )
    batch = t["responses"].size(0)
    for r in rows:
        if not 0 <= r < batch:
            raise ValueError(f"row {r} is outside the batch of {batch} rows")
    if len(set(rows)) != len(rows):
        raise ValueError(f"rows contains duplicates: {rows}")
    for key, value in t.items():
        if key not in HANDLED_KEYS and isinstance(value, torch.Tensor) and value.dim() >= 1 and value.size(0) == batch:
            raise ValueError(
                f"batch entry {key!r} has one row per sample and write_rows does not know how to replace its rows"
            )
    has_ref = REF_KEY in t
    for r, rollout in zip(rows, rollouts):
        ref = rollout.metadata.get("ref_logprobs")
        if (ref is not None) != has_ref:
            raise ValueError(
                f"row {r}: the batch {'has' if has_ref else 'has no'} {REF_KEY} but the replayed rollout "
                f"{'has none' if has_ref else 'carries them'}; the buffer was filled under a different KL setting"
            )
        if ref is not None and len(ref) != len(rollout):
            raise ValueError(f"row {r}: ref_logprobs length {len(ref)} != {len(rollout)} tokens")
        for name in ("prompt_ids", "score"):
            if name not in rollout.metadata:
                raise ValueError(f"row {r}: the replayed rollout has no metadata[{name!r}]; was it stored by this adapter?")
    return has_ref


def write_rows(
    t: Tensors, rows: Sequence[int], rollouts: Sequence[Rollout], advantages: Sequence[float], pad_token_id: int,
) -> dict[str, torch.Tensor]:
    """Return a copy of the batch with ``rows[k]`` replaced by ``rollouts[k]``.

    ``advantages[k]`` is written as the row's advantage (the caller has
    already applied any importance-sampling weight), broadcast over the
    response tokens into ``advantages`` and ``returns``. The batch is
    padded first if a replayed prompt or response is longer than the
    current width. ``ref_log_prob`` is written iff the batch has it, and
    every replayed rollout must agree. Any other per-row tensor is a
    ``ValueError``, since its rows would go stale.
    """
    rows_t = tuple(int(r) for r in rows)
    rollouts_t = tuple(rollouts)
    adv_t = tuple(float(a) for a in advantages)
    layout = check_layout(t)
    has_ref = _check_write_args(t, rows_t, rollouts_t, adv_t)
    target_p = max([layout.prompt_len] + [len(r.metadata["prompt_ids"]) for r in rollouts_t])
    target_r = max([layout.response_len] + [len(r) for r in rollouts_t])
    new = pad_batch(t, target_p, target_r, pad_token_id)
    for key in HANDLED_KEYS & set(new):
        new[key] = new[key].clone()
    for r, rollout, adv in zip(rows_t, rollouts_t, adv_t):
        _write_row(new, r, rollout, adv, pad_token_id, has_ref)
    return new


def _write_row(new: dict[str, torch.Tensor], r: int, rollout: Rollout, adv: float, pad_token_id: int, has_ref: bool) -> None:
    """Overwrite row ``r`` of the (already cloned) tensors in ``new``."""
    prompt = rollout.metadata["prompt_ids"]
    lp_r, n = len(prompt), len(rollout)
    lp, lr = new["prompts"].size(1), new["responses"].size(1)

    def as_row(values, like: torch.Tensor) -> torch.Tensor:
        return torch.tensor(values, dtype=like.dtype, device=like.device)

    new["prompts"][r] = pad_token_id
    new["prompts"][r, lp - lp_r:] = as_row(prompt, new["prompts"])
    new["responses"][r] = pad_token_id
    new["responses"][r, :n] = as_row(rollout.tokens, new["responses"])
    new["response_mask"][r] = 0
    new["response_mask"][r, :n] = 1
    new["attention_mask"][r] = 0
    new["attention_mask"][r, lp - lp_r:lp + n] = 1
    new["input_ids"][r] = torch.cat([new["prompts"][r], new["responses"][r]])
    prompt_mask = new["attention_mask"][r, :lp]
    prompt_pos = torch.clip(torch.cumsum(prompt_mask, dim=0) - 1, min=0).to(new["position_ids"].dtype)
    new["position_ids"][r, :lp] = prompt_pos
    new["position_ids"][r, lp:] = _response_positions(prompt_pos[-1], lr)
    for key in ("old_log_probs", "advantages", "returns", "token_level_scores", "token_level_rewards"):
        new[key][r] = 0.0
    new["old_log_probs"][r, :n] = as_row(rollout.logprobs, new["old_log_probs"])
    new["advantages"][r, :n] = adv
    new["returns"][r, :n] = adv
    new["token_level_scores"][r, n - 1] = float(rollout.metadata["score"])
    new["token_level_rewards"][r, n - 1] = float(rollout.metadata["score"])
    if has_ref:
        new[REF_KEY][r] = 0.0
        new[REF_KEY][r, :n] = as_row(rollout.metadata["ref_logprobs"], new[REF_KEY])


def verify_written_rows(t: Tensors, rows: Sequence[int], rollouts: Sequence[Rollout], advantages: Sequence[float]) -> None:
    """Re-read ``rows`` and confirm each holds exactly its rollout; ``ValueError`` otherwise.

    Prompt ids under the prompt columns of ``attention_mask``, response
    ids and behavior logprobs under ``response_mask``, the advantage at
    every response token, ``input_ids`` as the concatenation, and the
    response columns of ``attention_mask`` as ``response_mask``.
    """
    lp = t["prompts"].size(1)
    for r, rollout, adv in zip(rows, rollouts, advantages):
        pmask = t["attention_mask"][r, :lp].bool()
        cmask = t["response_mask"][r].bool()
        if t["prompts"][r][pmask].tolist() != list(rollout.metadata["prompt_ids"]):
            raise ValueError(f"row {r}: written prompt ids differ from the replayed rollout's")
        if t["responses"][r][cmask].tolist() != list(rollout.tokens):
            raise ValueError(f"row {r}: written response ids differ from the replayed rollout's tokens")
        logps = t["old_log_probs"][r][cmask].tolist()
        expected_logps = torch.tensor(list(rollout.logprobs), dtype=t["old_log_probs"].dtype).tolist()
        if logps != expected_logps:
            raise ValueError(f"row {r}: written behavior logprobs differ from the replayed rollout's")
        expected = float(torch.tensor(adv, dtype=t["advantages"].dtype))
        if any(float(a) != expected for a in t["advantages"][r][cmask].tolist()):
            raise ValueError(f"row {r}: written advantages are not {adv} at every response token")
        if not torch.equal(t["input_ids"][r], torch.cat([t["prompts"][r], t["responses"][r]])):
            raise ValueError(f"row {r}: input_ids is not prompts followed by responses")
        if not torch.equal(t["attention_mask"][r, lp:].bool(), cmask):
            raise ValueError(f"row {r}: the response columns of attention_mask differ from response_mask")


# ---------------------------------------------------------------------------
# Commitments and meta
# ---------------------------------------------------------------------------

def tensor_digest(t: Tensors) -> str:
    """BLAKE2b-256 over the final batch tensors the actor consumes.

    Each tensor in ``TENSOR_DIGEST_KEYS`` the batch has is fed as
    ``name|shape|kind`` followed by its values, integer tensors as int64
    and float tensors as float64, little-endian. Per-token values are
    zeroed outside ``response_mask`` and ``position_ids`` outside
    ``attention_mask`` first, so the digest depends on the values the
    model and the loss consume and the padded shape, not on dtype,
    device or padding, and a holder of the batch can recompute it
    without this package.
    """
    h = hashlib.blake2b(digest_size=32, person=b"verl-batch\x00\x00\x00\x00\x00\x00")
    rmask = t["response_mask"].detach().cpu() == 0
    amask = t["attention_mask"].detach().cpu() == 0
    for key in TENSOR_DIGEST_KEYS:
        if key not in t:
            continue
        x = t[key].detach().cpu().contiguous()
        if key in PER_TOKEN_KEYS:
            x = x.masked_fill(rmask, 0.0)
        elif key == "position_ids":
            x = x.masked_fill(amask, 0)
        kind = "f64" if x.is_floating_point() else "i64"
        h.update(f"{key}|{'x'.join(str(d) for d in x.shape)}|{kind}\n".encode())
        x = x.to(torch.float64) if x.is_floating_point() else x.to(torch.int64)
        h.update(x.numpy().astype("<f8" if kind == "f64" else "<i8", copy=False).tobytes())
    return h.hexdigest()


def global_token_num(attention_mask: torch.Tensor) -> list[int]:
    """verl's ``meta_info["global_token_num"]``: valid tokens per row of the final batch."""
    return [int(x) for x in attention_mask.sum(dim=-1).tolist()]


__all__ = [
    "HANDLED_KEYS",
    "Layout",
    "PER_TOKEN_KEYS",
    "REF_KEY",
    "REQUIRED_KEYS",
    "TENSOR_DIGEST_KEYS",
    "check_layout",
    "global_token_num",
    "pad_batch",
    "rows_to_groups",
    "tensor_digest",
    "verify_written_rows",
    "write_rows",
]
