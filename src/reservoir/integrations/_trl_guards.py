"""What the TRL adapter refuses, and the one vLLM key it drops instead.

``check_batch`` runs on every rank before any collective and before the
buffer is touched; ``refuse_sharded`` runs when an accelerator is attached.
"""

from __future__ import annotations

from typing import Any, Final

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

SAMPLING_LOGPROBS_KEY: Final[str] = "sampling_per_token_logps"
"""vLLM's sampling logprobs; dropped when TRL would not use them, refused otherwise."""

SHARDED_STRATEGIES: Final[tuple[str, ...]] = ("DEEPSPEED", "FSDP", "MEGATRON_LM")
"""Accelerate distributed types the adapter refuses: the owner-only telemetry and drift-gate forward
needs the whole model on rank 0, which a sharded strategy does not provide (only DDP is verified)."""


def refuse_sharded(accelerator: Any, num_processes: int) -> None:
    """Raise if ``accelerator`` shards parameters across more than one process."""
    kind = getattr(accelerator, "distributed_type", None)
    name = getattr(kind, "name", None) or (str(kind) if kind is not None else "")
    if num_processes > 1 and any(tag in name.upper() for tag in SHARDED_STRATEGIES):
        raise RuntimeError(
            f"ReservoirReplay does not support the {name} strategy: rank 0 alone runs the telemetry and "
            "drift-gate forward over the gathered batch, which under sharded parameters would wait on an "
            "all-gather the other ranks never join. Use DDP (the verified configuration)."
        )


def drop_unused_sampling_logprobs(output: dict, trainer: Any) -> dict:
    """Remove vLLM's sampling logprobs when nothing downstream reads them.

    TRL 1.13.0 consumes ``sampling_per_token_logps`` only for the vLLM
    importance-sampling correction and for off-policy masking. With
    both off, the key is dead weight that would otherwise make the
    adapter refuse the batch. Returns ``output`` itself when there is
    nothing to drop, a new dict without the key otherwise.
    """
    if SAMPLING_LOGPROBS_KEY not in output:
        return output
    correction = bool(getattr(trainer, "vllm_importance_sampling_correction", False))
    masking = getattr(trainer, "off_policy_mask_threshold", None) is not None
    if correction or masking or "importance_sampling_ratio" in output:
        return output  # check_batch refuses it with the usual message
    return {k: v for k, v in output.items() if k != SAMPLING_LOGPROBS_KEY}


def check_batch(output: dict, trainer: Any, beta: float) -> None:
    """Refuse configurations the adapter does not handle, before touching anything."""
    if getattr(trainer, "sync_ref_model", False) or getattr(getattr(trainer, "args", None), "sync_ref_model", False):
        raise ValueError("ReservoirReplay cannot use sync_ref_model: stored reference logprobs would become stale")
    if beta > 0.0 and getattr(trainer, "loss_type", None) == "vespo":
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


def dropout_without_old_logprobs(output: dict, trainer: Any) -> bool:
    """Whether active model dropout can make fresh-row behavior logprobs differ from the loss."""
    if "old_per_token_logps" in output or getattr(getattr(trainer, "args", None), "disable_dropout", False):
        return False
    model = getattr(trainer, "model", None)
    if model is None or not getattr(model, "training", False):
        return False
    modules = getattr(model, "modules", None)
    if callable(modules):
        for module in modules():
            if "dropout" in type(module).__name__.lower() and float(getattr(module, "p", 0)) > 0:
                return True
    config = getattr(model, "config", None)
    return any(float(getattr(config, key, 0) or 0) > 0 for key in
               ("attention_dropout", "hidden_dropout_prob", "dropout", "dropout_rate"))
