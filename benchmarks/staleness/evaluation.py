"""Held-out evaluation for the staleness sweep: avg@k during training, greedy pass@1 and MATH500 at the end.

On the GPU the colocated vLLM engine answers, after the trainer's current
weights are pushed to it (``sync_weights``), which is far faster than HF
``generate`` for 1,319 problems; the engine serves bfloat16, the trainer
holds float32. Without vLLM (the CPU smoke) HF ``generate`` answers, inside
a forked RNG so evaluating never moves the trainer's random stream. Each
item's score is the mean reward over its ``n`` samples, kept per item so
the report can bootstrap over seeds × items.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

RewardFn = Callable[..., list[float]]


def _chat_texts(tokenizer: Any, prompts: list) -> list[str]:
    return [tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]


def _generate_vllm(trainer: Any, prompts: list, *, n: int, temperature: float, max_new_tokens: int,
                   seed: int) -> list[list[str]]:
    from vllm import SamplingParams

    engine = trainer.vllm_generation
    if getattr(engine, "enable_sleep_mode", False):
        raise RuntimeError("evaluation through the colocated engine does not handle vLLM sleep mode")
    engine.sync_weights()
    trainer._last_loaded_step = trainer.state.global_step   # the next generation must not re-sync needlessly
    tokenizer = trainer.processing_class
    ids = [tokenizer(text, add_special_tokens=False)["input_ids"] for text in _chat_texts(tokenizer, prompts)]
    params = SamplingParams(n=n, temperature=temperature, max_tokens=max_new_tokens, seed=seed)
    outputs = engine.llm.generate([{"prompt_token_ids": i} for i in ids], sampling_params=params, use_tqdm=False)
    return [[o.text for o in out.outputs] for out in outputs]


def _generate_hf(trainer: Any, prompts: list, *, n: int, temperature: float, max_new_tokens: int, seed: int,
                 batch_size: int) -> list[list[str]]:
    import torch

    model = trainer.accelerator.unwrap_model(trainer.model)
    tokenizer = trainer.processing_class
    device = next(model.parameters()).device
    was_training = model.training
    padding_side, tokenizer.padding_side = tokenizer.padding_side, "left"
    sample = temperature > 0
    out: list[list[str]] = []
    model.eval()
    try:
        with torch.random.fork_rng(devices=[device.index or 0] if device.type == "cuda" else []):
            torch.manual_seed(seed)
            for start in range(0, len(prompts), batch_size):
                encoded = tokenizer(_chat_texts(tokenizer, prompts[start:start + batch_size]), return_tensors="pt",
                                    padding=True).to(device)
                with torch.no_grad():
                    generated = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=sample,
                                               temperature=temperature if sample else None, top_p=1.0 if sample else None,
                                               top_k=0 if sample else None, num_return_sequences=n,
                                               pad_token_id=tokenizer.pad_token_id)
                texts = tokenizer.batch_decode(generated[:, encoded["input_ids"].size(1):], skip_special_tokens=True)
                out.extend(texts[i:i + n] for i in range(0, len(texts), n))
    finally:
        tokenizer.padding_side = padding_side
        model.train(was_training)
    return out


def generate_texts(trainer: Any, prompts: list, *, n: int, temperature: float, max_new_tokens: int, seed: int,
                   batch_size: int) -> list[list[str]]:
    """``n`` completions per prompt from the trainer's current policy (temperature 0 is greedy)."""
    if getattr(trainer, "use_vllm", False):
        return _generate_vllm(trainer, prompts, n=n, temperature=temperature, max_new_tokens=max_new_tokens, seed=seed)
    return _generate_hf(trainer, prompts, n=n, temperature=temperature, max_new_tokens=max_new_tokens, seed=seed,
                        batch_size=batch_size)


def evaluate(trainer: Any, dataset: Any, reward_fn: RewardFn, *, n: int, temperature: float, max_new_tokens: int,
             seed: int, batch_size: int, step: Optional[int] = None) -> dict:
    """Mean reward over ``n`` samples per item, per item and overall."""
    prompts, answers = list(dataset["prompt"]), list(dataset["answer"])
    completions = generate_texts(trainer, prompts, n=n, temperature=temperature, max_new_tokens=max_new_tokens,
                                 seed=seed, batch_size=batch_size)
    per_item = []
    for texts, gold in zip(completions, answers, strict=True):
        scores = reward_fn(texts, answer=[gold] * len(texts))
        per_item.append(sum(scores) / len(scores))
    return {
        "step": step, "accuracy": sum(per_item) / len(per_item) if per_item else 0.0, "n": len(per_item), "k": n,
        "temperature": temperature, "max_new_tokens": max_new_tokens, "seed": seed,
        "decoding": "greedy" if temperature == 0 else "sample", "backend": "vllm" if getattr(trainer, "use_vllm", False) else "hf",
        "reward": getattr(reward_fn, "__name__", str(reward_fn)), "per_item": per_item,
    }


def fit_prompts(dataset: Any, tokenizer: Any, max_prompt_tokens: int) -> tuple[Any, int]:
    """``dataset`` without the prompts longer than ``max_prompt_tokens``, and how many were dropped.

    vLLM rejects a whole ``generate`` call when one prompt plus its token
    budget exceeds ``max_model_len``; dropping the few over-length items
    (the same ones for every arm) keeps the rest of the evaluation.
    """
    lengths = [len(tokenizer(text, add_special_tokens=False)["input_ids"])
               for text in _chat_texts(tokenizer, list(dataset["prompt"]))]
    keep = [i for i, n in enumerate(lengths) if n <= max_prompt_tokens]
    return (dataset if len(keep) == len(lengths) else dataset.select(keep)), len(lengths) - len(keep)


def guarded(run: Callable[[], dict], **context: Any) -> dict:
    """``run()`` with its wall clock, or ``{"error": ...}``: an evaluation failure must not lose a trained run."""
    started = time.time()
    try:
        out = run()
    except Exception as exc:  # recorded in the run record; the report lists evaluations that failed
        return {**context, "error": f"{type(exc).__name__}: {exc}", "seconds": time.time() - started}
    return {**out, "seconds": time.time() - started}


def make_periodic_eval(trainer: Any, dataset: Any, reward_fn: RewardFn, *, every: int, n: int, max_new_tokens: int,
                       seed: int, batch_size: int, sink: list) -> Any:
    """A ``TrainerCallback`` appending ``evaluate(..., temperature=1.0)`` to ``sink`` every ``every`` steps."""
    from transformers import TrainerCallback

    class PeriodicEval(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if every > 0 and state.global_step % every == 0:
                sink.append(guarded(lambda: evaluate(
                    trainer, dataset, reward_fn, n=n, temperature=1.0, max_new_tokens=max_new_tokens,
                    seed=seed + state.global_step, batch_size=batch_size, step=state.global_step), step=state.global_step))
            return control

    return PeriodicEval()
