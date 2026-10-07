"""GSM8K prompts, the exact-match reward and a greedy held-out evaluation.

The staleness sweep trains on the GSM8K train split with one reward: 1.0
when the number after the ``####`` marker (or, failing that, the last
number in the completion) equals the gold answer, else 0.0. The same
function scores the greedy held-out evaluation at the end of a run.

``even_length_reward`` is the CPU smoke reward from
``benchmarks/modal/trl_replay_real.py``: a near-coin-flip that gives the
tiny test model live and dead groups so replay is exercised without a GPU.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation
from typing import Any, Callable, Optional

DATASET_ID = "openai/gsm8k"
DATASET_CONFIG = "main"
MARKER = "####"
SYSTEM_PROMPT = (
    "Solve the math problem. Reason step by step, then write the final numeric answer "
    f"on its own last line as '{MARKER} <number>'."
)
_NUMBER = re.compile(r"(?<!\d)-?\d[\d,]*(?:\.\d+)?")

RewardFn = Callable[..., list[float]]


def completion_text(completion: Any) -> str:
    """The text of a TRL completion: the last message's content when conversational, else ``str``."""
    if isinstance(completion, list):
        return str(completion[-1].get("content", "")) if completion else ""
    return str(completion)


def extract_answer(text: str) -> Optional[str]:
    """The number after the last ``####`` marker, else the last number in ``text``; commas removed."""
    if MARKER in text:
        match = _NUMBER.search(text.rsplit(MARKER, 1)[1])
        if match is not None:
            return match.group(0).replace(",", "")
    numbers = _NUMBER.findall(text)
    return numbers[-1].replace(",", "") if numbers else None


def answers_match(predicted: Optional[str], gold: Optional[str]) -> bool:
    """Numeric equality of two extracted answers; anything unparsable is a mismatch."""
    if predicted is None or gold is None:
        return False
    try:
        return Decimal(predicted) == Decimal(gold)
    except InvalidOperation:
        return False


def gold_answer(answer_field: str) -> Optional[str]:
    """GSM8K's gold answer: the number after ``####`` in the ``answer`` column."""
    return extract_answer(answer_field)


def gsm8k_reward(completions: list, answer: list, **kwargs: Any) -> list[float]:
    """1.0 when the completion's extracted answer equals the gold ``answer``, else 0.0."""
    return [1.0 if answers_match(extract_answer(completion_text(c)), g) else 0.0 for c, g in zip(completions, answer)]


def even_length_reward(completions: list, **kwargs: Any) -> list[float]:
    """1.0 when the completion text has an even number of characters (the CPU smoke reward)."""
    return [1.0 if len(completion_text(c)) % 2 == 0 else 0.0 for c in completions]


REWARDS: dict[str, RewardFn] = {"gsm8k": gsm8k_reward, "even_length": even_length_reward}


def build_dataset(split: str, size: Optional[int], seed: int):
    """GSM8K ``split`` as conversational prompts with the gold number in ``answer``; ``size`` rows after a seeded shuffle."""
    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split=split)
    if size is not None:
        dataset = dataset.shuffle(seed=seed).select(range(min(size, len(dataset))))

    def convert(row: dict) -> dict:
        return {
            "prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": row["question"]}],
            "answer": gold_answer(row["answer"]),
        }

    return dataset.map(convert, remove_columns=["question"])


def evaluate_greedy(model: Any, tokenizer: Any, dataset: Any, *, max_new_tokens: int, batch_size: int,
                    reward_fn: RewardFn = gsm8k_reward) -> dict:
    """Greedy accuracy of ``model`` on ``dataset`` under ``reward_fn`` (HF ``generate``, left padding)."""
    import torch

    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    padding_side, tokenizer.padding_side = tokenizer.padding_side, "left"
    correct, n = 0.0, 0
    try:
        for start in range(0, len(dataset), batch_size):
            rows = dataset[start:start + batch_size]
            texts = [tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in rows["prompt"]]
            encoded = tokenizer(texts, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                generated = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False,
                                           pad_token_id=tokenizer.pad_token_id)
            completions = tokenizer.batch_decode(generated[:, encoded["input_ids"].size(1):], skip_special_tokens=True)
            scores = reward_fn(completions, answer=rows["answer"])
            correct += sum(scores)
            n += len(scores)
    finally:
        tokenizer.padding_side = padding_side
        model.train(was_training)
    return {"accuracy": correct / n if n else 0.0, "correct": correct, "n": n, "max_new_tokens": max_new_tokens,
            "decoding": "greedy", "reward": getattr(reward_fn, "__name__", str(reward_fn))}
