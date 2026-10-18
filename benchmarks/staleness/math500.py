"""MATH500 prompts and a self-contained answer check, the sweep's out-of-distribution evaluation.

The gold answers (``HuggingFaceH4/MATH-500``) are LaTeX. A completion's
answer is the content of its last ``\\boxed{...}``, else the text after a
``####`` marker (the GSM8K format the policy is trained on), else its last
number. Two answers match when their normalized strings are equal or both
parse as the same exact rational. The normalization (drop ``\\left``/
``\\right``, spacing commands, ``\\text{}``, ``^\\circ``, ``\\%``, a leading
``x =``; ``\\dfrac``/``\\tfrac`` to ``\\frac``) is narrower than Math-Verify's
symbolic equivalence, so it can only undercount: an answer it accepts is
correct, one it rejects may still be equivalent. Every arm is scored by
the same function, so the comparison across arms is fair.
"""

from __future__ import annotations

import re
from fractions import Fraction
from typing import Any, Optional

from benchmarks.staleness.gsm8k import completion_text

DATASET_ID = "HuggingFaceH4/MATH-500"
SYSTEM_PROMPT = "Solve the problem. Reason step by step, then put the final answer in \\boxed{}."
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
_FRAC = re.compile(r"^(-?)\\frac\{?(-?\d+)\}?\{?(-?\d+)\}?$")


def _last_boxed(text: str) -> Optional[str]:
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    depth, i = 0, start + len("\\boxed")
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None


def extract_answer(text: str) -> Optional[str]:
    boxed = _last_boxed(text)
    if boxed is not None:
        return boxed.strip()
    if "####" in text:
        tail = text.rsplit("####", 1)[1].strip().splitlines()
        if tail and tail[0].strip():
            return tail[0].strip()
    numbers = _NUMBER.findall(text)
    return numbers[-1] if numbers else None


def _unwrap_text(s: str) -> str:
    return re.sub(r"\\(?:text|textbf|mathrm|mbox)\{([^{}]*)\}", r"\1", s)


def normalize(answer: str) -> str:
    s = _unwrap_text(answer.strip())
    for token in ("\\left", "\\right", "\\!", "\\,", "\\;", "\\:", "\\ ", "$", "\\%", "%"):
        s = s.replace(token, "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = re.sub(r"\^\{?\\circ\}?", "", s)
    s = re.sub(r"\s+", "", s).rstrip(".")
    s = re.sub(r"^[a-zA-Z]=", "", s)
    if re.fullmatch(r"-?\d{1,3}(,\d{3})+(\.\d+)?", s):
        s = s.replace(",", "")
    return s


def _as_fraction(s: str) -> Optional[Fraction]:
    match = _FRAC.match(s)
    try:
        if match:
            sign = -1 if match.group(1) else 1
            return sign * Fraction(int(match.group(2)), int(match.group(3)))
        if re.fullmatch(r"-?\d+(\.\d+)?(/-?\d+)?", s):
            return Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None
    return None


def answers_match(predicted: Optional[str], gold: Optional[str]) -> bool:
    if not predicted or not gold:
        return False
    p, g = normalize(predicted), normalize(gold)
    if not p:
        return False
    if p == g:
        return True
    pf, gf = _as_fraction(p), _as_fraction(g)
    return pf is not None and pf == gf


def math_reward(completions: list, answer: list, **kwargs: Any) -> list[float]:
    """1.0 when the completion's answer matches the gold ``answer``, else 0.0."""
    return [1.0 if answers_match(extract_answer(completion_text(c)), g) else 0.0 for c, g in zip(completions, answer)]


def build_dataset(size: Optional[int], seed: int):
    """MATH500 as conversational prompts with the gold LaTeX answer in ``answer``; all 500 when ``size`` is None."""
    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, split="test")
    if size is not None:
        dataset = dataset.shuffle(seed=seed).select(range(min(size, len(dataset))))

    def convert(row: dict) -> dict:
        return {"prompt": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": row["problem"]}],
                "answer": row["answer"]}

    return dataset.map(convert, remove_columns=[c for c in dataset.column_names if c != "answer"])
