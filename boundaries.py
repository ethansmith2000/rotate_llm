"""
Boundary policies decide WHERE one model hands off to the next.

Critical design point: for token modes we count with ONE reference tokenizer
(tiktoken) regardless of which model is generating. If we counted with each
provider's own tokenizer, "8 tokens" would mean a different span at every
handoff and would confound the experiment. The reference tokenizer is a fixed
yardstick; it does not need to match any model's real tokenizer.

A policy takes the FULL accumulated text plus the newly generated continuation
and returns how much of the continuation to KEEP before switching models. The
runner appends only the kept slice, then rotates.

Modes:
    fixed_words      keep exactly N words
    jitter_words     keep uniform(min,max) words
    fixed_tokens     keep exactly N reference-tokens
    jitter_tokens    keep uniform(min,max) reference-tokens
    sentence         keep 1 sentence
    sentence_group   keep uniform(min,max) sentences
"""

from __future__ import annotations
import random
import re
from dataclasses import dataclass

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")  # the fixed reference yardstick
except Exception:  # pragma: no cover
    _ENC = None

_SENT_SPLIT = re.compile(r"(?<=[.!?])[\"')\]]?\s+")


def _ref_token_prefix(text: str, n: int) -> str:
    """First n reference-tokens of text, decoded back to a string."""
    if _ENC is None:
        # fallback: ~4 chars/token approximation if tiktoken unavailable
        return text[: n * 4]
    toks = _ENC.encode(text)
    return _ENC.decode(toks[:n])


def _word_prefix(text: str, n: int) -> str:
    parts = re.split(r"(\s+)", text)  # keep separators so we can rejoin cleanly
    out, count = [], 0
    for p in parts:
        out.append(p)
        if p.strip():
            count += 1
        if count >= n:
            break
    return "".join(out)


def _sentence_prefix(text: str, n: int) -> str:
    pieces = _SENT_SPLIT.split(text)
    if len(pieces) <= n:
        return text
    # rejoin first n sentences with their trailing whitespace best-effort
    # (split consumed the delimiter; re-add a single space, good enough for prose)
    return " ".join(pieces[:n]).strip()


@dataclass
class Boundary:
    mode: str
    lo: int = 6
    hi: int = 6  # hi==lo => fixed

    def draw(self) -> int:
        return self.lo if self.lo >= self.hi else random.randint(self.lo, self.hi)

    def keep(self, continuation: str, k: int | None = None) -> str:
        k = self.draw() if k is None else k
        if self.mode in ("fixed_words", "jitter_words"):
            return _word_prefix(continuation, k)
        if self.mode in ("fixed_tokens", "jitter_tokens"):
            return _ref_token_prefix(continuation, k)
        if self.mode == "sentence":
            return _sentence_prefix(continuation, 1)
        if self.mode == "sentence_group":
            return _sentence_prefix(continuation, k)
        raise ValueError(f"unknown boundary mode {self.mode!r}")


def ref_token_len(text: str) -> int:
    if _ENC is None:
        return max(1, len(text) // 4)
    return len(_ENC.encode(text))
