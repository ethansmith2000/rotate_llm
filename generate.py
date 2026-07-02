"""
Generation loop: rotate models along a boundary policy, preserving coherence by
feeding accumulated text as context each turn. Emits both the final document and
a provenance log (which model produced which character span) — the provenance is
the analytically interesting artifact: it lets you correlate detection with
handoff density and with specific models.
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field

from adapters import Adapter
from boundaries import Boundary, ref_token_len


@dataclass
class Span:
    model: str
    start: int
    end: int
    text: str
    boundary_n: int


@dataclass
class Document:
    prompt: str
    text: str
    spans: list[Span] = field(default_factory=list)
    boundary_mode: str = ""
    boundary_lo: int = 0
    boundary_hi: int = 0
    roster: list[str] = field(default_factory=list)
    target_ref_tokens: int = 0

    @property
    def n_handoffs(self) -> int:
        return sum(
            1
            for prev, cur in zip(self.spans, self.spans[1:])
            if prev.model != cur.model
        )

    def to_row(self) -> dict:
        return {
            "prompt": self.prompt[:120],
            "boundary_mode": self.boundary_mode,
            "boundary_lo": self.boundary_lo,
            "boundary_hi": self.boundary_hi,
            "roster": ",".join(self.roster),
            "n_handoffs": self.n_handoffs,
            "ref_tokens": ref_token_len(self.text),
            "text": self.text,
        }


def _next_model(
    roster: list[Adapter],
    i: int,
    order: str,
    previous: Adapter | None = None,
) -> Adapter:
    if order == "round_robin":
        return roster[i % len(roster)]
    if order == "random":
        choices = [a for a in roster if a is not previous]
        return random.choice(choices or roster)
    raise ValueError(f"unknown model order {order!r}")


def _request_budget(boundary: Boundary, boundary_n: int, remaining_ref_tokens: int) -> int:
    if boundary.mode in ("fixed_tokens", "jitter_tokens"):
        return max(1, min(boundary_n + 4, remaining_ref_tokens + 4))
    return max(1, min(boundary_n + 4, remaining_ref_tokens + 32))


def generate(
    prompt: str,
    roster: list[Adapter],
    boundary: Boundary,
    target_ref_tokens: int = 400,
    order: str = "round_robin",  # or "random"
    max_turns: int = 200,
) -> Document:
    text = ""
    spans: list[Span] = []
    turn = 0
    previous_model: Adapter | None = None
    while ref_token_len(text) < target_ref_tokens and turn < max_turns:
        remaining = max(1, target_ref_tokens - ref_token_len(text))
        boundary_n = boundary.draw()
        if boundary.mode in ("fixed_tokens", "jitter_tokens"):
            boundary_n = min(boundary_n, remaining)

        model = _next_model(roster, turn, order, previous_model)
        cont = model.continue_text(
            prompt,
            text,
            max_new_tokens=_request_budget(boundary, boundary_n, remaining),
        )
        kept = boundary.keep(cont, boundary_n).strip()
        if not kept:
            turn += 1
            continue
        sep = "" if (not text or text.endswith(("\n", " "))) else " "
        start = len(text) + len(sep)
        text = text + sep + kept
        spans.append(Span(model.name, start, len(text), kept, boundary_n))
        previous_model = model
        turn += 1

    return Document(
        prompt=prompt,
        text=text.strip(),
        spans=spans,
        boundary_mode=boundary.mode,
        boundary_lo=boundary.lo,
        boundary_hi=boundary.hi,
        roster=[a.name for a in roster],
        target_ref_tokens=target_ref_tokens,
    )
