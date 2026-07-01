"""
Model adapters for the rotation harness.

Each adapter exposes a single method:

    continue_text(prompt: str, context: str, max_new_tokens: int) -> str

where `context` is the accumulated text so far. The adapter must ask the model
to CONTINUE `context` (not restart), so the spliced document stays coherent and
any Pangram detection is a real detection rather than an artifact of broken text.

All boundary counting is done OUTSIDE the adapter using a single reference
tokenizer (see boundaries.py), so provider tokenizer differences never affect
the rotation interval. Adapters are asked to overproduce; the runner truncates
to the exact boundary.

Credentials: set the relevant env vars in YOUR environment. This sandbox has no
network, so nothing here calls out from Claude's side.
    OPENAI_API_KEY, GOOGLE_API_KEY, ANTHROPIC_API_KEY
Open models are routed through whatever OpenAI-compatible endpoint you point
OPEN_BASE_URL / OPEN_API_KEY at (e.g. a local vLLM / Ollama / Together server).
"""

from __future__ import annotations
import os
from dataclasses import dataclass


CONTINUE_SYSTEM = (
    "You are continuing a piece of writing. Continue directly from where the "
    "text stops, matching its topic, register, and voice. Do not repeat prior "
    "text, do not add headings, meta-commentary, or quotation marks around your "
    "continuation. Just produce the next passage of prose."
)


def _continue_user(prompt: str, context: str) -> str:
    if not context.strip():
        return (
            f"Write a piece responding to the following request. "
            f"Begin the prose now.\n\nREQUEST:\n{prompt}"
        )
    return (
        f"The overall writing task is:\n\nREQUEST:\n{prompt}\n\n"
        f"Here is the text so far. Continue it seamlessly:\n\n{context}"
    )


@dataclass
class Adapter:
    name: str            # label used in the provenance log
    provider: str        # openai | google | anthropic | open
    model: str           # provider-specific model id

    def continue_text(self, prompt: str, context: str, max_new_tokens: int) -> str:
        if self.provider == "openai":
            return self._openai(prompt, context, max_new_tokens)
        if self.provider == "anthropic":
            return self._anthropic(prompt, context, max_new_tokens)
        if self.provider == "google":
            return self._google(prompt, context, max_new_tokens)
        if self.provider == "open":
            return self._openai(prompt, context, max_new_tokens, use_open=True)
        raise ValueError(f"unknown provider {self.provider!r}")

    # --- OpenAI / OpenAI-compatible (also used for open models) -------------
    def _openai(self, prompt, context, max_new_tokens, use_open=False):
        from openai import OpenAI
        if use_open:
            client = OpenAI(
                api_key=os.environ.get("OPEN_API_KEY", "x"),
                base_url=os.environ["OPEN_BASE_URL"],
            )
        else:
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        # overproduce; runner truncates to the reference-tokenizer boundary
        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": CONTINUE_SYSTEM},
                {"role": "user", "content": _continue_user(prompt, context)},
            ],
            max_tokens=int(max_new_tokens * 2.5) + 32,
            temperature=1.0,
        )
        return resp.choices[0].message.content or ""

    # --- Anthropic ----------------------------------------------------------
    def _anthropic(self, prompt, context, max_new_tokens):
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        resp = client.messages.create(
            model=self.model,
            max_tokens=int(max_new_tokens * 2.5) + 32,
            system=CONTINUE_SYSTEM,
            messages=[{"role": "user", "content": _continue_user(prompt, context)}],
            temperature=1.0,
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")

    # --- Google Gemini ------------------------------------------------------
    def _google(self, prompt, context, max_new_tokens):
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])
        resp = client.models.generate_content(
            model=self.model,
            contents=_continue_user(prompt, context),
            config=types.GenerateContentConfig(
                system_instruction=CONTINUE_SYSTEM,
                max_output_tokens=int(max_new_tokens * 2.5) + 32,
                temperature=1.0,
            ),
        )
        return resp.text or ""


# Edit this roster to match what you have keys for. `name` is what shows up in
# the provenance log and charts, so make it human-readable and version-specific.
DEFAULT_ROSTER = [
    Adapter("gpt-4o",            "openai",    "gpt-4o"),
    Adapter("gpt-4.1",           "openai",    "gpt-4.1"),
    Adapter("gemini-2.5-pro",    "google",    "gemini-2.5-pro"),
    Adapter("claude-opus-4.6",   "anthropic", "claude-opus-4-6"),
    Adapter("claude-sonnet-4.6", "anthropic", "claude-sonnet-4-6"),
    # Adapter("llama-3.3-70b",   "open",      "meta-llama/Llama-3.3-70B-Instruct"),
]
