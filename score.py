"""
Thin Pangram scoring wrapper.

Set PANGRAM_API_KEY in your environment. Endpoint/field names are centralized
here so if Pangram's schema differs from the defaults you only edit one place.
Check their current API docs and adjust ENDPOINT / the response parsing to match.

This sandbox has no network access, so scoring runs on YOUR machine.
"""

from __future__ import annotations
import os
import time
import requests

ENDPOINT = os.environ.get("PANGRAM_ENDPOINT", "https://text.api.pangram.com/")


def _first_present(data: dict, keys: tuple[str, ...]):
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def score_text(text: str, max_retries: int = 4) -> dict:
    """Return a normalized dict: {ai_likelihood: float, raw: <full response>}."""
    key = os.environ["PANGRAM_API_KEY"]
    headers = {"Content-Type": "application/json", "x-api-key": key}
    payload = {"text": text}
    for attempt in range(max_retries):
        try:
            r = requests.post(ENDPOINT, json=payload, headers=headers, timeout=60)
            if r.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            data = r.json()
            # Adjust these keys to match Pangram's live schema:
            likelihood = _first_present(
                data,
                ("ai_likelihood", "likelihood_ai", "score"),
            )
            if likelihood is not None:
                likelihood = float(likelihood)
            return {"ai_likelihood": likelihood, "raw": data}
        except requests.RequestException:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError("unreachable")
