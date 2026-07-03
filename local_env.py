"""
Local credential loading.

The committed code reads environment variables. For local convenience, an ignored
keys.py file may define API_KEYS = {"NAME": "value"}; this module copies those
values into os.environ without printing them.
"""

from __future__ import annotations

import os

PLACEHOLDERS = {"", "YOUR_KEY_HERE"}
FIREWORKS_BASE_URL = "https://api.fireworks.ai/inference/v1"


def load_local_keys() -> list[str]:
    """Load non-placeholder values from ignored keys.py into the environment."""
    try:
        from keys import API_KEYS
    except ModuleNotFoundError:
        return []

    if not isinstance(API_KEYS, dict):
        raise TypeError("keys.py must define API_KEYS as a dict")

    loaded = []
    for name, value in API_KEYS.items():
        if value is None:
            continue
        value = str(value).strip()
        if value in PLACEHOLDERS:
            continue
        os.environ.setdefault(name, value)
        loaded.append(name)

    if os.environ.get("FIREWORKS_API_KEY") and not os.environ.get("OPEN_API_KEY"):
        os.environ["OPEN_API_KEY"] = os.environ["FIREWORKS_API_KEY"]
        loaded.append("OPEN_API_KEY")
    if os.environ.get("FIREWORKS_API_KEY") and not os.environ.get("OPEN_BASE_URL"):
        os.environ["OPEN_BASE_URL"] = FIREWORKS_BASE_URL
        loaded.append("OPEN_BASE_URL")

    return loaded
