"""
Experiment runner.

Sweeps a grid of boundary policies (plus single-model baselines) over a set of
prompts, scores each generated document with Pangram, and writes a tidy CSV +
per-document provenance JSON for the write-up.

The single-model baselines are essential: your claim is about *rotation*, so you
need each model's solo detection score as the control. If a spliced doc scores
the same as its constituent models solo, rotation did nothing — which is the
result you're predicting.

Run in YOUR environment (needs network + keys). Usage:
    python run_experiment.py
Edit PROMPTS and the SWEEP below to taste.
"""

from __future__ import annotations
import json
import dataclasses
from pathlib import Path

import pandas as pd

from adapters import DEFAULT_ROSTER, Adapter
from boundaries import Boundary
from generate import generate
from score import score_text


PROMPTS = [
    "Write a ~350-word explainer on why sourdough starters need regular feeding.",
    "Write a ~350-word reflection on what makes a city walkable.",
    "Write a ~350-word piece on the tradeoffs of remote work for junior employees.",
]

# (label, boundary, roster, order). Single-model baselines use a 1-model roster
# and a huge interval so no rotation ever fires.
def build_sweep():
    full = DEFAULT_ROSTER
    sweep = []

    # --- rotation conditions ---
    for lo, hi in [(6, 6), (4, 8), (15, 15), (50, 50), (200, 200)]:
        tag = f"words_{lo}" if lo == hi else f"words_{lo}-{hi}"
        sweep.append((f"rotate_{tag}",
                      Boundary("jitter_words", lo, hi), full, "round_robin"))
    for lo, hi in [(6, 6), (4, 8), (15, 15), (50, 50)]:
        tag = f"tok_{lo}" if lo == hi else f"tok_{lo}-{hi}"
        sweep.append((f"rotate_{tag}",
                      Boundary("jitter_tokens", lo, hi), full, "round_robin"))
    sweep.append(("rotate_sentence",
                  Boundary("sentence"), full, "round_robin"))
    sweep.append(("rotate_sentence_2-3",
                  Boundary("sentence_group", 2, 3), full, "round_robin"))
    sweep.append(("rotate_random_order_words_6",
                  Boundary("fixed_words", 6, 6), full, "random"))

    # --- single-model baselines (controls) ---
    for a in full:
        sweep.append((f"baseline_{a.name}",
                      Boundary("fixed_words", 10_000, 10_000), [a], "round_robin"))

    return sweep


def main(out_dir="results", target_ref_tokens=400, reps=2):
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    prov_dir = out / "provenance"
    prov_dir.mkdir(exist_ok=True)

    rows = []
    doc_id = 0
    for label, boundary, roster, order in build_sweep():
        for prompt in PROMPTS:
            for rep in range(reps):
                doc = generate(prompt, roster, boundary,
                               target_ref_tokens=target_ref_tokens, order=order)
                try:
                    s = score_text(doc.text)
                    ai = s["ai_likelihood"]
                except Exception as e:  # keep going; log the failure
                    ai, s = None, {"error": str(e)}

                row = doc.to_row()
                row.update(condition=label, rep=rep, doc_id=doc_id,
                           ai_likelihood=ai)
                rows.append(row)

                (prov_dir / f"doc_{doc_id:04d}.json").write_text(json.dumps({
                    "doc_id": doc_id, "condition": label,
                    "spans": [dataclasses.asdict(sp) for sp in doc.spans],
                    "score": s,
                }, indent=2))
                doc_id += 1
                print(f"[{doc_id}] {label:28s} ai={ai}")

    df = pd.DataFrame(rows)
    df.to_csv(out / "results.csv", index=False)

    # quick summary: mean AI likelihood per condition
    summary = (df.dropna(subset=["ai_likelihood"])
                 .groupby("condition")["ai_likelihood"]
                 .agg(["mean", "std", "count"])
                 .sort_values("mean"))
    summary.to_csv(out / "summary.csv")
    print("\n=== mean AI likelihood by condition (low = evaded more) ===")
    print(summary.to_string())


if __name__ == "__main__":
    main()
