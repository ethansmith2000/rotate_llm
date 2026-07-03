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
    python run_experiment.py --smoke
Edit PROMPTS and the SWEEP below to taste.
"""

from __future__ import annotations
import argparse
import csv
import json
import dataclasses
import os
import statistics
from pathlib import Path

from adapters import DEFAULT_ROSTER, Adapter
from boundaries import Boundary
from generate import generate
from local_env import load_local_keys
from score import score_text

load_local_keys()


PROVIDER_REQUIREMENTS = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "google": ("GOOGLE_API_KEY",),
    "open": ("OPEN_BASE_URL", "OPEN_API_KEY"),
}

PROMPTS = [
    "Write a ~350-word explainer on why sourdough starters need regular feeding.",
    "Write a ~350-word reflection on what makes a city walkable.",
    "Write a ~350-word piece on the tradeoffs of remote work for junior employees.",
]


def _available_roster(roster: list[Adapter]) -> tuple[list[Adapter], list[tuple[Adapter, list[str]]]]:
    available = []
    skipped = []
    for adapter in roster:
        required = PROVIDER_REQUIREMENTS.get(adapter.provider)
        if required is None:
            skipped.append((adapter, [f"unknown provider {adapter.provider!r}"]))
            continue
        missing = [name for name in required if not os.environ.get(name)]
        if missing:
            skipped.append((adapter, missing))
        else:
            available.append(adapter)
    return available, skipped


def _print_roster_status(roster: list[Adapter], skipped: list[tuple[Adapter, list[str]]]):
    print("Using roster:", ", ".join(a.name for a in roster) or "(none)")
    for adapter, missing in skipped:
        print(f"Skipping {adapter.name}: missing {', '.join(missing)}")


def _write_results_csv(rows: list[dict], path: Path):
    if not rows:
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _summary_rows(rows: list[dict]) -> list[dict]:
    by_condition: dict[str, list[float]] = {}
    for row in rows:
        ai = row.get("ai_likelihood")
        if ai is None:
            continue
        by_condition.setdefault(row["condition"], []).append(float(ai))

    summary = []
    for condition, values in by_condition.items():
        summary.append({
            "condition": condition,
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else None,
            "count": len(values),
        })
    return sorted(summary, key=lambda row: row["mean"])


def _write_summary_csv(summary: list[dict], path: Path):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "mean", "std", "count"])
        writer.writeheader()
        writer.writerows(summary)


def _print_summary(summary: list[dict]):
    print("\n=== mean AI likelihood by condition (low = evaded more) ===")
    if not summary:
        print("(no successful scores)")
        return
    print(f"{'condition':32s} {'mean':>10s} {'std':>10s} {'count':>5s}")
    for row in summary:
        std = "" if row["std"] is None else f"{row['std']:.6f}"
        print(f"{row['condition']:32s} {row['mean']:10.6f} {std:>10s} {row['count']:5d}")


# (label, boundary, roster, order). Single-model baselines use a 1-model roster
# and one document-sized token boundary so no rotation fires.
def build_sweep(target_ref_tokens=400, smoke=False, roster: list[Adapter] | None = None):
    full = list(roster or DEFAULT_ROSTER)
    if not full:
        raise RuntimeError("no configured models available")

    if smoke:
        smoke_roster = full[:2] if len(full) > 1 else full
        sweep = []
        if len(smoke_roster) > 1:
            sweep.append(("smoke_rotate_words_6",
                          Boundary("fixed_words", 6, 6),
                          smoke_roster, "round_robin"))
        sweep.extend(
            (f"smoke_baseline_{a.name}",
             Boundary("fixed_tokens", target_ref_tokens, target_ref_tokens),
             [a], "round_robin")
            for a in smoke_roster
        )
        return sweep

    sweep = []

    # --- rotation conditions ---
    if len(full) > 1:
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
                      Boundary("fixed_tokens", target_ref_tokens, target_ref_tokens),
                      [a], "round_robin"))

    return sweep


def main(out_dir="results", target_ref_tokens=400, reps=2, smoke=False, max_words=500):
    if not os.environ.get("PANGRAM_API_KEY"):
        raise RuntimeError("PANGRAM_API_KEY is required for scoring")

    roster, skipped = _available_roster(DEFAULT_ROSTER)
    _print_roster_status(roster, skipped)

    prompts = PROMPTS
    if smoke:
        prompts = PROMPTS[:1]
        target_ref_tokens = min(target_ref_tokens, 80)
        if max_words is not None:
            max_words = min(max_words, 120)
        reps = 1
        if out_dir == "results":
            out_dir = "results_smoke"

    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    prov_dir = out / "provenance"
    prov_dir.mkdir(exist_ok=True)

    rows = []
    doc_id = 0
    for label, boundary, roster, order in build_sweep(
        target_ref_tokens,
        smoke=smoke,
        roster=roster,
    ):
        for prompt in prompts:
            for rep in range(reps):
                doc = generate(prompt, roster, boundary,
                               target_ref_tokens=target_ref_tokens,
                               order=order,
                               max_words=max_words)
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
                    "target_ref_tokens": target_ref_tokens,
                    "max_words": max_words,
                    "spans": [dataclasses.asdict(sp) for sp in doc.spans],
                    "score": s,
                }, indent=2))
                doc_id += 1
                print(f"[{doc_id}] {label:28s} ai={ai}")

    _write_results_csv(rows, out / "results.csv")
    summary = _summary_rows(rows)
    _write_summary_csv(summary, out / "summary.csv")
    _print_summary(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the rotation experiment.")
    parser.add_argument("--out-dir", default="results",
                        help="Directory for CSVs and provenance JSON.")
    parser.add_argument("--target-ref-tokens", type=int, default=400,
                        help="Target document length in reference tokens.")
    parser.add_argument("--reps", type=int, default=2,
                        help="Repetitions per condition/prompt.")
    parser.add_argument("--max-words", type=int, default=500,
                        help="Maximum generated document length in words. Use 0 to disable.")
    parser.add_argument("--smoke", action="store_true",
                        help="Run one cheap prompt/condition pass before a full sweep.")
    args = parser.parse_args()
    max_words = None if args.max_words <= 0 else args.max_words
    main(out_dir=args.out_dir,
         target_ref_tokens=args.target_ref_tokens,
         reps=args.reps,
         smoke=args.smoke,
         max_words=max_words)
