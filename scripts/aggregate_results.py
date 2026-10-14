#!/usr/bin/env python3
"""
Aggregate saved runs into one table.

Covers three families (same test split, so numbers align):
  1. Frozen-probe runs        (test_metrics.json        in frozen_*)
  2. VLM-as-judge runs        (vlm_metrics.json          in vlm_*)
  3. Caption-only LLM runs    (caption_llm_metrics.json  in caption_llm_*)

Writes:
  - <runs>/summary_all.csv
  - <runs>/summary_all.md
and prints the table to stdout sorted by Pearson r.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FAMILY_SPECS = [
    ("frozen_probe", "frozen_", "test_metrics.json", "test"),
    ("vlm_judge", "vlm_", "vlm_metrics.json", None),
    ("caption_llm", "caption_llm_", "caption_llm_metrics.json", None),
]


def load_row(run: Path, family: str, metrics_file: str, nested_key):
    mp = run / metrics_file
    if not mp.exists():
        return None
    try:
        blob = json.loads(mp.read_text())
    except json.JSONDecodeError:
        return None
    m = blob[nested_key] if nested_key and nested_key in blob else blob
    pearson = m.get("pearson_r")
    if pearson is None:
        return None
    row = {
        "family": family,
        "run": run.name,
        "model": blob.get("model") or run.name.split("_", 1)[-1],
        "pearson_r": round(float(m["pearson_r"]), 4),
        "spearman_r": round(float(m["spearman_r"]), 4),
        "mae": round(float(m["mae"]), 4),
        "rmse": round(float(m["rmse"]), 4),
        "best_epoch": blob.get("best_epoch"),
        "epochs_run": blob.get("epochs_run"),
        "n_scored": blob.get("n_scored"),
    }
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out-basename", default="summary_all")
    a = ap.parse_args()

    runs = Path(a.runs_dir)
    rows = []
    for fam, prefix, metrics_file, nested in FAMILY_SPECS:
        for run in sorted(runs.glob(f"{prefix}*")):
            if not run.is_dir():
                continue
            r = load_row(run, fam, metrics_file, nested)
            if r is not None:
                rows.append(r)

    rows.sort(key=lambda r: r["pearson_r"], reverse=True)

    cols = ["family", "run", "model", "pearson_r", "spearman_r", "mae", "rmse", "best_epoch", "epochs_run", "n_scored"]

    csv_path = runs / f"{a.out_basename}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    md_lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        md_lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    md_path = runs / f"{a.out_basename}.md"
    md_path.write_text("\n".join(md_lines) + "\n")

    print(f"{'family':14s}{'run':42s}{'Pearson':>10s}{'Spearman':>10s}{'MAE':>8s}{'RMSE':>8s}")
    for r in rows:
        print(f"{r['family']:14s}{r['run']:42s}{r['pearson_r']:>+10.4f}{r['spearman_r']:>+10.4f}{r['mae']:>8.3f}{r['rmse']:>8.3f}")
    print(f"\nwrote {csv_path}\nwrote {md_path}")


if __name__ == "__main__":
    main()
