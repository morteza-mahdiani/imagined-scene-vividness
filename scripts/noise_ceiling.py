#!/usr/bin/env python3
"""Estimate the image-determined noise ceiling on human vividness via split-half.

Given the detection-filtered trial table (``final_cleaned_vividness_data.csv``),
randomly split the 1 991 participants into two halves, compute each half's
per-image mean rating, and correlate the two halves on images shared by both.

The Spearman-Brown corrected value (2r / (1+r)) is our estimate of the
*image-determined* signal an oracle could reach from images alone.

We also report per-image reliability with N trials (test-retest).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _nsd_to_idx(s: pd.Series) -> pd.Series:
    return s.astype(str).str.extract(r"nsd(\d{5})")[0].astype(float).astype("Int64")


def split_half(df: pd.DataFrame, rng: np.random.Generator):
    parts = df["participation"].unique()
    rng.shuffle(parts)
    a = set(parts[: len(parts) // 2])
    half_a = df[df["participation"].isin(a)]
    half_b = df[~df["participation"].isin(a)]
    ma = half_a.groupby("stim1_idx")["rating_zscore"].mean()
    mb = half_b.groupby("stim1_idx")["rating_zscore"].mean()
    common = ma.index.intersection(mb.index)
    return ma.loc[common].to_numpy(), mb.loc[common].to_numpy(), len(common)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clean-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-splits", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    df = pd.read_csv(a.clean_csv, low_memory=False,
                     usecols=["participation", "stim1_name", "rating_zscore", "include"])
    df = df[df["include"].astype(str).str.lower().isin(("true", "1"))].copy()
    df["stim1_idx"] = _nsd_to_idx(df["stim1_name"])
    df = df.dropna(subset=["stim1_idx"])
    df["stim1_idx"] = df["stim1_idx"].astype(int)
    print(f"[data] {len(df)} trials, {df['participation'].nunique()} participants, "
          f"{df['stim1_idx'].nunique()} images", flush=True)

    rng = np.random.default_rng(a.seed)
    rs = np.zeros(a.n_splits, dtype=np.float64)
    rs_sb = np.zeros_like(rs)
    for i in range(a.n_splits):
        a_arr, b_arr, n = split_half(df, rng)
        r = float(np.corrcoef(a_arr, b_arr)[0, 1])
        rs[i] = r
        rs_sb[i] = 2 * r / (1 + r)   # Spearman-Brown doubling
        if (i + 1) % 10 == 0:
            print(f"  split {i+1}/{a.n_splits}  r={r:+.4f}  SB={rs_sb[i]:+.4f}  n_images={n}", flush=True)

    # Also: Cronbach-style reliability by # trials per image
    counts = df.groupby("stim1_idx").size()
    per_count = (
        counts.value_counts().sort_index().head(10).to_dict()
    )

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_trials": int(len(df)),
        "n_participants": int(df["participation"].nunique()),
        "n_images": int(df["stim1_idx"].nunique()),
        "mean_trials_per_image": float(counts.mean()),
        "splits": int(a.n_splits),
        "half_half_r": {
            "mean": float(rs.mean()),
            "sd": float(rs.std()),
            "p5": float(np.quantile(rs, 0.05)),
            "p95": float(np.quantile(rs, 0.95)),
        },
        "spearman_brown_full_sample_ceiling": {
            "mean": float(rs_sb.mean()),
            "sd": float(rs_sb.std()),
            "p5": float(np.quantile(rs_sb, 0.05)),
            "p95": float(np.quantile(rs_sb, 0.95)),
        },
        "trials_per_image_frequency_top10": {int(k): int(v) for k, v in per_count.items()},
    }
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
