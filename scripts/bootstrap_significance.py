#!/usr/bin/env python3
"""Bootstrap CIs and paired-bootstrap tests for model<->human correlations.

Reads ``test_predictions.txt`` (or ``.csv``) from every run dir, aligns on the
common set of test indices, and saves:

    per_model_bootstrap.csv     Pearson r and Spearman rho with 95% bootstrap CI
    paired_bootstrap_pearson.csv pairwise delta-r and bootstrap p-values
    paired_bootstrap_pearson.png pairwise heatmap of delta-r (signed)
    paired_bootstrap_pvalue.png  pairwise heatmap of two-sided p-values
    summary.json                 machine-readable summary

Each metric uses the same bootstrap resample of images, so paired tests
are matched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_preds(run_dir: Path):
    p = run_dir / "test_predictions.txt"
    if not p.exists():
        p = run_dir / "test_predictions.csv"
    if not p.exists():
        return None
    if p.suffix == ".csv":
        a = np.genfromtxt(p, delimiter=",", names=True)
        pred_field = "score" if "score" in a.dtype.names else "pred"
        return (
            a["h5_idx"].astype(int),
            a["target"].astype(float),
            a[pred_field].astype(float),
        )
    a = np.loadtxt(p, comments="#")
    if a.ndim == 1:
        a = a.reshape(1, -1)
    return a[:, 0].astype(int), a[:, 1].astype(float), a[:, 2].astype(float)


def gather_runs(runs_dir: Path, include):
    models = {}
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        if include and not any(s in d.name for s in include):
            continue
        r = load_preds(d)
        if r is None:
            continue
        models[d.name] = r
    return models


def align_to_common(models):
    common = None
    for n, (idx, _, _) in models.items():
        s = set(int(i) for i in idx)
        common = s if common is None else common & s
    common = np.array(sorted(common))
    aligned = {}
    target_ref = None
    for n, (idx, target, pred) in models.items():
        table = {int(v): k for k, v in enumerate(idx)}
        rows = np.array([table[int(i)] for i in common])
        aligned[n] = (target[rows], pred[rows])
        if target_ref is None:
            target_ref = target[rows]
    return common, target_ref, aligned


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    den = np.linalg.norm(a) * np.linalg.norm(b) + 1e-12
    return float((a * b).sum() / den)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(stats.spearmanr(a, b).correlation)


def bootstrap_correlations(target, preds, n_boot, seed):
    rng = np.random.default_rng(seed)
    N = len(target)
    M = preds.shape[0]
    pearson_boot = np.zeros((n_boot, M), dtype=np.float32)
    spearman_boot = np.zeros((n_boot, M), dtype=np.float32)
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        t = target[idx]
        for m in range(M):
            p = preds[m, idx]
            pearson_boot[b, m] = pearson(t, p)
            spearman_boot[b, m] = spearman(t, p)
        if (b + 1) % max(1, n_boot // 10) == 0:
            print(f"  bootstrap {b + 1}/{n_boot}", flush=True)
    return pearson_boot, spearman_boot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--include",
        nargs="+",
        default=["frozen_", "vlm_probe", "handcrafted"],
        help="run-name substrings to include",
    )
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--reference-models",
        nargs="*",
        default=None,
        help="if given, only run pairwise tests with ref vs others (reduces time)",
    )
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = gather_runs(runs_dir, args.include)
    if len(models) < 2:
        raise SystemExit(f"need at least 2 runs, got {list(models)}")

    common, target, aligned = align_to_common(models)
    names = list(aligned.keys())
    M = len(names)
    print(f"[align] {len(common)} common test images across {M} runs")
    print("[runs]", names)

    preds = np.stack([aligned[n][1] for n in names], axis=0)
    point_pearson = np.array([pearson(target, preds[i]) for i in range(M)])
    point_spearman = np.array([spearman(target, preds[i]) for i in range(M)])

    pearson_boot, spearman_boot = bootstrap_correlations(
        target=target, preds=preds, n_boot=args.n_boot, seed=args.seed,
    )

    rows = []
    for i, n in enumerate(names):
        pearson_lo, pearson_hi = np.percentile(pearson_boot[:, i], [2.5, 97.5])
        spearman_lo, spearman_hi = np.percentile(spearman_boot[:, i], [2.5, 97.5])
        rows.append(
            {
                "model": n,
                "n": int(len(target)),
                "pearson_r": float(point_pearson[i]),
                "pearson_ci_lo": float(pearson_lo),
                "pearson_ci_hi": float(pearson_hi),
                "spearman_rho": float(point_spearman[i]),
                "spearman_ci_lo": float(spearman_lo),
                "spearman_ci_hi": float(spearman_hi),
            }
        )
    rows.sort(key=lambda r: -r["pearson_r"])

    per_model_path = out_dir / "per_model_bootstrap.csv"
    with per_model_path.open("w") as f:
        f.write("model,n,pearson_r,pearson_ci_lo,pearson_ci_hi,spearman_rho,spearman_ci_lo,spearman_ci_hi\n")
        for r in rows:
            f.write(
                f"{r['model']},{r['n']},{r['pearson_r']:.4f},{r['pearson_ci_lo']:.4f},"
                f"{r['pearson_ci_hi']:.4f},{r['spearman_rho']:.4f},"
                f"{r['spearman_ci_lo']:.4f},{r['spearman_ci_hi']:.4f}\n"
            )
    print(f"[save] {per_model_path}")

    if args.reference_models:
        ref_idx = [names.index(n) for n in args.reference_models if n in names]
    else:
        ref_idx = list(range(M))

    delta_pearson = np.zeros((M, M), dtype=np.float32)
    pvalue_pearson = np.full((M, M), np.nan, dtype=np.float32)
    pair_rows = []
    for i in ref_idx:
        for j in range(M):
            if i == j:
                continue
            diff = pearson_boot[:, i] - pearson_boot[:, j]
            mean_diff = float(diff.mean())
            ci_lo, ci_hi = np.percentile(diff, [2.5, 97.5])
            p_two = 2.0 * float(min((diff <= 0).mean(), (diff >= 0).mean()))
            delta_pearson[i, j] = mean_diff
            pvalue_pearson[i, j] = p_two
            pair_rows.append(
                {
                    "model_a": names[i],
                    "model_b": names[j],
                    "delta_pearson": mean_diff,
                    "delta_ci_lo": float(ci_lo),
                    "delta_ci_hi": float(ci_hi),
                    "p_two_sided": p_two,
                }
            )

    pair_path = out_dir / "paired_bootstrap_pearson.csv"
    with pair_path.open("w") as f:
        f.write("model_a,model_b,delta_pearson,delta_ci_lo,delta_ci_hi,p_two_sided\n")
        for r in pair_rows:
            f.write(
                f"{r['model_a']},{r['model_b']},{r['delta_pearson']:.4f},"
                f"{r['delta_ci_lo']:.4f},{r['delta_ci_hi']:.4f},{r['p_two_sided']:.4f}\n"
            )
    print(f"[save] {pair_path}")

    fig, ax = plt.subplots(figsize=(10, 8))
    abs_max = max(0.05, float(np.nanmax(np.abs(delta_pearson))))
    im = ax.imshow(delta_pearson, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max)
    ax.set_xticks(range(M)); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(M)); ax.set_yticklabels(names, fontsize=8)
    for i in range(M):
        for j in range(M):
            if i == j:
                continue
            ax.text(j, i, f"{delta_pearson[i, j]:+.3f}", ha="center", va="center", fontsize=6)
    ax.set_title("paired bootstrap delta Pearson r (row - col)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "paired_bootstrap_pearson.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(pvalue_pearson, cmap="viridis_r", vmin=0.0, vmax=0.2)
    ax.set_xticks(range(M)); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(M)); ax.set_yticklabels(names, fontsize=8)
    for i in range(M):
        for j in range(M):
            if np.isnan(pvalue_pearson[i, j]):
                continue
            ax.text(j, i, f"{pvalue_pearson[i, j]:.2f}", ha="center", va="center", fontsize=6)
    ax.set_title("paired bootstrap two-sided p-values")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_dir / "paired_bootstrap_pvalue.png", dpi=150)
    plt.close(fig)

    summary = {
        "n_common_test": int(len(common)),
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
        "models": names,
        "per_model": rows,
        "include": args.include,
    }
    with (out_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[save] {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
