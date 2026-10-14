#!/usr/bin/env python3
"""PCA and residual agreement of model predictions.

Shared image-level component across models, and agreement that remains
after each prediction is linearly regressed on the human target.

Reads ``test_predictions.txt`` from every run dir, aligns on common test
images, and saves:

    pca_components.npz             scores, loadings, explained variance ratio
    pca_summary.json               PC1 vs human r, model loadings on PC1..PC3
    pca_pc1_vs_human.png           PC1 score vs human z target
    pca_loadings.png               bar plot of model loadings on PC1
    pca_explained_variance.png     scree plot

    residual_agreement.csv         pairwise residual-residual r and prediction-prediction r
    residual_agreement.png         heatmap of residual-residual r
    residual_summary.json          mean/median resid-resid r, mean pred-pred r,
                                   mean model-human r, bootstrap CIs

If ``--n-boot`` > 0 (default 1000) the script also bootstraps the headline
agreement numbers over images for 95% CIs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

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


def pearson_columns(M):
    Mc = M - M.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(Mc, axis=0) + 1e-12
    return (Mc.T @ Mc) / np.outer(norms, norms)


def pearson_vec(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--include", nargs="+",
                    default=["frozen_", "vlm_probe", "handcrafted"])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    models = gather_runs(runs_dir, args.include)
    if len(models) < 3:
        raise SystemExit(f"need at least 3 models for PCA, got {list(models)}")
    common, target, aligned = align_to_common(models)
    names = list(aligned.keys())
    M = len(names)
    print(f"[align] {len(common)} common test images x {M} models")

    P = np.stack([aligned[n][1] for n in names], axis=1)
    Pz = (P - P.mean(axis=0)) / (P.std(axis=0) + 1e-12)

    cov = (Pz.T @ Pz) / (Pz.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(-eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    explained = eigvals / max(eigvals.sum(), 1e-12)
    scores = Pz @ eigvecs

    pc1_human_r = pearson_vec(scores[:, 0], target)
    pc_human_r = [pearson_vec(scores[:, k], target) for k in range(min(5, M))]

    np.savez_compressed(
        out_dir / "pca_components.npz",
        scores=scores.astype(np.float32),
        loadings=eigvecs.astype(np.float32),
        eigvals=eigvals.astype(np.float32),
        explained_variance_ratio=explained.astype(np.float32),
        names=np.array(names, dtype=object),
        target=target.astype(np.float32),
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    sign = 1.0 if pc1_human_r >= 0 else -1.0
    ax.scatter(target, sign * scores[:, 0], s=4, alpha=0.4)
    ax.set_xlabel("human vividness z")
    ax.set_ylabel(f"PC1 score (sign-aligned to human, r={sign * pc1_human_r:+.3f})")
    ax.set_title(f"PC1 across {M} models vs human (N={len(common)})")
    fig.tight_layout()
    fig.savefig(out_dir / "pca_pc1_vs_human.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, max(3, 0.3 * M)))
    ax.barh(range(M), sign * eigvecs[:, 0], color="#4c72b0")
    ax.set_yticks(range(M)); ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("loading on PC1 (sign-aligned to human)")
    ax.invert_yaxis()
    ax.set_title("model loadings on PC1")
    fig.tight_layout()
    fig.savefig(out_dir / "pca_loadings.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(M), explained, color="#dd8452")
    ax.set_xticks(range(M))
    ax.set_xticklabels([f"PC{k + 1}" for k in range(M)], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("explained variance ratio")
    ax.set_title(f"PCA scree (PC1={explained[0]*100:.1f}%, PC2={explained[1]*100:.1f}%)")
    fig.tight_layout()
    fig.savefig(out_dir / "pca_explained_variance.png", dpi=150)
    plt.close(fig)

    pca_summary = {
        "n_common_test": int(len(common)),
        "n_models": int(M),
        "models": names,
        "explained_variance_ratio": [float(x) for x in explained],
        "pc_vs_human_r": [float(x) for x in pc_human_r],
        "pc1_vs_human_r": float(pc1_human_r),
        "pc1_loadings_signed_to_human": [float(x) for x in (sign * eigvecs[:, 0])],
    }
    with (out_dir / "pca_summary.json").open("w") as f:
        json.dump(pca_summary, f, indent=2)

    residuals = np.zeros_like(P)
    for i in range(M):
        x = P[:, i]
        denom = np.dot(target - target.mean(), target - target.mean()) + 1e-12
        beta = np.dot(target - target.mean(), x - x.mean()) / denom
        alpha = x.mean() - beta * target.mean()
        residuals[:, i] = x - (alpha + beta * target)

    pred_corr = pearson_columns(P)
    resid_corr = pearson_columns(residuals)

    iu = np.triu_indices(M, 1)
    mean_pred_r = float(pred_corr[iu].mean())
    median_pred_r = float(np.median(pred_corr[iu]))
    mean_resid_r = float(resid_corr[iu].mean())
    median_resid_r = float(np.median(resid_corr[iu]))
    model_human_r = np.array([pearson_vec(P[:, i], target) for i in range(M)])
    mean_mh_r = float(model_human_r.mean())

    boot_summary = None
    if args.n_boot > 0:
        rng = np.random.default_rng(args.seed)
        N = P.shape[0]
        pred_means = np.zeros(args.n_boot)
        resid_means = np.zeros(args.n_boot)
        mh_means = np.zeros(args.n_boot)
        for b in range(args.n_boot):
            sel = rng.integers(0, N, size=N)
            Pb = P[sel]; Rb = residuals[sel]; tb = target[sel]
            cP = pearson_columns(Pb); cR = pearson_columns(Rb)
            pred_means[b] = float(cP[iu].mean())
            resid_means[b] = float(cR[iu].mean())
            mh_means[b] = float(np.mean([pearson_vec(Pb[:, i], tb) for i in range(M)]))
        def ci(arr):
            return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]
        boot_summary = {
            "mean_pred_pred_r_ci": ci(pred_means),
            "mean_resid_resid_r_ci": ci(resid_means),
            "mean_model_human_r_ci": ci(mh_means),
            "n_boot": int(args.n_boot),
        }

    with (out_dir / "residual_agreement.csv").open("w") as f:
        f.write("model_a,model_b,pred_pred_r,resid_resid_r\n")
        for i in range(M):
            for j in range(i + 1, M):
                f.write(
                    f"{names[i]},{names[j]},{pred_corr[i, j]:.4f},{resid_corr[i, j]:.4f}\n"
                )

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, mat, label in zip(axes, [pred_corr, resid_corr], ["prediction-prediction r", "residual-residual r"]):
        im = ax.imshow(mat, cmap="viridis", vmin=-0.2, vmax=1.0)
        ax.set_xticks(range(M)); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(M)); ax.set_yticklabels(names, fontsize=8)
        for i in range(M):
            for j in range(M):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color="white" if mat[i, j] < 0.55 else "black", fontsize=6)
        ax.set_title(label)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(
        f"Pred-pred mean r = {mean_pred_r:+.3f} | Resid-resid mean r = {mean_resid_r:+.3f}"
        f" | Model-human mean r = {mean_mh_r:+.3f}",
    )
    fig.tight_layout()
    fig.savefig(out_dir / "residual_agreement.png", dpi=150)
    plt.close(fig)

    summary = {
        "n_common_test": int(len(common)),
        "models": names,
        "mean_pred_pred_r": mean_pred_r,
        "median_pred_pred_r": median_pred_r,
        "mean_resid_resid_r": mean_resid_r,
        "median_resid_resid_r": median_resid_r,
        "mean_model_human_r": mean_mh_r,
        "model_human_r": dict(zip(names, [float(x) for x in model_human_r])),
        "bootstrap_ci_n_image": boot_summary,
    }
    with (out_dir / "residual_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print("[summary]", json.dumps({k: v for k, v in summary.items() if k != "model_human_r"}, indent=2))
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
