#!/usr/bin/env python3
"""Residual signal, later components, rank stability, ceiling, and layers.

Runs five analyses and writes them under ``--out-dir``:

    pc1_residual/       PC1-residual experiment (post-hoc)
    pc23_loadings/      PC2 / PC3 loadings + model-family coloring
    rank_stability/     bootstrap rank-frequency table + plot
    noise_ceiling/      calibration plot positioning every model + PC1 vs ceiling
    layerwise_polish/   layer-wise r with bootstrap CIs and Spearman alongside

Inputs are the saved run directory and a noise-ceiling JSON file.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


_FAMILY_RULES = [
    ("CNN",                ["resnet", "vgg", "convnext"]),
    ("ViT supervised",     ["vit_b16"]),
    ("MAE",                ["mae"]),
    ("DINO",               ["dinov2", "dinov3"]),
    ("CLIP family",        ["clip"]),
    ("SigLIP family",      ["siglip"]),
    ("VLM probe",          ["vlm_probe", "qwen"]),
    ("Handcrafted",        ["handcrafted"]),
]

_FAMILY_COLORS = {
    "CNN":             "#1f77b4",
    "ViT supervised":  "#9467bd",
    "MAE":             "#8c564b",
    "DINO":            "#2ca02c",
    "CLIP family":     "#ff7f0e",
    "SigLIP family":   "#d62728",
    "VLM probe":       "#7f7f7f",
    "Handcrafted":     "#bcbd22",
}


def family_of(name):
    n = name.lower()
    for fam, keys in _FAMILY_RULES:
        if any(k in n for k in keys):
            return fam
    return "Other"


def load_preds(run_dir):
    p = run_dir / "test_predictions.txt"
    if not p.exists():
        p = run_dir / "test_predictions.csv"
    if not p.exists():
        return None
    if p.suffix == ".csv":
        a = np.genfromtxt(p, delimiter=",", names=True)
        pf = "score" if "score" in a.dtype.names else "pred"
        return a["h5_idx"].astype(int), a["target"].astype(float), a[pf].astype(float)
    a = np.loadtxt(p, comments="#")
    if a.ndim == 1:
        a = a.reshape(1, -1)
    return a[:, 0].astype(int), a[:, 1].astype(float), a[:, 2].astype(float)


def gather(runs_dir, include):
    out = {}
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir():
            continue
        if include and not any(s in d.name for s in include):
            continue
        r = load_preds(d)
        if r is not None:
            out[d.name] = r
    return out


def align(models):
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


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def regress(x, y):
    """Return slope, intercept, residual y - (a + bx)."""
    a, b = np.polyfit(x, y, 1)
    return a, b, y - (a * x + b)


def pc1_residual_experiment(common, target, aligned, names, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    P = np.stack([aligned[n][1] for n in names], axis=1)
    Pz = (P - P.mean(0)) / (P.std(0) + 1e-12)
    cov = (Pz.T @ Pz) / (Pz.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(-eigvals)
    eigvecs = eigvecs[:, order]
    pc1 = Pz @ eigvecs[:, 0]
    sign = 1.0 if pearson(pc1, target) >= 0 else -1.0
    pc1 = sign * pc1

    slope, intercept, target_resid = regress(pc1, target)

    rows = []
    for n in names:
        p = aligned[n][1]
        s, i, p_resid = regress(pc1, p)
        r_marg = pearson(p, target)
        r_resid = pearson(p_resid, target_resid)
        r_pred_pc1 = pearson(p, pc1)
        r2_pc1 = pearson(pc1, target) ** 2
        ssres_pc1 = float(((target - (slope * pc1 + intercept)) ** 2).sum())
        ssres_combo = float(np.linalg.lstsq(
            np.c_[pc1, p, np.ones(len(target))], target, rcond=None
        )[1][0]) if len(target) > 3 else 0.0
        sstot = float(((target - target.mean()) ** 2).sum() + 1e-12)
        r2_combo = 1.0 - ssres_combo / sstot
        delta_r2 = r2_combo - r2_pc1
        rows.append({
            "model": n,
            "family": family_of(n),
            "r_marginal": r_marg,
            "r_with_pc1": r_pred_pc1,
            "r_residual_vs_human_residual": r_resid,
            "r2_pc1": r2_pc1,
            "r2_combined": r2_combo,
            "delta_r2_over_pc1": delta_r2,
        })

    rows.sort(key=lambda r: -r["delta_r2_over_pc1"])
    with (out_dir / "pc1_residual_summary.csv").open("w") as f:
        f.write("model,family,r_marginal,r_with_pc1,r_residual_vs_human_residual,"
                "r2_pc1,r2_combined,delta_r2_over_pc1\n")
        for r in rows:
            f.write(f"{r['model']},{r['family']},{r['r_marginal']:.4f},"
                    f"{r['r_with_pc1']:.4f},{r['r_residual_vs_human_residual']:.4f},"
                    f"{r['r2_pc1']:.4f},{r['r2_combined']:.4f},"
                    f"{r['delta_r2_over_pc1']:.4f}\n")
    with (out_dir / "pc1_residual_summary.json").open("w") as f:
        json.dump(rows, f, indent=2)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(rows))))
    ys = np.arange(len(rows))
    deltas = [r["delta_r2_over_pc1"] for r in rows]
    cols = [_FAMILY_COLORS.get(r["family"], "#444") for r in rows]
    ax.barh(ys, deltas, color=cols)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["model"] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("ΔR² over PC1 (full-model regression on the test split)")
    ax.set_title(
        "PC1-residual experiment: how much each model adds OVER the shared "
        f"machine proxy\n(r(PC1, human) = {pearson(pc1, target):+.3f})"
    )
    fig.tight_layout()
    fig.savefig(out_dir / "pc1_residual_delta_r2.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(rows))))
    rs = [r["r_residual_vs_human_residual"] for r in rows]
    ax.barh(ys, rs, color=cols)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["model"] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlabel("Pearson r between (model − PC1) residual and (human − PC1) residual")
    ax.set_title("Does any model predict the human signal that lies OUTSIDE PC1?")
    fig.tight_layout()
    fig.savefig(out_dir / "pc1_residual_resid_vs_resid.png", dpi=150)
    plt.close(fig)
    return pc1, target_resid


def pc23_loadings(common, target, aligned, names, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    P = np.stack([aligned[n][1] for n in names], axis=1)
    Pz = (P - P.mean(0)) / (P.std(0) + 1e-12)
    cov = (Pz.T @ Pz) / (Pz.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(-eigvals)
    eigvals = eigvals[order]; eigvecs = eigvecs[:, order]
    explained = eigvals / max(eigvals.sum(), 1e-12)
    scores = Pz @ eigvecs

    pc_human_r = [float(pearson(scores[:, k], target)) for k in range(min(5, len(names)))]
    sign1 = 1.0 if pc_human_r[0] >= 0 else -1.0
    eigvecs[:, 0] *= sign1
    scores[:, 0] *= sign1

    fams = [family_of(n) for n in names]
    by_family = {}
    for f in fams:
        by_family.setdefault(f, [])
    for i, n in enumerate(names):
        by_family[fams[i]].append(i)

    fig, axes = plt.subplots(1, 3, figsize=(15, max(4, 0.3 * len(names))))
    for k, ax in zip([0, 1, 2], axes):
        for fam in by_family:
            ix = by_family[fam]
            ax.scatter(eigvecs[ix, k], np.array(ix), s=80,
                       color=_FAMILY_COLORS.get(fam, "#444"), label=fam)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color="black", lw=0.5)
        title = f"PC{k + 1} (explained {explained[k] * 100:.1f}%, "
        title += f"r vs human = {pc_human_r[k]:+.3f})"
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("loading")
    axes[0].legend(loc="upper right", fontsize=7)
    fig.suptitle("PCA loadings on first three components, coloured by model family")
    fig.tight_layout()
    fig.savefig(out_dir / "pc1_pc2_pc3_loadings_by_family.png", dpi=150)
    plt.close(fig)

    out = {
        "explained_variance_ratio_top5": [float(x) for x in explained[:5]],
        "pc_vs_human_r_top5": pc_human_r,
        "components": [
            {"model": n, "family": fams[i],
             "loading_pc1": float(eigvecs[i, 0]),
             "loading_pc2": float(eigvecs[i, 1]),
             "loading_pc3": float(eigvecs[i, 2])}
            for i, n in enumerate(names)
        ],
    }
    with (out_dir / "pc23_loadings.json").open("w") as f:
        json.dump(out, f, indent=2)
    with (out_dir / "pc23_loadings.csv").open("w") as f:
        f.write("model,family,pc1_loading,pc2_loading,pc3_loading\n")
        for c in out["components"]:
            f.write(f"{c['model']},{c['family']},{c['loading_pc1']:.4f},"
                    f"{c['loading_pc2']:.4f},{c['loading_pc3']:.4f}\n")


def rank_stability(common, target, aligned, names, out_dir, n_boot=10000, seed=42):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    M = len(names)
    P = np.stack([aligned[n][1] for n in names], axis=1)
    rank_counts = np.zeros((M, M), dtype=np.int64)
    top1 = np.zeros(M, dtype=np.int64)
    top3 = np.zeros(M, dtype=np.int64)
    N = len(target)
    for b in range(n_boot):
        sel = rng.integers(0, N, size=N)
        Pb = P[sel]; tb = target[sel]
        rs = np.array([pearson(Pb[:, i], tb) for i in range(M)])
        order = np.argsort(-rs)
        ranks = np.empty(M, dtype=int)
        ranks[order] = np.arange(M)
        for i, r in enumerate(ranks):
            rank_counts[i, r] += 1
        top1[order[0]] += 1
        for j in range(min(3, M)):
            top3[order[j]] += 1

    rank_freq = rank_counts.astype(np.float64) / float(n_boot)
    fig, ax = plt.subplots(figsize=(10, max(5, 0.4 * M)))
    im = ax.imshow(rank_freq, aspect="auto", cmap="viridis")
    ax.set_yticks(range(M)); ax.set_yticklabels(names, fontsize=8)
    ax.set_xticks(range(M)); ax.set_xticklabels([f"#{r + 1}" for r in range(M)], fontsize=8)
    ax.set_title(f"bootstrap rank frequency over {n_boot} resamples")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "rank_stability_heatmap.png", dpi=150)
    plt.close(fig)

    rows = []
    for i, n in enumerate(names):
        rows.append({
            "model": n,
            "p_top1": float(top1[i]) / float(n_boot),
            "p_top3": float(top3[i]) / float(n_boot),
            "modal_rank": int(np.argmax(rank_counts[i])) + 1,
            "rank_distribution": [float(x) for x in rank_freq[i]],
        })
    rows.sort(key=lambda r: -r["p_top1"])
    with (out_dir / "rank_stability.csv").open("w") as f:
        f.write("model,p_top1,p_top3,modal_rank\n")
        for r in rows:
            f.write(f"{r['model']},{r['p_top1']:.4f},{r['p_top3']:.4f},{r['modal_rank']}\n")
    with (out_dir / "rank_stability.json").open("w") as f:
        json.dump({"n_boot": n_boot, "rows": rows}, f, indent=2)


def noise_ceiling_plot(common, target, aligned, names,
                       noise_ceiling_json, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    nc = json.loads(Path(noise_ceiling_json).read_text())
    rel_full = float(nc["spearman_brown_full_sample_ceiling"]["mean"])
    rel_full_lo = float(nc["spearman_brown_full_sample_ceiling"]["p5"])
    rel_full_hi = float(nc["spearman_brown_full_sample_ceiling"]["p95"])
    r_ceiling = float(np.sqrt(rel_full))
    r_ceiling_lo = float(np.sqrt(max(rel_full_lo, 0.0)))
    r_ceiling_hi = float(np.sqrt(max(rel_full_hi, 0.0)))

    P = np.stack([aligned[n][1] for n in names], axis=1)
    Pz = (P - P.mean(0)) / (P.std(0) + 1e-12)
    cov = (Pz.T @ Pz) / (Pz.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(-eigvals)
    eigvecs = eigvecs[:, order]
    pc1 = Pz @ eigvecs[:, 0]
    sign = 1.0 if pearson(pc1, target) >= 0 else -1.0
    pc1 = sign * pc1
    r_pc1_human = pearson(pc1, target)

    rs = np.array([pearson(aligned[n][1], target) for n in names])
    fams = [family_of(n) for n in names]

    rows = []
    for i, n in enumerate(names):
        rows.append({
            "model": n,
            "family": fams[i],
            "r": float(rs[i]),
            "fraction_of_ceiling": float(rs[i] / r_ceiling),
        })
    rows.sort(key=lambda r: -r["r"])

    fig, ax = plt.subplots(figsize=(10, max(4, 0.3 * len(names))))
    ys = np.arange(len(rows))
    cols = [_FAMILY_COLORS.get(r["family"], "#444") for r in rows]
    ax.barh(ys, [r["r"] for r in rows], color=cols)
    ax.axvline(r_ceiling, color="black", lw=1.0,
               label=f"image-conditional Pearson-r ceiling = {r_ceiling:.3f}")
    ax.axvspan(r_ceiling_lo, r_ceiling_hi, color="grey", alpha=0.15,
               label=f"95% interval [{r_ceiling_lo:.3f}, {r_ceiling_hi:.3f}]")
    ax.axvline(r_pc1_human, color="red", ls="--", lw=1.0,
               label=f"PC1 of 14-model battery = {r_pc1_human:+.3f}")
    ax.set_yticks(ys)
    ax.set_yticklabels([r["model"] for r in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Pearson r against mean human vividness (test)")
    ax.set_title(
        "Calibration: every frozen probe vs the image-conditional ceiling\n"
        f"(top model = {rows[0]['r'] / r_ceiling * 100:.1f}% of ceiling, "
        f"PC1 = {r_pc1_human / r_ceiling * 100:.1f}% of ceiling)"
    )
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "noise_ceiling_calibration.png", dpi=150)
    plt.close(fig)

    summary = {
        "reliability_full_sample_mean": rel_full,
        "reliability_full_sample_ci": [rel_full_lo, rel_full_hi],
        "image_conditional_pearson_r_ceiling": r_ceiling,
        "image_conditional_pearson_r_ceiling_ci": [r_ceiling_lo, r_ceiling_hi],
        "pc1_vs_human_r": float(r_pc1_human),
        "pc1_fraction_of_ceiling": float(r_pc1_human / r_ceiling),
        "rows": rows,
    }
    with (out_dir / "noise_ceiling_calibration.json").open("w") as f:
        json.dump(summary, f, indent=2)


def layerwise_polish(layerwise_dir, out_dir, n_boot=2000, seed=42):
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    rows = []
    for sub in sorted(p for p in layerwise_dir.iterdir() if p.is_dir()):
        if "__" not in sub.name:
            continue
        model, layer = sub.name.split("__", 1)
        pred = sub / "test_predictions.txt"
        if not pred.exists():
            continue
        a = np.loadtxt(pred, comments="#")
        if a.ndim == 1:
            a = a.reshape(1, -1)
        target = a[:, 1]; p = a[:, 2]
        N = len(target)
        bs = np.zeros(n_boot); bsr = np.zeros(n_boot)
        for b in range(n_boot):
            sel = rng.integers(0, N, size=N)
            bs[b] = pearson(p[sel], target[sel])
            ranks_p = np.argsort(np.argsort(p[sel])).astype(float)
            ranks_t = np.argsort(np.argsort(target[sel])).astype(float)
            bsr[b] = pearson(ranks_p, ranks_t)
        depth = re.search(r"(\d+)", layer)
        d_key = int(depth.group(1)) if depth else 0
        rows.append({
            "model": model, "layer": layer, "depth_key": d_key,
            "n": int(N),
            "pearson_r": float(pearson(p, target)),
            "pearson_ci_lo": float(np.percentile(bs, 2.5)),
            "pearson_ci_hi": float(np.percentile(bs, 97.5)),
            "spearman_rho": float(pearson(np.argsort(np.argsort(p)).astype(float),
                                          np.argsort(np.argsort(target)).astype(float))),
            "spearman_ci_lo": float(np.percentile(bsr, 2.5)),
            "spearman_ci_hi": float(np.percentile(bsr, 97.5)),
        })
    rows.sort(key=lambda r: (r["model"], r["depth_key"]))

    with (out_dir / "layerwise_polish.csv").open("w") as f:
        f.write("model,layer,depth_key,n,pearson_r,pearson_ci_lo,pearson_ci_hi,"
                "spearman_rho,spearman_ci_lo,spearman_ci_hi\n")
        for r in rows:
            f.write(",".join([
                r["model"], r["layer"], str(r["depth_key"]), str(r["n"]),
                f"{r['pearson_r']:.4f}",
                f"{r['pearson_ci_lo']:.4f}", f"{r['pearson_ci_hi']:.4f}",
                f"{r['spearman_rho']:.4f}",
                f"{r['spearman_ci_lo']:.4f}", f"{r['spearman_ci_hi']:.4f}"]) + "\n")
    with (out_dir / "layerwise_polish.json").open("w") as f:
        json.dump(rows, f, indent=2)

    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)
    for m in by_model:
        by_model[m].sort(key=lambda r: r["depth_key"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key, ci_lo, ci_hi, label in [
        (axes[0], "pearson_r", "pearson_ci_lo", "pearson_ci_hi", "Pearson r"),
        (axes[1], "spearman_rho", "spearman_ci_lo", "spearman_ci_hi", "Spearman ρ"),
    ]:
        for m, pts in by_model.items():
            xs = [r["depth_key"] for r in pts]
            ys = [r[key] for r in pts]
            lo = [r[key] - r[ci_lo] for r in pts]
            hi = [r[ci_hi] - r[key] for r in pts]
            ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3, label=m)
        ax.set_xlabel("layer depth (block index)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("layer-wise vividness probe with 95% bootstrap CIs")
    fig.tight_layout()
    fig.savefig(out_dir / "layerwise_pearson_spearman_ci.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--noise-ceiling-json", required=True)
    ap.add_argument("--include", nargs="+",
                    default=["frozen_", "vlm_probe", "handcrafted"])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    models = gather(runs_dir, args.include)
    common, target, aligned = align(models)
    names = list(aligned.keys())
    print(f"[align] {len(common)} common test images, {len(names)} models")

    pc1_residual_experiment(common, target, aligned, names, out_dir / "pc1_residual")
    pc23_loadings(common, target, aligned, names, out_dir / "pc23_loadings")
    rank_stability(common, target, aligned, names, out_dir / "rank_stability",
                   n_boot=args.n_boot, seed=args.seed)
    noise_ceiling_plot(common, target, aligned, names,
                       args.noise_ceiling_json,
                       out_dir / "noise_ceiling")
    layerwise_dir = runs_dir / "layerwise_probes"
    if layerwise_dir.is_dir():
        layerwise_polish(layerwise_dir, out_dir / "layerwise_polish",
                         n_boot=min(2000, args.n_boot), seed=args.seed)
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
