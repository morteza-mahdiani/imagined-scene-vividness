#!/usr/bin/env python3
"""Cross-model agreement analysis.

Reads ``test_predictions.txt`` from every run dir, aligns on the common
test images, and prints / saves:

    * model-model Pearson correlation matrix
    * model-target Pearson correlation
    * a heatmap of both
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
        return a["h5_idx"].astype(int), a["target"].astype(float), a["score"].astype(float)
    a = np.loadtxt(p, comments="#")
    if a.ndim == 1:
        a = a.reshape(1, -1)
    return a[:, 0].astype(int), a[:, 1].astype(float), a[:, 2].astype(float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--include", nargs="+", default=None,
                    help="only these run names (substrings match)")
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    models = {}
    for d in sorted(Path(args.runs_dir).iterdir()):
        if not d.is_dir():
            continue
        if args.include and not any(s in d.name for s in args.include):
            continue
        r = load_preds(d)
        if r is None:
            continue
        idx, target, pred = r
        models[d.name] = (idx, target, pred)

    names = list(models.keys())
    if len(names) < 2:
        print("need >= 2 models"); return

    # align on intersection of test indices
    common = None
    for n in names:
        s = set(int(i) for i in models[n][0])
        common = s if common is None else common & s
    common = np.array(sorted(common))
    print(f"[align] {len(common)} common test images across {len(names)} runs", flush=True)

    # matrix of predictions and target
    P = np.zeros((len(names), len(common)))
    human = None
    for i, n in enumerate(names):
        idx_, target_, pred_ = models[n]
        table = {int(v): k for k, v in enumerate(idx_)}
        rows = np.array([table[int(i)] for i in common])
        P[i] = pred_[rows]
        if human is None:
            human = target_[rows]

    MM = np.corrcoef(P)
    MH = np.array([float(np.corrcoef(P[i], human)[0, 1]) for i in range(len(names))])

    # text summary
    lines = ["model\tr(model,human)"]
    for n, r in sorted(zip(names, MH), key=lambda x: -x[1]):
        lines.append(f"{n}\t{r:+.4f}")
    lines.append("")
    lines.append(f"mean inter-model r = {MM[np.triu_indices_from(MM, 1)].mean():+.4f}")
    lines.append(f"median inter-model r = {np.median(MM[np.triu_indices_from(MM, 1)]):+.4f}")
    lines.append(f"mean model-human r   = {MH.mean():+.4f}")
    text = "\n".join(lines)
    (out / "agreement_summary.txt").write_text(text + "\n")
    print(text)

    json.dump({
        "names": names,
        "n_common_test": int(len(common)),
        "mean_inter_model_r": float(MM[np.triu_indices_from(MM, 1)].mean()),
        "median_inter_model_r": float(np.median(MM[np.triu_indices_from(MM, 1)])),
        "model_human_r": dict(zip(names, [float(v) for v in MH])),
        "mean_model_human_r": float(MH.mean()),
    }, open(out / "agreement_summary.json", "w"), indent=2)

    # figure: combined heatmap
    fig, axes = plt.subplots(1, 2, figsize=(12, 6),
                             gridspec_kw={"width_ratios": [5, 1]})
    ax = axes[0]
    im = ax.imshow(MM, cmap="viridis", vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{MM[i, j]:.2f}", ha="center", va="center",
                    color="white" if MM[i, j] < 0.85 else "black", fontsize=6)
    ax.set_title(f"model <-> model Pearson (N={len(common)} test images)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax2 = axes[1]
    order = np.argsort(-MH)
    ax2.barh(range(len(names)), MH[order], color="#dd8452")
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels([names[o] for o in order], fontsize=8)
    ax2.set_xlim(0, 0.45)
    ax2.invert_yaxis()
    ax2.set_title("model <-> human r")
    for i, v in enumerate(MH[order]):
        ax2.text(v + 0.005, i, f"{v:+.3f}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "cross_model_agreement_full.png", dpi=150)
    plt.close(fig)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
