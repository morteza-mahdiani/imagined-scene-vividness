#!/usr/bin/env python3
"""Incremental R^2 of model predictions over interpretable image controls,
plus a category-controlled analysis using a clustered pseudo-category id.

Inputs:
    --controls-npz : output of build_image_controls.py (--merge)
    --runs-dir     : runs root containing test_predictions.txt for each model
    --include      : run-name substrings (default frozen_, vlm_probe, handcrafted)

For each model with predictions on the test split we report:
    r(controls -> y) on test
    r(model -> y) on test
    r([controls, model] -> y) on test
    delta_r2 = r2(combined) - r2(controls)
    incremental F-test approximate p-value (image-bootstrap)

Category-controlled (cluster_id_qwen):
    - within-category Pearson r (avg across categories with >= K images)
    - category-demeaned r: corr(y - <y|cat>, p - <p|cat>)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


def load_preds(run_dir: Path):
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


def gather_runs(runs_dir, include):
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


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def ols_r2(X, y):
    X = np.c_[X, np.ones(len(X))]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum() + 1e-12)
    return 1.0 - ss_res / ss_tot, yhat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controls-npz", required=True)
    ap.add_argument("--runs-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--include", nargs="+",
                    default=["frozen_", "vlm_probe", "handcrafted"])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-cat-size", type=int, default=50)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    z = np.load(args.controls_npz, allow_pickle=True)
    feat_names = list(z["feature_names"])
    h5_idx = z["h5_idx"].astype(np.int64)
    feat_mat = np.stack([z[n] for n in feat_names], axis=1).astype(np.float64)
    cluster_id = z["cluster_id_qwen"].astype(np.int64) if "cluster_id_qwen" in z.files else None
    print(f"[controls] N={len(h5_idx)}  features={len(feat_names)}")

    feat_lookup = {int(i): k for k, i in enumerate(h5_idx)}

    rng = np.random.default_rng(args.seed)
    runs = gather_runs(Path(args.runs_dir), args.include)
    if not runs:
        raise SystemExit("no runs found")

    rows = []
    for name, (idx, target, pred) in runs.items():
        keep = [k for k, i in enumerate(idx) if int(i) in feat_lookup]
        if len(keep) < 200:
            print(f"[skip] {name}: only {len(keep)} matched controls")
            continue
        keep = np.array(keep)
        idx_k = idx[keep]; y = target[keep]; p = pred[keep]
        rows_idx = np.array([feat_lookup[int(i)] for i in idx_k])
        F = feat_mat[rows_idx]
        Fz = (F - F.mean(0)) / (F.std(0) + 1e-12)

        r_ctrl = pearson(F @ np.linalg.lstsq(np.c_[Fz, np.ones(len(F))], y, rcond=None)[0][:-1], y)
        r2_ctrl, _ = ols_r2(Fz, y)
        r_pred = pearson(p, y)
        r2_pred, _ = ols_r2(p[:, None], y)
        Xc = np.c_[Fz, p[:, None]]
        r2_combo, _ = ols_r2(Xc, y)
        delta = r2_combo - r2_ctrl

        boot_delta = np.zeros(args.n_boot)
        for b in range(args.n_boot):
            sel = rng.integers(0, len(y), size=len(y))
            yb = y[sel]; Fzb = Fz[sel]; pb = p[sel]
            r2c, _ = ols_r2(Fzb, yb)
            r2j, _ = ols_r2(np.c_[Fzb, pb[:, None]], yb)
            boot_delta[b] = r2j - r2c
        ci_lo, ci_hi = np.percentile(boot_delta, [2.5, 97.5])

        within_cat = None
        demeaned_r = None
        if cluster_id is not None:
            cid = cluster_id[rows_idx]
            mask = cid >= 0
            if mask.sum() >= 100:
                cs = cid[mask]; ys = y[mask]; ps = p[mask]
                cat_means_y = {}; cat_means_p = {}; cat_ys = {}; cat_ps = {}
                for c in np.unique(cs):
                    m = cs == c
                    if m.sum() >= args.min_cat_size:
                        cat_ys[c] = ys[m]; cat_ps[c] = ps[m]
                        cat_means_y[c] = float(ys[m].mean())
                        cat_means_p[c] = float(ps[m].mean())
                if cat_ys:
                    rs = []
                    weights = []
                    for c in cat_ys:
                        if cat_ys[c].std() < 1e-9 or cat_ps[c].std() < 1e-9:
                            continue
                        rs.append(pearson(cat_ys[c], cat_ps[c]))
                        weights.append(len(cat_ys[c]))
                    if rs:
                        within_cat = float(np.average(rs, weights=weights))
                    y_dm = np.zeros_like(ys); p_dm = np.zeros_like(ps)
                    for c in np.unique(cs):
                        m = cs == c
                        y_dm[m] = ys[m] - ys[m].mean()
                        p_dm[m] = ps[m] - ps[m].mean()
                    demeaned_r = pearson(y_dm, p_dm)

        rows.append({
            "model": name,
            "n": int(len(y)),
            "r_controls_lstsq": float(r_ctrl),
            "r2_controls": float(r2_ctrl),
            "r_pred": float(r_pred),
            "r2_pred": float(r2_pred),
            "r2_combined": float(r2_combo),
            "delta_r2": float(delta),
            "delta_r2_ci_lo": float(ci_lo),
            "delta_r2_ci_hi": float(ci_hi),
            "within_category_r": within_cat,
            "category_demeaned_r": demeaned_r,
        })

    rows.sort(key=lambda r: -r["r_pred"])
    with (out_dir / "controls_regression.csv").open("w") as f:
        f.write(
            "model,n,r_controls,r2_controls,r_pred,r2_pred,r2_combined,delta_r2,"
            "delta_r2_ci_lo,delta_r2_ci_hi,within_category_r,category_demeaned_r\n"
        )
        for r in rows:
            f.write(
                f"{r['model']},{r['n']},{r['r_controls_lstsq']:.4f},{r['r2_controls']:.4f},"
                f"{r['r_pred']:.4f},{r['r2_pred']:.4f},{r['r2_combined']:.4f},"
                f"{r['delta_r2']:.4f},{r['delta_r2_ci_lo']:.4f},{r['delta_r2_ci_hi']:.4f},"
                f"{r['within_category_r'] if r['within_category_r'] is not None else 'NA'},"
                f"{r['category_demeaned_r'] if r['category_demeaned_r'] is not None else 'NA'}\n"
            )

    summary = {
        "n_features": len(feat_names),
        "feature_names": feat_names,
        "min_cat_size": args.min_cat_size,
        "rows": rows,
    }
    with (out_dir / "controls_regression.json").open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
