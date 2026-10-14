#!/usr/bin/env python3
"""Shared-component analysis for per-model vividness predictions.

The numerical procedure follows ``scripts/pca_residual_analysis.py`` in the
research codebase. Commits in that repository are authored by Morteza
Mahdiani. This script is a reduced copy for a public example: it reads
prediction tables, summarizes agreement with a target, extracts the leading
principal component of the standardized predictions, and measures residual
agreement after each prediction is linearly regressed on the target.

It does not train encoders, read images, or reproduce the statistics
reported in the paper. The checked-in inputs under ``example/synthetic_runs``
are simulated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def write_synthetic(root: Path, n: int = 400, seed: int = 0) -> None:
    """Write four simulated prediction tables with a known shared factor.

    ``shared`` is a latent image factor. The target and three models are
    noisy linear observations of that factor; ``model_noise`` is independent
    of it. Coefficients are fixed so a later run can check the direction of
    the result. These are not human ratings and not model outputs.
    """
    rng = np.random.default_rng(seed)
    shared = rng.normal(size=n)
    target = 0.8 * shared + 0.6 * rng.normal(size=n)
    specs = {
        "model_shared_strong": 1.0 * shared + 0.3 * rng.normal(size=n),
        "model_shared_mid": 0.7 * shared + 0.5 * rng.normal(size=n),
        "model_shared_weak": 0.4 * shared + 0.8 * rng.normal(size=n),
        "model_noise": rng.normal(size=n),
    }
    root.mkdir(parents=True, exist_ok=True)
    header = (
        "# synthetic example, not human data\n"
        "# columns: synthetic_idx target pred\n"
        "# generator: example/run_shared_component.py --write-synthetic "
        f"--seed {seed} --n-images {n}\n"
    )
    idx = np.arange(n, dtype=int)
    for name, pred in specs.items():
        dest = root / name
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / "test_predictions.txt"
        body = "".join(
            f"{int(i)} {float(y):.8f} {float(p):.8f}\n"
            for i, y, p in zip(idx, target, pred)
        )
        path.write_text(header + body)


def load_preds(run_dir: Path):
    path = run_dir / "test_predictions.txt"
    if not path.is_file():
        path = run_dir / "test_predictions.csv"
    if not path.is_file():
        return None
    if path.suffix == ".csv":
        table = np.genfromtxt(path, delimiter=",", names=True)
        pred_field = "score" if "score" in table.dtype.names else "pred"
        return (
            table["h5_idx"].astype(int),
            table["target"].astype(float),
            table[pred_field].astype(float),
        )
    table = np.loadtxt(path, comments="#")
    if table.ndim == 1:
        table = table.reshape(1, -1)
    if table.shape[1] < 3:
        raise SystemExit(f"{path} needs three columns: index, target, prediction")
    return table[:, 0].astype(int), table[:, 1].astype(float), table[:, 2].astype(float)


def gather_runs(runs_dir: Path):
    models = {}
    for directory in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        loaded = load_preds(directory)
        if loaded is not None:
            models[directory.name] = loaded
    return models


def align_to_common(models):
    common = None
    for idx, _, _ in models.values():
        keys = set(int(i) for i in idx)
        common = keys if common is None else common & keys
    if not common:
        raise SystemExit("no common image indices across prediction files")
    common = np.array(sorted(common))
    aligned = {}
    target_ref = None
    for name, (idx, target, pred) in models.items():
        table = {int(value): row for row, value in enumerate(idx)}
        rows = np.array([table[int(i)] for i in common])
        aligned[name] = pred[rows]
        if target_ref is None:
            target_ref = target[rows]
        elif not np.allclose(target_ref, target[rows]):
            raise SystemExit(f"target column in {name} disagrees with the other files")
    return common, target_ref, aligned


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return pearson(np.argsort(np.argsort(a)), np.argsort(np.argsort(b)))


def pearson_columns(matrix: np.ndarray) -> np.ndarray:
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(centered, axis=0) + 1e-12
    return (centered.T @ centered) / np.outer(norms, norms)


def residualize_on_target(predictions: np.ndarray, target: np.ndarray) -> np.ndarray:
    residuals = np.zeros_like(predictions)
    centered_target = target - target.mean()
    denom = float(np.dot(centered_target, centered_target)) + 1e-12
    for i in range(predictions.shape[1]):
        column = predictions[:, i]
        beta = float(np.dot(centered_target, column - column.mean()) / denom)
        alpha = float(column.mean() - beta * target.mean())
        residuals[:, i] = column - (alpha + beta * target)
    return residuals


def bootstrap_means(predictions, residuals, target, n_boot: int, seed: int):
    if n_boot <= 0:
        return None
    rng = np.random.default_rng(seed)
    n_images, n_models = predictions.shape
    upper = np.triu_indices(n_models, 1)
    pred_means = np.empty(n_boot)
    resid_means = np.empty(n_boot)
    model_target_means = np.empty(n_boot)
    for draw in range(n_boot):
        sel = rng.integers(0, n_images, size=n_images)
        pred_draw = predictions[sel]
        resid_draw = residuals[sel]
        target_draw = target[sel]
        pred_means[draw] = pearson_columns(pred_draw)[upper].mean()
        resid_means[draw] = pearson_columns(resid_draw)[upper].mean()
        model_target_means[draw] = np.mean(
            [pearson(pred_draw[:, i], target_draw) for i in range(n_models)]
        )

    def ci(values):
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    return {
        "n_boot": int(n_boot),
        "seed": int(seed),
        "mean_pred_pred_r_ci95": ci(pred_means),
        "mean_resid_resid_r_ci95": ci(resid_means),
        "mean_model_target_r_ci95": ci(model_target_means),
    }


def analyze(runs_dir: Path, out_dir: Path, n_boot: int, seed: int) -> dict:
    models = gather_runs(runs_dir)
    if len(models) < 3:
        raise SystemExit(f"need at least 3 models with prediction files, found {list(models)}")
    common, target, aligned = align_to_common(models)
    names = list(aligned)
    predictions = np.column_stack([aligned[name] for name in names])
    standardized = (predictions - predictions.mean(axis=0)) / (predictions.std(axis=0) + 1e-12)

    covariance = (standardized.T @ standardized) / (standardized.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    explained = eigenvalues / max(float(eigenvalues.sum()), 1e-12)
    scores = standardized @ eigenvectors
    raw_pc1_target_r = pearson(scores[:, 0], target)
    # The sign of a principal component is arbitrary. Align it so the
    # reported correlation with the target is non-negative, matching the
    # sign-aligned loadings written by the research analysis.
    sign = 1.0 if raw_pc1_target_r >= 0 else -1.0
    pc1_target_r = sign * raw_pc1_target_r

    residuals = residualize_on_target(predictions, target)
    pred_corr = pearson_columns(predictions)
    resid_corr = pearson_columns(residuals)
    upper = np.triu_indices(len(names), 1)
    model_target = [
        {"model": name, "pearson_r": pearson(predictions[:, i], target), "spearman_r": spearman(predictions[:, i], target)}
        for i, name in enumerate(names)
    ]
    summary = {
        "input": str(runs_dir),
        "synthetic_example": all(name.startswith("model_") for name in names),
        "n_images": int(len(common)),
        "n_models": int(len(names)),
        "models": names,
        "per_model_vs_target": model_target,
        "explained_variance_ratio": [float(value) for value in explained],
        "pc1_vs_target_r": float(pc1_target_r),
        "pc1_loadings_sign_aligned_to_target": [float(value) for value in (sign * eigenvectors[:, 0])],
        "mean_pred_pred_r": float(pred_corr[upper].mean()),
        "mean_resid_resid_r": float(resid_corr[upper].mean()),
        "mean_model_target_r": float(np.mean([row["pearson_r"] for row in model_target])),
        "bootstrap": bootstrap_means(predictions, residuals, target, n_boot, seed),
        "scope": (
            "Agreement among prediction columns and their leading principal "
            "component. Not a reproduction of the paper's reported coefficients."
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (out_dir / "residual_agreement.csv").open("w") as handle:
        handle.write("model_a,model_b,pred_pred_r,resid_resid_r\n")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                handle.write(
                    f"{names[i]},{names[j]},{pred_corr[i, j]:.6f},{resid_corr[i, j]:.6f}\n"
                )
    return summary


def expect_synthetic(summary: dict) -> None:
    by_name = {row["model"]: row["pearson_r"] for row in summary["per_model_vs_target"]}
    required = {"model_shared_strong", "model_shared_mid", "model_shared_weak", "model_noise"}
    missing = required.difference(by_name)
    if missing:
        raise SystemExit(f"--expect-synthetic did not find {sorted(missing)}")
    if not (by_name["model_shared_strong"] > by_name["model_shared_mid"] > by_name["model_shared_weak"]):
        raise SystemExit(f"synthetic ordering failed: {by_name}")
    if abs(by_name["model_noise"]) > 0.2:
        raise SystemExit(f"noise model correlation is unexpectedly large: {by_name['model_noise']}")
    if summary["explained_variance_ratio"][0] <= summary["explained_variance_ratio"][1]:
        raise SystemExit("PC1 does not explain more variance than PC2 on the synthetic input")
    if summary["mean_resid_resid_r"] <= 0:
        raise SystemExit("expected residual agreement among the synthetic shared models")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "synthetic_runs",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "output",
    )
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-images", type=int, default=400)
    parser.add_argument("--write-synthetic", action="store_true")
    parser.add_argument("--expect-synthetic", action="store_true")
    args = parser.parse_args()

    if args.write_synthetic:
        write_synthetic(args.predictions_dir, n=args.n_images, seed=args.seed)
        print(f"wrote synthetic predictions to {args.predictions_dir}")

    summary = analyze(args.predictions_dir, args.out_dir, args.n_boot, args.seed)
    if args.expect_synthetic:
        expect_synthetic(summary)
    brief = {
        "n_images": summary["n_images"],
        "n_models": summary["n_models"],
        "pc1_explained_variance_ratio": summary["explained_variance_ratio"][0],
        "pc1_vs_target_r": summary["pc1_vs_target_r"],
        "mean_pred_pred_r": summary["mean_pred_pred_r"],
        "mean_resid_resid_r": summary["mean_resid_resid_r"],
        "per_model_pearson_r": {
            row["model"]: row["pearson_r"] for row in summary["per_model_vs_target"]
        },
    }
    print(json.dumps(brief, indent=2))
    print(f"wrote {args.out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
