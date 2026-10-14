"""Command line for the vividness analyses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from vividness.io import read_controls, read_predictions, read_trials
from vividness.stats import (
    bootstrap_agreement,
    bootstrap_correlations,
    category_correlations,
    correlation,
    image_target,
    incremental_r2,
    r_squared,
    shared_component,
    split_half_ceiling,
    zscore_trials,
)


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(path)


def _interval(values: np.ndarray) -> list[float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return [float(low), float(high)]


def _align_controls(image_ids, control_ids, features, category):
    lookup = {identifier: index for index, identifier in enumerate(control_ids)}
    missing = [identifier for identifier in image_ids if identifier not in lookup]
    if missing:
        raise SystemExit(f"control table is missing {len(missing)} images from the prediction table")
    rows = np.array([lookup[identifier] for identifier in image_ids])
    aligned_category = None if category is None else category[rows]
    return features[rows], aligned_category


def command_target(args: argparse.Namespace) -> None:
    frame = read_trials(args.trials, args.participant_column, args.image_column, args.rating_column)
    if args.zscore:
        frame = zscore_trials(frame)
    target = image_target(frame)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame({"image_id": target.index, "target": target.to_numpy()})
    table.to_csv(out, index=False)
    print(out)


def command_component(args: argparse.Namespace) -> None:
    _, target, names, predictions = read_predictions(args.predictions, args.id_column, args.target_column)
    result = shared_component(predictions, target)
    agreement = bootstrap_agreement(predictions, target, args.n_boot, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_images": int(len(target)),
        "n_boot": int(args.n_boot),
        "seed": int(args.seed),
        "models": names,
        "explained_variance_ratio": [float(value) for value in result["explained_variance_ratio"]],
        "pc_target_r": [float(value) for value in result["pc_target_r"]],
        "pc1_loadings": [float(value) for value in result["loadings"][:, 0]],
        "model_target_r": {name: float(value) for name, value in zip(names, result["model_target_r"])},
        "mean_prediction_r": result["mean_prediction_r"],
        "median_prediction_r": result["median_prediction_r"],
        "mean_residual_r": result["mean_residual_r"],
        "median_residual_r": result["median_residual_r"],
        "agreement_ci95": {name: _interval(values) for name, values in agreement.items()},
        "beyond_first_component": [
            {
                "model": name,
                "r_with_component": item["r_with_component"],
                "residual_r": item["residual_r"],
                "delta_r2": item["delta_r2"],
            }
            for name, item in zip(names, result["beyond_first_component"])
        ],
    }
    _write(out / "component.json", payload)

    figure, axis = plt.subplots(figsize=(5, 4))
    axis.bar(range(len(payload["explained_variance_ratio"])), payload["explained_variance_ratio"], color="#c47b4a")
    axis.set_xlabel("component")
    axis.set_ylabel("explained variance")
    figure.tight_layout()
    figure.savefig(out / "explained_variance.png", dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, max(2.5, 0.35 * len(names))))
    order = np.argsort(payload["pc1_loadings"])
    axis.barh(np.array(names)[order], np.array(payload["pc1_loadings"])[order], color="#3d5a80")
    axis.set_xlabel("loading on the first component")
    figure.tight_layout()
    figure.savefig(out / "loadings.png", dpi=150)
    plt.close(figure)

    shown = min(5, len(payload["pc_target_r"]))
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.bar(range(shown), payload["pc_target_r"][:shown], color="#3d5a80")
    axis.set_xlabel("component")
    axis.set_ylabel("correlation with the target")
    figure.tight_layout()
    figure.savefig(out / "components_vs_target.png", dpi=150)
    plt.close(figure)


def command_bootstrap(args: argparse.Namespace) -> None:
    _, target, names, predictions = read_predictions(args.predictions, args.id_column, args.target_column)
    pearson, spearman = bootstrap_correlations(predictions, target, args.n_boot, args.seed)
    rows = []
    for index, name in enumerate(names):
        rows.append(
            {
                "model": name,
                "n": int(len(target)),
                "pearson_r": correlation(predictions[:, index], target),
                "pearson_ci95": _interval(pearson[:, index]),
                "spearman_r": correlation(predictions[:, index], target, method="spearman"),
                "spearman_ci95": _interval(spearman[:, index]),
            }
        )
    rows.sort(key=lambda row: row["pearson_r"], reverse=True)
    pairs = []
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            difference = pearson[:, left] - pearson[:, right]
            pairs.append(
                {
                    "model_a": names[left],
                    "model_b": names[right],
                    "delta_r": float(
                        correlation(predictions[:, left], target) - correlation(predictions[:, right], target)
                    ),
                    "ci95": _interval(difference),
                }
            )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write(
        out / "bootstrap.json",
        {"n_boot": int(args.n_boot), "seed": int(args.seed), "models": rows, "paired_delta_r": pairs},
    )

    order = np.argsort([-correlation(predictions[:, index], target) for index in range(len(names))])
    delta = np.full((len(names), len(names)), np.nan)
    for left in range(len(names)):
        for right in range(len(names)):
            if left == right:
                continue
            delta[left, right] = correlation(predictions[:, left], target) - correlation(predictions[:, right], target)
    delta = delta[np.ix_(order, order)]
    ordered_names = [names[index] for index in order]
    figure, axis = plt.subplots(figsize=(max(5, 0.45 * len(names)), max(4, 0.45 * len(names))))
    limit = max(0.05, float(np.nanmax(np.abs(delta))))
    image = axis.imshow(delta, cmap="RdBu_r", vmin=-limit, vmax=limit)
    axis.set_xticks(range(len(names)), ordered_names, rotation=45, ha="right")
    axis.set_yticks(range(len(names)), ordered_names)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(out / "paired_delta.png", dpi=150)
    plt.close(figure)


def command_ceiling(args: argparse.Namespace) -> None:
    frame = read_trials(args.trials, args.participant_column, args.image_column, args.rating_column)
    if args.zscore:
        frame = zscore_trials(frame)
    payload = split_half_ceiling(frame, args.n_splits, args.seed)
    payload["seed"] = int(args.seed)
    _write(Path(args.out), payload)


def command_controls(args: argparse.Namespace) -> None:
    image_ids, target, names, predictions = read_predictions(
        args.predictions, args.id_column, args.target_column
    )
    control_ids, features, feature_names, category = read_controls(
        args.controls, args.id_column, args.category_column
    )
    features, category = _align_controls(image_ids, control_ids, features, category)
    rng = np.random.default_rng(args.seed)
    n_images = len(target)
    rows = []
    r2_controls = None
    for index, name in enumerate(names):
        fitted = incremental_r2(features, predictions[:, index], target)
        r2_controls = fitted["r2_controls"]
        draws = np.empty(args.n_boot, dtype=float)
        standardized = fitted["features"]
        for draw in range(args.n_boot):
            chosen = rng.integers(0, n_images, size=n_images)
            controls_only = r_squared(standardized[chosen], target[chosen])
            combined = r_squared(
                np.column_stack([standardized[chosen], predictions[chosen, index]]),
                target[chosen],
            )
            draws[draw] = combined - controls_only
        row = {
            "model": name,
            "n": int(n_images),
            "r": fitted["r"],
            "r2": fitted["r2"],
            "r2_combined": fitted["r2_combined"],
            "delta_r2": fitted["delta_r2"],
            "delta_r2_ci95": _interval(draws),
        }
        if category is not None:
            row.update(category_correlations(predictions[:, index], target, category, args.min_category_size))
        rows.append(row)
    rows.sort(key=lambda row: row["r"], reverse=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write(
        out / "controls.json",
        {
            "n_images": int(n_images),
            "n_features": len(feature_names),
            "features": feature_names,
            "r2_controls": r2_controls,
            "n_boot": int(args.n_boot),
            "seed": int(args.seed),
            "models": rows,
        },
    )
    figure, axis = plt.subplots(figsize=(6, max(2.5, 0.35 * len(rows))))
    axis.barh(
        [row["model"] for row in rows][::-1],
        [row["delta_r2"] for row in rows][::-1],
        color="#c47b4a",
    )
    axis.set_xlabel("added variance")
    figure.tight_layout()
    figure.savefig(out / "delta_r2.png", dpi=150)
    plt.close(figure)


def _trials(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--participant-column", default="participant")
    parser.add_argument("--image-column", default="image")
    parser.add_argument("--rating-column", default="rating")
    parser.add_argument("--zscore", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser(description="Imagined-scene vividness analyses.")
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--predictions", type=Path, required=True)
    shared.add_argument("--id-column", default="image_id")
    shared.add_argument("--target-column", default="target")
    commands = parser.add_subparsers(dest="command", required=True)

    target = commands.add_parser("target")
    _trials(target)
    target.add_argument("--out", type=Path, required=True)
    target.set_defaults(func=command_target)

    component = commands.add_parser("component", parents=[shared])
    component.add_argument("--out", type=Path, required=True)
    component.add_argument("--n-boot", type=int, default=1000)
    component.add_argument("--seed", type=int, default=42)
    component.set_defaults(func=command_component)

    bootstrap = commands.add_parser("bootstrap", parents=[shared])
    bootstrap.add_argument("--out", type=Path, required=True)
    bootstrap.add_argument("--n-boot", type=int, default=10000)
    bootstrap.add_argument("--seed", type=int, default=42)
    bootstrap.set_defaults(func=command_bootstrap)

    ceiling = commands.add_parser("ceiling")
    _trials(ceiling)
    ceiling.add_argument("--out", type=Path, required=True)
    ceiling.add_argument("--n-splits", type=int, default=50)
    ceiling.add_argument("--seed", type=int, default=42)
    ceiling.set_defaults(func=command_ceiling)

    controls = commands.add_parser("controls", parents=[shared])
    controls.add_argument("--controls", type=Path, required=True)
    controls.add_argument("--category-column", default=None)
    controls.add_argument("--min-category-size", type=int, default=50)
    controls.add_argument("--out", type=Path, required=True)
    controls.add_argument("--n-boot", type=int, default=1000)
    controls.add_argument("--seed", type=int, default=42)
    controls.set_defaults(func=command_controls)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
