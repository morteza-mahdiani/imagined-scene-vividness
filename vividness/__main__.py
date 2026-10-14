"""Command line for the vividness analyses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vividness.io import read_predictions, read_trials
from vividness.stats import bootstrap_correlations, correlation, shared_component, split_half_ceiling


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(path)


def command_component(args: argparse.Namespace) -> None:
    _, target, names, predictions = read_predictions(args.predictions, args.id_column, args.target_column)
    result = shared_component(predictions, target)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_images": int(len(target)),
        "models": names,
        "explained_variance_ratio": [float(value) for value in result["explained_variance_ratio"]],
        "pc1_target_r": result["pc1_target_r"],
        "pc1_loadings": [float(value) for value in result["loadings"][:, 0]],
        "model_target_r": {name: float(value) for name, value in zip(names, result["model_target_r"])},
        "mean_prediction_r": result["mean_prediction_r"],
        "mean_residual_r": result["mean_residual_r"],
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

    figure, axis = plt.subplots(figsize=(5, 4))
    axis.scatter(target, result["scores"][:, 0], s=8, alpha=0.45, color="#3d5a80")
    axis.set_xlabel("target")
    axis.set_ylabel("first component")
    figure.tight_layout()
    figure.savefig(out / "component_vs_target.png", dpi=150)
    plt.close(figure)


def command_bootstrap(args: argparse.Namespace) -> None:
    _, target, names, predictions = read_predictions(args.predictions, args.id_column, args.target_column)
    samples = bootstrap_correlations(predictions, target, args.n_boot, args.seed)
    rows = []
    for index, name in enumerate(names):
        point = correlation(predictions[:, index], target)
        low, high = np.percentile(samples[:, index], [2.5, 97.5])
        rows.append(
            {
                "model": name,
                "n": int(len(target)),
                "pearson_r": point,
                "ci95": [float(low), float(high)],
            }
        )
    rows.sort(key=lambda row: row["pearson_r"], reverse=True)
    pairs = []
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            difference = samples[:, left] - samples[:, right]
            low, high = np.percentile(difference, [2.5, 97.5])
            pairs.append(
                {
                    "model_a": names[left],
                    "model_b": names[right],
                    "delta_r": float(difference.mean()),
                    "ci95": [float(low), float(high)],
                }
            )
    out = Path(args.out)
    _write(
        out / "bootstrap.json",
        {"n_boot": int(args.n_boot), "seed": int(args.seed), "models": rows, "paired_delta_r": pairs},
    )


def command_ceiling(args: argparse.Namespace) -> None:
    frame = read_trials(args.trials, args.participant_column, args.image_column, args.rating_column)
    payload = split_half_ceiling(frame, args.n_splits, args.seed)
    payload["seed"] = int(args.seed)
    _write(Path(args.out), payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Imagined-scene vividness analyses.")
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--predictions", type=Path, required=True)
    shared.add_argument("--id-column", default="image_id")
    shared.add_argument("--target-column", default="target")
    commands = parser.add_subparsers(dest="command", required=True)

    component = commands.add_parser("component", parents=[shared])
    component.add_argument("--out", type=Path, required=True)
    component.set_defaults(func=command_component)

    bootstrap = commands.add_parser("bootstrap", parents=[shared])
    bootstrap.add_argument("--out", type=Path, required=True)
    bootstrap.add_argument("--n-boot", type=int, default=1000)
    bootstrap.add_argument("--seed", type=int, default=0)
    bootstrap.set_defaults(func=command_bootstrap)

    ceiling = commands.add_parser("ceiling")
    ceiling.add_argument("--trials", type=Path, required=True)
    ceiling.add_argument("--out", type=Path, required=True)
    ceiling.add_argument("--participant-column", default="participant")
    ceiling.add_argument("--image-column", default="image")
    ceiling.add_argument("--rating-column", default="rating")
    ceiling.add_argument("--n-splits", type=int, default=50)
    ceiling.add_argument("--seed", type=int, default=0)
    ceiling.set_defaults(func=command_ceiling)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
