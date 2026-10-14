"""Correlations, the shared component, and the split-half ceiling."""

from __future__ import annotations

import numpy as np
from scipy import stats


def correlation(x: np.ndarray, y: np.ndarray, method: str = "pearson") -> float:
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if method == "pearson":
        value = stats.pearsonr(x, y).statistic
    elif method == "spearman":
        value = stats.spearmanr(x, y).statistic
    else:
        raise ValueError(method)
    if not np.isfinite(value):
        raise ValueError(f"{method} correlation is undefined")
    return float(value)


def pairwise_mean(values: np.ndarray) -> float:
    """Mean of the off-diagonal upper triangle of a square matrix."""
    index = np.triu_indices(values.shape[0], k=1)
    return float(values[index].mean())


def correlation_matrix(matrix: np.ndarray) -> np.ndarray:
    return np.corrcoef(matrix, rowvar=False)


def residualize(predictions: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Remove the linear contribution of the target from each column."""
    design = np.column_stack([np.ones(len(target)), target])
    residuals = np.empty_like(predictions, dtype=float)
    for column in range(predictions.shape[1]):
        coefficients, *_ = np.linalg.lstsq(design, predictions[:, column], rcond=None)
        residuals[:, column] = predictions[:, column] - design @ coefficients
    return residuals


def shared_component(predictions: np.ndarray, target: np.ndarray) -> dict:
    """PCA of column-standardized predictions, with PC1 oriented toward the target."""
    scale = predictions.std(axis=0, ddof=0)
    if np.any(scale == 0) or float(np.std(target)) == 0.0:
        raise ValueError("the target and every prediction column need nonzero variance")
    standardized = (predictions - predictions.mean(axis=0)) / scale
    _, singular_values, axes = np.linalg.svd(standardized, full_matrices=False)
    explained = singular_values**2
    explained = explained / explained.sum()
    scores = standardized @ axes.T
    loadings = axes.T
    sign = 1.0 if correlation(scores[:, 0], target) >= 0 else -1.0
    scores[:, 0] *= sign
    loadings[:, 0] *= sign
    model_target = [
        correlation(predictions[:, index], target) for index in range(predictions.shape[1])
    ]
    residual_r = pairwise_mean(correlation_matrix(residualize(predictions, target)))
    if not np.isfinite(residual_r):
        raise ValueError("residual correlation is undefined")
    return {
        "explained_variance_ratio": explained,
        "scores": scores,
        "loadings": loadings,
        "pc1_target_r": correlation(scores[:, 0], target),
        "model_target_r": model_target,
        "mean_prediction_r": pairwise_mean(correlation_matrix(predictions)),
        "mean_residual_r": residual_r,
    }


def bootstrap_correlations(
    predictions: np.ndarray,
    target: np.ndarray,
    n_boot: int,
    seed: int,
) -> np.ndarray:
    """Pearson correlations on shared image resamples. Shape (n_boot, n_models)."""
    rng = np.random.default_rng(seed)
    n_images, n_models = predictions.shape
    samples = np.empty((n_boot, n_models), dtype=float)
    for draw in range(n_boot):
        chosen = rng.integers(0, n_images, size=n_images)
        target_draw = target[chosen]
        for model in range(n_models):
            samples[draw, model] = correlation(predictions[chosen, model], target_draw)
    return samples


def split_half_ceiling(frame, n_splits: int, seed: int) -> dict:
    """Participant split-half reliability and its Spearman–Brown projection.

    Participants are divided into two halves. Each half contributes a
    per-image mean. The correlation of those means, on images observed in
    both halves, is projected to the full participant sample by the
    Spearman–Brown formula 2r / (1 + r).
    """
    rng = np.random.default_rng(seed)
    participants = frame["participant"].unique()
    if len(participants) < 2:
        raise ValueError("need at least two participants")
    half_r = np.empty(n_splits, dtype=float)
    projected = np.empty(n_splits, dtype=float)
    for split in range(n_splits):
        order = rng.permutation(participants)
        first = set(order[: len(order) // 2])
        left = frame[frame["participant"].isin(first)]
        right = frame[~frame["participant"].isin(first)]
        left_mean = left.groupby("image")["rating"].mean()
        right_mean = right.groupby("image")["rating"].mean()
        shared = left_mean.index.intersection(right_mean.index)
        value = correlation(left_mean.loc[shared].to_numpy(), right_mean.loc[shared].to_numpy())
        half_r[split] = value
        projected[split] = 2.0 * value / (1.0 + value)
    return {
        "n_trials": int(len(frame)),
        "n_participants": int(len(participants)),
        "n_images": int(frame["image"].nunique()),
        "n_splits": int(n_splits),
        "split_half_r": _summary(half_r),
        "spearman_brown_r": _summary(projected),
    }


def _summary(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
    }
