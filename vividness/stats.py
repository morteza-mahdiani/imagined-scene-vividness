"""The statistics reported for the paper's main results."""

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
    return float(_upper(values).mean())


def pairwise_median(values: np.ndarray) -> float:
    """Median of the off-diagonal upper triangle of a square matrix."""
    return float(np.median(_upper(values)))


def correlation_matrix(matrix: np.ndarray) -> np.ndarray:
    return np.corrcoef(matrix, rowvar=False)


def standardize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    scale = matrix.std(axis=0, ddof=0)
    if np.any(scale == 0):
        raise ValueError("every column needs nonzero variance")
    return (matrix - matrix.mean(axis=0)) / scale


def residualize(predictions: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Remove the linear contribution of one variable from each column."""
    target = np.asarray(target, dtype=float).ravel()
    design = np.column_stack([np.ones(len(target)), target])
    residuals = np.empty_like(predictions, dtype=float)
    for column in range(predictions.shape[1]):
        coefficients, *_ = np.linalg.lstsq(design, predictions[:, column], rcond=None)
        residuals[:, column] = predictions[:, column] - design @ coefficients
    return residuals


def r_squared(columns: np.ndarray, outcome: np.ndarray) -> float:
    """Variance explained by a linear fit with an intercept."""
    columns = np.asarray(columns, dtype=float)
    outcome = np.asarray(outcome, dtype=float).ravel()
    if columns.ndim == 1:
        columns = columns[:, None]
    design = np.column_stack([np.ones(len(outcome)), columns])
    coefficients, *_ = np.linalg.lstsq(design, outcome, rcond=None)
    residual = float(np.sum((outcome - design @ coefficients) ** 2))
    total = float(np.sum((outcome - outcome.mean()) ** 2))
    if total == 0.0:
        raise ValueError("r-squared is undefined")
    return 1.0 - residual / total


def zscore_trials(frame):
    """Z-score ratings within participant. Constant participants are dropped."""
    work = frame.loc[:, ["participant", "image", "rating"]].copy()
    grouped = work.groupby("participant")["rating"]
    center = grouped.transform("mean")
    count = grouped.transform("size")
    centered = work["rating"] - center
    scale = np.sqrt(centered.pow(2).groupby(work["participant"]).transform("sum") / count)
    keep = scale > 0
    work = work.loc[keep].copy()
    work["rating"] = centered.loc[keep] / scale.loc[keep]
    if work["participant"].nunique() < 2:
        raise ValueError("need at least two participants with varying ratings")
    return work


def image_target(frame):
    """Mean of within-participant z-scored ratings, one value per image."""
    scored = zscore_trials(frame)
    return scored.groupby("image", sort=False)["rating"].mean()


def shared_component(predictions: np.ndarray, target: np.ndarray) -> dict:
    """PCA of column-standardized predictions, with PC1 oriented toward the target."""
    target = np.asarray(target, dtype=float).ravel()
    if float(np.std(target)) == 0.0:
        raise ValueError("the target needs nonzero variance")
    standardized = standardize(predictions)
    covariance = (standardized.T @ standardized) / (standardized.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    explained = eigenvalues / eigenvalues.sum()
    scores = standardized @ eigenvectors
    loadings = eigenvectors.copy()
    sign = 1.0 if correlation(scores[:, 0], target) >= 0 else -1.0
    scores[:, 0] *= sign
    loadings[:, 0] *= sign
    prediction_r = correlation_matrix(predictions)
    residual_r = correlation_matrix(residualize(predictions, target))
    if not np.isfinite(pairwise_mean(residual_r)):
        raise ValueError("residual correlation is undefined")
    model_resid = residualize(predictions, scores[:, 0])
    human_resid = residualize(target.reshape(-1, 1), scores[:, 0]).ravel()
    r2_component = r_squared(scores[:, 0], target)
    beyond = []
    for index in range(predictions.shape[1]):
        residual_value = _optional_correlation(model_resid[:, index], human_resid)
        both = r_squared(np.column_stack([scores[:, 0], predictions[:, index]]), target)
        beyond.append(
            {
                "r_with_component": correlation(predictions[:, index], scores[:, 0]),
                "residual_r": residual_value,
                "delta_r2": both - r2_component,
            }
        )
    return {
        "explained_variance_ratio": explained,
        "scores": scores,
        "loadings": loadings,
        "pc_target_r": [correlation(scores[:, index], target) for index in range(scores.shape[1])],
        "model_target_r": [
            correlation(predictions[:, index], target) for index in range(predictions.shape[1])
        ],
        "mean_prediction_r": pairwise_mean(prediction_r),
        "median_prediction_r": pairwise_median(prediction_r),
        "mean_residual_r": pairwise_mean(residual_r),
        "median_residual_r": pairwise_median(residual_r),
        "beyond_first_component": beyond,
    }


def bootstrap_correlations(
    predictions: np.ndarray,
    target: np.ndarray,
    n_boot: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pearson and Spearman correlations on shared image resamples.

    Each array has shape (n_boot, n_models).
    """
    rng = np.random.default_rng(seed)
    n_images, n_models = predictions.shape
    pearson = np.empty((n_boot, n_models), dtype=float)
    spearman = np.empty((n_boot, n_models), dtype=float)
    for draw in range(n_boot):
        chosen = rng.integers(0, n_images, size=n_images)
        target_draw = target[chosen]
        prediction_draw = predictions[chosen]
        for model in range(n_models):
            column = prediction_draw[:, model]
            pearson[draw, model] = correlation(column, target_draw)
            spearman[draw, model] = correlation(column, target_draw, method="spearman")
    return pearson, spearman


def bootstrap_agreement(
    predictions: np.ndarray,
    target: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Image-resample intervals for the mean agreement statistics.

    Residuals are fit once on the full table, then the rows are resampled.
    """
    residuals = residualize(predictions, target)
    rng = np.random.default_rng(seed)
    n_images, n_models = predictions.shape
    prediction_mean = np.empty(n_boot, dtype=float)
    residual_mean = np.empty(n_boot, dtype=float)
    model_mean = np.empty(n_boot, dtype=float)
    for draw in range(n_boot):
        chosen = rng.integers(0, n_images, size=n_images)
        target_draw = target[chosen]
        prediction_mean[draw] = pairwise_mean(correlation_matrix(predictions[chosen]))
        residual_mean[draw] = pairwise_mean(correlation_matrix(residuals[chosen]))
        model_mean[draw] = float(
            np.mean(
                [
                    correlation(predictions[chosen, model], target_draw)
                    for model in range(n_models)
                ]
            )
        )
    return {
        "mean_prediction_r": prediction_mean,
        "mean_residual_r": residual_mean,
        "mean_model_target_r": model_mean,
    }


def split_half_ceiling(frame, n_splits: int, seed: int) -> dict:
    """Split-half reliability of the image-level target.

    Ratings are z-scored within participant. Participants are then divided
    into two halves, and each half contributes a per-image mean. The
    correlation of those means is projected to the full sample by the
    Spearman–Brown formula 2r / (1 + r). Its square root bounds the
    correlation an image-level prediction can reach.
    """
    scored = zscore_trials(frame)
    rng = np.random.default_rng(seed)
    participants = scored["participant"].unique()
    half_r = np.empty(n_splits, dtype=float)
    projected = np.empty(n_splits, dtype=float)
    bound = np.empty(n_splits, dtype=float)
    for split in range(n_splits):
        order = rng.permutation(participants)
        first = set(order[: len(order) // 2])
        left = scored[scored["participant"].isin(first)]
        right = scored[~scored["participant"].isin(first)]
        left_mean = left.groupby("image")["rating"].mean()
        right_mean = right.groupby("image")["rating"].mean()
        shared = left_mean.index.intersection(right_mean.index)
        value = correlation(left_mean.loc[shared].to_numpy(), right_mean.loc[shared].to_numpy())
        full = 2.0 * value / (1.0 + value)
        if full < 0.0 or not np.isfinite(full):
            raise ValueError("split-half correlation is undefined")
        half_r[split] = value
        projected[split] = full
        bound[split] = float(np.sqrt(full))
    return {
        "n_trials": int(len(scored)),
        "n_participants": int(len(participants)),
        "n_images": int(scored["image"].nunique()),
        "n_splits": int(n_splits),
        "split_half_r": _summary(half_r),
        "spearman_brown": _summary(projected),
        "correlation_bound": _summary(bound),
    }


def incremental_r2(features: np.ndarray, prediction: np.ndarray, target: np.ndarray) -> dict:
    """Variance a prediction adds on top of a standardized feature table."""
    features = standardize(features)
    prediction = np.asarray(prediction, dtype=float).ravel()
    target = np.asarray(target, dtype=float).ravel()
    controls = r_squared(features, target)
    alone = r_squared(prediction, target)
    combined = r_squared(np.column_stack([features, prediction]), target)
    return {
        "features": features,
        "r": correlation(prediction, target),
        "r2": alone,
        "r2_controls": controls,
        "r2_combined": combined,
        "delta_r2": combined - controls,
    }


def category_correlations(
    prediction: np.ndarray,
    target: np.ndarray,
    category: np.ndarray,
    min_size: int,
) -> dict:
    """Within-category correlation, and correlation after removing category means."""
    prediction = np.asarray(prediction, dtype=float).ravel()
    target = np.asarray(target, dtype=float).ravel()
    category = np.asarray(category)
    values = []
    weights = []
    demeaned_prediction = np.empty_like(prediction)
    demeaned_target = np.empty_like(target)
    for label in np.unique(category):
        chosen = category == label
        demeaned_prediction[chosen] = prediction[chosen] - prediction[chosen].mean()
        demeaned_target[chosen] = target[chosen] - target[chosen].mean()
        if int(chosen.sum()) < min_size:
            continue
        if float(np.std(prediction[chosen])) == 0.0 or float(np.std(target[chosen])) == 0.0:
            continue
        values.append(correlation(prediction[chosen], target[chosen]))
        weights.append(int(chosen.sum()))
    within = float(np.average(values, weights=weights)) if values else None
    demeaned = _optional_correlation(demeaned_prediction, demeaned_target)
    return {"within_category_r": within, "category_demeaned_r": demeaned}


def _upper(values: np.ndarray) -> np.ndarray:
    index = np.triu_indices(values.shape[0], k=1)
    return values[index]


def _optional_correlation(x: np.ndarray, y: np.ndarray):
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return None
    return correlation(x, y)


def _summary(values: np.ndarray) -> dict:
    return {
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "ci95": [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))],
    }
