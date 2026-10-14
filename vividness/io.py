"""Load the tables used by the analyses."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_predictions(path: Path, id_column: str, target_column: str):
    """Return image ids, the target, model names, and a prediction matrix.

    The file is a CSV. One column identifies images, one column is the
    target, and every remaining column is a model's prediction. Image
    identifiers are unique.
    """
    frame = pd.read_csv(path)
    missing = {id_column, target_column} - set(frame.columns)
    if missing:
        raise SystemExit(f"{path} is missing columns: {sorted(missing)}")
    model_names = [column for column in frame.columns if column not in (id_column, target_column)]
    if len(model_names) < 2:
        raise SystemExit("provide at least two model columns")
    if frame[id_column].duplicated().any():
        raise SystemExit(f"duplicate values in {id_column}")
    target = frame[target_column].to_numpy(dtype=float)
    predictions = frame[model_names].to_numpy(dtype=float)
    if not (pd.notna(target).all() and pd.notna(frame[model_names]).all().all()):
        raise SystemExit("prediction table contains missing values")
    return frame[id_column].to_numpy(), target, model_names, predictions


def read_trials(path: Path, participant_column: str, image_column: str, rating_column: str):
    """Return a trial table with one rating per participant and image."""
    frame = pd.read_csv(path, usecols=[participant_column, image_column, rating_column])
    frame = frame.dropna()
    if frame.empty:
        raise SystemExit(f"{path} has no complete trials")
    return frame.rename(
        columns={
            participant_column: "participant",
            image_column: "image",
            rating_column: "rating",
        }
    )


def read_controls(path: Path, id_column: str, category_column: str | None):
    """Return image ids, a feature matrix, feature names, and an optional category.

    Every column other than the image id and the category is a feature.
    Image identifiers are unique.
    """
    frame = pd.read_csv(path)
    if id_column not in frame.columns:
        raise SystemExit(f"{path} is missing column: {id_column}")
    if category_column and category_column not in frame.columns:
        raise SystemExit(f"{path} is missing column: {category_column}")
    if frame[id_column].duplicated().any():
        raise SystemExit(f"duplicate values in {id_column}")
    skipped = {id_column}
    if category_column:
        skipped.add(category_column)
    names = [column for column in frame.columns if column not in skipped]
    if not names:
        raise SystemExit("provide at least one control column")
    features = frame[names].apply(pd.to_numeric, errors="coerce")
    if not features.notna().all().all():
        raise SystemExit("control table contains missing or non-numeric values")
    category = frame[category_column].to_numpy() if category_column else None
    return frame[id_column].to_numpy(), features.to_numpy(dtype=float), names, category
