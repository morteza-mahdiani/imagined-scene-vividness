"""Train a probe on a frozen encoder."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from vividness.encoders import Encoder, build_encoder, list_models
from vividness.probe import MLPRegressor
from vividness.stats import correlation


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_ids(
    ids: np.ndarray,
    splits: Path | None,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train, validation, and test identifiers.

    A split file supplies ``train_idx``, ``val_idx``, and ``test_idx``.
    Otherwise the rows are permuted with ``seed``: the first test fraction,
    then the validation fraction, and the remainder for training.
    """
    if splits is not None:
        stored = np.load(splits)
        return stored["train_idx"], stored["val_idx"], stored["test_idx"]
    generator = np.random.default_rng(seed)
    order = ids[generator.permutation(len(ids))]
    n_test = int(test_frac * len(ids))
    n_val = int(val_frac * len(ids))
    return order[n_test + n_val :], order[n_test : n_test + n_val], order[:n_test]


class ImageTable(Dataset):
    def __init__(self, frame: pd.DataFrame, preprocess) -> None:
        self.paths = frame["path"].tolist()
        self.targets = frame["scaled"].to_numpy(dtype=np.float32)
        self.ids = frame["image_id"].to_numpy()
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        image = Image.open(self.paths[index]).convert("RGB")
        return self.preprocess(image), float(self.targets[index]), self.ids[index]


def _collate(batch):
    images, targets, ids = zip(*batch)
    return torch.stack(list(images)), torch.tensor(targets, dtype=torch.float32), list(ids)


def _loaders(frame: pd.DataFrame, groups: dict[str, np.ndarray], encoder: Encoder, batch_size: int, workers: int, seed: int):
    loaders = {}
    generator = torch.Generator()
    generator.manual_seed(seed)
    for name, chosen in groups.items():
        subset = frame[frame["image_id"].isin(set(chosen.tolist()))]
        missing = set(chosen.tolist()) - set(subset["image_id"].tolist())
        if missing:
            raise SystemExit(f"{len(missing)} split ids are missing from the table")
        loaders[name] = DataLoader(
            ImageTable(subset, encoder.preprocess),
            batch_size=batch_size,
            shuffle=name == "train",
            num_workers=workers,
            collate_fn=_collate,
            generator=generator if name == "train" else None,
        )
    return loaders


def _epoch(encoder: nn.Module, probe: nn.Module, loader: DataLoader, device: str, optimizer=None) -> tuple[float, np.ndarray, np.ndarray, list]:
    training = optimizer is not None
    probe.train(training)
    total = 0.0
    count = 0
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    identifiers: list = []
    for images, target, ids in loader:
        images = images.to(device)
        target = target.to(device)
        with torch.set_grad_enabled(training):
            with torch.no_grad():
                features = encoder(images).float()
            predicted = probe(features)
            loss = torch.nn.functional.mse_loss(predicted, target)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total += float(loss.item()) * len(target)
        count += len(target)
        predictions.append(predicted.detach().cpu().numpy())
        targets.append(target.detach().cpu().numpy())
        identifiers.extend(ids)
    return total / max(count, 1), np.concatenate(predictions), np.concatenate(targets), identifiers


def train_probe(args) -> None:
    if args.list_models:
        print(list_models())
        return
    if args.table is None or args.model is None or args.out is None:
        raise SystemExit("train requires --table, --model, and --out")
    set_seed(args.seed)
    table = pd.read_csv(args.table)
    for column in (args.id_column, args.target_column, args.path_column):
        if column not in table.columns:
            raise SystemExit(f"{args.table} is missing column: {column}")
    table = table.rename(
        columns={args.id_column: "image_id", args.target_column: "target", args.path_column: "path"}
    )
    if table["image_id"].duplicated().any():
        raise SystemExit("duplicate image ids")
    if args.image_root:
        root = Path(args.image_root)
        table["path"] = table["path"].map(lambda name: str(root / name))
    target = table["target"].to_numpy(dtype=float)
    center = float(target.mean())
    scale = float(target.std() + 1e-8)
    table["scaled"] = ((target - center) / scale).astype(np.float32)
    train_ids, val_ids, test_ids = split_ids(
        table["image_id"].to_numpy(), args.splits, args.val_frac, args.test_frac, args.seed
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = build_encoder(args.model, device)
    probe = MLPRegressor(encoder.embed_dim, args.hidden, args.dropout).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=args.lr, weight_decay=1e-2)
    loaders = _loaders(
        table,
        {"train": train_ids, "val": val_ids, "test": test_ids},
        encoder,
        args.batch_size,
        args.workers,
        args.seed,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, _, _, _ = _epoch(encoder.backbone, probe, loaders["train"], device, optimizer)
        val_loss, val_pred, val_target, _ = _epoch(encoder.backbone, probe, loaders["val"], device)
        original_target = val_target * scale + center
        original_pred = val_pred * scale + center
        score = correlation(original_pred, original_target)
        print(f"[{epoch:03d}] train {train_loss:.5f}  val {val_loss:.5f}  r {score:.3f}", flush=True)
        if val_loss < best:
            best = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in probe.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if args.patience and stale >= args.patience:
                break
    if best_state is not None:
        probe.load_state_dict(best_state)
    _, test_pred, test_target, test_kept = _epoch(encoder.backbone, probe, loaders["test"], device)
    original_target = test_target * scale + center
    original_pred = test_pred * scale + center
    metrics = {
        "model": args.model,
        "n_test": int(len(test_target)),
        "best_epoch": int(best_epoch),
        "pearson_r": correlation(original_pred, original_target),
        "spearman_r": correlation(original_pred, original_target, method="spearman"),
    }
    (out / "test_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    pd.DataFrame(
        {"image_id": test_kept, "target": original_target, args.model: original_pred}
    ).to_csv(out / "test_predictions.csv", index=False)
    torch.save(
        {
            "model": args.model,
            "hidden": int(args.hidden),
            "embed_dim": int(encoder.embed_dim),
            "target_mean": center,
            "target_std": scale,
            "head": probe.state_dict(),
        },
        out / "head_best.pt",
    )
    print(out / "head_best.pt")
    print(json.dumps(metrics, indent=2))
