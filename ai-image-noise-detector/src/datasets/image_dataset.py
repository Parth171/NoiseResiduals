"""Metadata-CSV-driven PyTorch dataset and dataloader factory.

Every model in this project consumes data through ``ImageCsvDataset``: a split
CSV (see ``split_generators.py``) fully determines which images belong to
which split, making experiments reproducible from the committed CSVs alone.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from src.datasets.transforms import build_transform
from src.features.residuals import ResidualTransform
from src.utils.config import PROJECT_ROOT

VALID_SPLITS = ("train", "val", "test")


class ImageCsvDataset(Dataset):
    """Serves (image_tensor, label) pairs for one split of a metadata CSV.

    The per-row metadata (path, generator, ...) stays available in ``self.df``
    in the same order as dataset indices, which the evaluation code uses for
    per-generator metrics.
    """

    def __init__(
        self,
        csv_path: str | Path,
        split: str,
        transform: Callable | None = None,
    ) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")
        csv_path = Path(csv_path)
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        if len(self.df) == 0:
            raise ValueError(f"No rows with split={split!r} in {csv_path}")
        self.split = split
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        path = Path(row["path"])
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(float(row["label"]), dtype=torch.float32)
        return image, label


def build_dataloader(
    cfg: dict, split: str, normalize: bool = True, balance: bool = False
) -> DataLoader:
    """Create a DataLoader for one split, driven entirely by the config.

    ``balance=True`` (training only) draws samples with per-class weights so
    each batch is ~50% real / 50% fake regardless of the dataset's class
    ratio; the minority class repeats more often within an epoch.

    If the config's ``input.type`` is "residual", the residual extraction step
    is appended to the transform (and ImageNet normalization is disabled).
    """
    data_cfg = cfg["data"]
    input_cfg = cfg.get("input", {})
    train = split == "train"

    residual = None
    if input_cfg.get("type") == "residual":
        residual = ResidualTransform(
            method=input_cfg.get("residual_method", "gaussian"),
            kernel_size=input_cfg.get("residual_kernel_size", 5),
            gaussian_sigma=input_cfg.get("gaussian_sigma", 1.0),
        )
        normalize = False

    transform = build_transform(
        image_size=data_cfg["image_size"],
        train=train,
        crop=data_cfg.get("train_crop", "center"),
        normalize=normalize,
        residual=residual,
    )
    dataset = ImageCsvDataset(cfg["paths"]["splits_csv"], split, transform)

    sampler = None
    if balance and train:
        class_counts = dataset.df["label"].value_counts()
        weights = dataset.df["label"].map(lambda y: 1.0 / class_counts[y]).to_numpy().copy()
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(dataset),
            replacement=True,
        )

    return DataLoader(
        dataset,
        batch_size=data_cfg["batch_size"],
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
