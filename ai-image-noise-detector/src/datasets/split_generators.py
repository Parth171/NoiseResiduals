"""Build metadata CSVs and train/val/test splits.

Expected raw-data directory convention (one folder per dataset):

    <data_root>/
      real/                     real photos (label 0, generator "real")
        img001.jpg ...
      fake/                     AI-generated images (label 1)
        stable_diffusion_14/
          img001.png ...
        midjourney_v5/
          ...

Metadata CSV schema (one row per image, committed to data/splits/):

    path,label,generator,split,dataset,source_id

``path`` is stored relative to the project root so CSVs are portable.
Label convention: 0 = real, 1 = AI-generated.

``source_id`` is the filename stem. For paired datasets such as Synthbuster,
fakes are named after the RAISE-1k photo whose derived prompt generated them,
so a real photo and all fakes derived from it share one source_id. Splits can
be *grouped* by source_id so that related images never straddle splits
(content leakage would otherwise inflate test performance).

Rows excluded by a split policy are marked split="unused" (kept in the CSV
for provenance; dataloaders only ever select train/val/test).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import PROJECT_ROOT

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

Fracs = tuple[float, float, float]


def _list_images(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_metadata(data_root: str | Path, dataset_name: str) -> pd.DataFrame:
    """Scan ``data_root`` (real/ and fake/<generator>/) into a metadata frame."""
    data_root = Path(data_root)
    rows = []

    real_dir = data_root / "real"
    if real_dir.is_dir():
        for path in _list_images(real_dir):
            rows.append((path, 0, "real", path.stem))

    fake_dir = data_root / "fake"
    if fake_dir.is_dir():
        for generator_dir in sorted(p for p in fake_dir.iterdir() if p.is_dir()):
            for path in _list_images(generator_dir):
                rows.append((path, 1, generator_dir.name, path.stem))

    if not rows:
        raise FileNotFoundError(
            f"No images found under {data_root} (expected real/ and fake/<generator>/)"
        )

    df = pd.DataFrame(rows, columns=["path", "label", "generator", "source_id"])
    df["path"] = df["path"].map(
        lambda p: str(p.relative_to(PROJECT_ROOT)) if p.is_relative_to(PROJECT_ROOT) else str(p)
    )
    df["dataset"] = dataset_name
    return df


def _split_indices(n: int, fracs: Fracs, rng: np.random.Generator):
    """Shuffle ``n`` indices and cut them into train/val/test groups."""
    order = rng.permutation(n)
    n_train = int(round(fracs[0] * n))
    n_val = int(round(fracs[1] * n))
    return order[:n_train], order[n_train:n_train + n_val], order[n_train + n_val:]


def _partition_sources(
    sources: pd.Series, fracs: Fracs, seed: int
) -> dict[str, str]:
    """Randomly assign each unique source_id to train/val/test."""
    unique = np.sort(sources.unique())
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = _split_indices(len(unique), fracs, rng)
    mapping: dict[str, str] = {}
    for idx_group, split in ((train_idx, "train"), (val_idx, "val"), (test_idx, "test")):
        for i in idx_group:
            mapping[unique[i]] = split
    return mapping


def assign_random_split(
    df: pd.DataFrame,
    fracs: Fracs = (0.70, 0.15, 0.15),
    seed: int = 42,
    group_by_source: bool = False,
) -> pd.DataFrame:
    """Random train/val/test split.

    - ``group_by_source=False``: stratified per generator, so every generator
      (and the real class) appears in each split with the same proportions.
    - ``group_by_source=True``: unique source_ids are partitioned instead, and
      every image inherits its source's split. For paired datasets this both
      prevents content leakage across splits and stratifies generators exactly
      (each source contributes one image per generator).
    """
    assert abs(sum(fracs) - 1.0) < 1e-6, "split fractions must sum to 1"
    df = df.copy()
    df["split"] = ""

    if group_by_source:
        mapping = _partition_sources(df["source_id"], fracs, seed)
        df["split"] = df["source_id"].map(mapping)
        return df

    rng = np.random.default_rng(seed)
    for _, group in df.groupby("generator"):
        train_idx, val_idx, test_idx = _split_indices(len(group), fracs, rng)
        df.loc[group.index[train_idx], "split"] = "train"
        df.loc[group.index[val_idx], "split"] = "val"
        df.loc[group.index[test_idx], "split"] = "test"
    return df


def assign_cross_generator_split(
    df: pd.DataFrame,
    held_out_generators: list[str],
    fracs: Fracs = (0.70, 0.15, 0.15),
    seed: int = 42,
    group_by_source: bool = False,
) -> pd.DataFrame:
    """Cross-generator split for the generalization experiment.

    - Real images: split train/val/test as usual (test needs real images too).
    - Fake images from *held-out* generators: only ever appear in test.
    - Fake images from remaining (training) generators: train and val only.

    With ``group_by_source=True`` the source partition additionally rules
    everything: reals follow their source; training-generator fakes are kept
    only when their source is in train/val; held-out fakes are kept only when
    their source is in test. Excluded rows get split="unused". This keeps the
    test set fully disjoint in content from training, at the cost of test
    size (15% of sources).
    """
    df = df.copy()
    df["split"] = ""

    known = set(df.loc[df["label"] == 1, "generator"].unique())
    missing = set(held_out_generators) - known
    if missing:
        raise ValueError(f"Held-out generators not found in metadata: {sorted(missing)}")
    held_out = set(held_out_generators)

    if group_by_source:
        mapping = _partition_sources(df["source_id"], fracs, seed)
        source_split = df["source_id"].map(mapping)
        for i in df.index:
            generator, split = df.at[i, "generator"], source_split[i]
            if generator == "real":
                df.at[i, "split"] = split
            elif generator in held_out:
                df.at[i, "split"] = "test" if split == "test" else "unused"
            else:
                df.at[i, "split"] = split if split in ("train", "val") else "unused"
        return df

    rng = np.random.default_rng(seed)
    for generator, group in df.groupby("generator"):
        if generator == "real":
            train_idx, val_idx, test_idx = _split_indices(len(group), fracs, rng)
            df.loc[group.index[train_idx], "split"] = "train"
            df.loc[group.index[val_idx], "split"] = "val"
            df.loc[group.index[test_idx], "split"] = "test"
        elif generator in held_out:
            df.loc[group.index, "split"] = "test"
        else:
            train_frac = fracs[0] + fracs[2]  # fold unused test share into train
            train_idx, val_idx, _ = _split_indices(
                len(group), (train_frac, fracs[1], 0.0), rng
            )
            df.loc[group.index[train_idx], "split"] = "train"
            df.loc[group.index[val_idx], "split"] = "val"
    return df


def save_split_csv(df: pd.DataFrame, output_path: str | Path) -> Path:
    """Write a split CSV with the canonical column order."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = df[["path", "label", "generator", "split", "dataset", "source_id"]]
    df.to_csv(output_path, index=False)
    return output_path


def summarize_split(df: pd.DataFrame) -> pd.DataFrame:
    """Image counts per (split, label) with per-generator breakdown."""
    return (
        df.groupby(["split", "label", "generator"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["split", "label", "generator"])
        .reset_index(drop=True)
    )
