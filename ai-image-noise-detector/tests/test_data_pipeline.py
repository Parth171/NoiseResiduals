"""End-to-end smoke test of the data pipeline on a temporary dummy dataset:
metadata scan -> random and cross-generator splits -> dataset -> batch."""

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from PIL import Image

from src.datasets.image_dataset import ImageCsvDataset
from src.datasets.split_generators import (
    assign_cross_generator_split,
    assign_random_split,
    build_metadata,
    save_split_csv,
)
from src.datasets.transforms import build_transform


def _make_tiny_dataset(root: Path, n_real: int = 12, n_per_gen: int = 6) -> None:
    """Paired naming like Synthbuster: fake s<i>.png derives from real s<i>.png."""
    rng = np.random.default_rng(0)

    def save(path: Path) -> None:
        arr = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(path)

    for i in range(n_real):
        save(root / "real" / f"s{i:03d}.png")
    for gen in ("gen_a", "gen_b"):
        for i in range(n_per_gen):
            save(root / "fake" / gen / f"s{i:03d}.png")


def test_pipeline() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_tiny_dataset(root)

        df = build_metadata(root, "TinyTest")
        assert len(df) == 12 + 2 * 6
        assert set(df["label"].unique()) == {0, 1}
        assert set(df["generator"].unique()) == {"real", "gen_a", "gen_b"}

        # random split: every generator present in every split
        rand = assign_random_split(df, (0.5, 0.25, 0.25), seed=0)
        counts = rand.groupby(["generator", "split"]).size().unstack(fill_value=0)
        assert (counts > 0).all().all(), f"empty split cell:\n{counts}"

        # deterministic given the same seed
        rand2 = assign_random_split(df, (0.5, 0.25, 0.25), seed=0)
        pd.testing.assert_frame_equal(rand, rand2)

        # cross-generator split: held-out fakes only in test, others never in test
        cross = assign_cross_generator_split(df, ["gen_b"], (0.5, 0.25, 0.25), seed=0)
        gen_b = cross[cross["generator"] == "gen_b"]
        assert (gen_b["split"] == "test").all()
        gen_a = cross[cross["generator"] == "gen_a"]
        assert set(gen_a["split"]) <= {"train", "val"}
        real_test = cross[(cross["generator"] == "real") & (cross["split"] == "test")]
        assert len(real_test) > 0, "test split must contain real images"

        # grouped random split: all images of a source share one split
        grouped = assign_random_split(df, (0.5, 0.25, 0.25), seed=0, group_by_source=True)
        per_source = grouped.groupby("source_id")["split"].nunique()
        assert (per_source == 1).all(), "a source_id straddles splits"
        assert set(grouped["split"].unique()) == {"train", "val", "test"}

        # grouped cross-generator split: no content overlap between test and
        # train/val, held-out fakes only in test, excluded rows marked unused
        gcross = assign_cross_generator_split(
            df, ["gen_b"], (0.5, 0.25, 0.25), seed=0, group_by_source=True
        )
        used = gcross[gcross["split"] != "unused"]
        test_sources = set(used.loc[used["split"] == "test", "source_id"])
        trainval_sources = set(used.loc[used["split"] != "test", "source_id"])
        assert not (test_sources & trainval_sources), "content leaks into test"
        assert set(used.loc[used["generator"] == "gen_b", "split"]) == {"test"}
        assert set(used.loc[used["generator"] == "gen_a", "split"]) <= {"train", "val"}

        # CSV round-trip + dataset + batch shapes
        csv_path = root / "split.csv"
        save_split_csv(rand, csv_path)
        transform = build_transform(image_size=224, train=True, crop="random")
        ds = ImageCsvDataset(csv_path, "train", transform)
        image, label = ds[0]
        assert image.shape == (3, 224, 224)
        assert label.item() in (0.0, 1.0)

        from torch.utils.data import DataLoader
        batch_images, batch_labels = next(iter(DataLoader(ds, batch_size=4)))
        assert batch_images.shape == (4, 3, 224, 224)
        assert batch_labels.shape == (4,)

    print("All data pipeline tests passed.")


if __name__ == "__main__":
    test_pipeline()
