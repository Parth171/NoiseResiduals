"""Visualize noise residuals for example real and fake images.

Produces a grid (rows = example images, columns = original + each residual
method) saved to reports/figures/residual_examples.png. Residuals are
amplified and recentered for display only; the model sees the raw values.

Usage:
    python -m src.analysis.visualize_residuals \
        [--splits-csv data/splits/synthbuster_random.csv] [--n-per-class 2]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.datasets.transforms import build_transform
from src.features.residuals import ResidualTransform
from src.utils.config import PROJECT_ROOT, load_config

METHODS = ("gaussian", "laplacian", "sobel", "median", "srm")


def _to_display(t: torch.Tensor, amplify: float = 5.0) -> torch.Tensor:
    """Map a residual tensor to a viewable [0, 1] image (display only)."""
    return (t * amplify + 0.5).clamp(0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-csv", default="data/splits/synthbuster_random.csv")
    parser.add_argument("--n-per-class", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config("configs/residual_cnn.yaml")
    df = pd.read_csv(PROJECT_ROOT / args.splits_csv)
    test = df[df["split"] == "test"]
    real = test[test["label"] == 0].sample(args.n_per_class, random_state=args.seed)
    fake = test[test["label"] == 1].sample(args.n_per_class, random_state=args.seed)
    rows = pd.concat([real, fake])

    to_tensor = build_transform(
        image_size=cfg["data"]["image_size"], train=False, normalize=False
    )
    transforms_by_method = {m: ResidualTransform(m) for m in METHODS}

    n_rows, n_cols = len(rows), 1 + len(METHODS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.4 * n_cols, 2.6 * n_rows))
    for r, (_, row) in enumerate(rows.iterrows()):
        image = Image.open(PROJECT_ROOT / row["path"]).convert("RGB")
        x = to_tensor(image)
        title = "REAL" if row["label"] == 0 else f"FAKE ({row['generator']})"

        axes[r, 0].imshow(x.permute(1, 2, 0))
        axes[r, 0].set_ylabel(title, fontsize=9)
        if r == 0:
            axes[r, 0].set_title("original", fontsize=10)

        for c, method in enumerate(METHODS, start=1):
            res = transforms_by_method[method](x)
            if method == "srm":  # channels are filter outputs, show the first
                disp = _to_display(res[:1]).squeeze(0)
                axes[r, c].imshow(disp, cmap="gray")
            else:
                axes[r, c].imshow(_to_display(res).permute(1, 2, 0))
            if r == 0:
                axes[r, c].set_title(method, fontsize=10)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Noise residuals (amplified 5x for display)", fontsize=12)
    fig.tight_layout()

    out = cfg["paths"]["figures_dir"] / "residual_examples.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
