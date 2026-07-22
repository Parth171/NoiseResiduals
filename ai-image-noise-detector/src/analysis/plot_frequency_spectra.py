"""Interpretability figure for the frequency detector.

Left panel: mean radial log-power spectrum of real vs fake test images (with
per-generator curves), showing where in frequency space the classes separate.
Right panel: the strongest logistic-regression weights, i.e. which named
features the classifier actually relies on.

Usage:
    python -m src.analysis.plot_frequency_spectra
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.utils.config import PROJECT_ROOT, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/frequency.yaml")
    parser.add_argument("--top-weights", type=int, default=15)
    args = parser.parse_args()

    cfg = load_config(args.config)
    experiment = cfg["experiment_name"]
    csv_stem = Path(cfg["paths"]["splits_csv"]).stem
    features_csv = PROJECT_ROOT / "data/processed" / f"frequency_features_{csv_stem}.csv"
    weights_csv = cfg["paths"]["metrics_dir"] / experiment / "feature_weights.csv"

    df = pd.read_csv(features_csv)
    test = df[df["split"] == "test"]
    radial_cols = [c for c in df.columns if c.startswith("radial_")]
    x = range(len(radial_cols))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    for generator, group in test[test["label"] == 1].groupby("generator"):
        ax1.plot(x, group[radial_cols].mean(), lw=0.8, alpha=0.45, label=generator)
    ax1.plot(
        x, test.loc[test["label"] == 0, radial_cols].mean(),
        color="black", lw=2.5, label="REAL",
    )
    ax1.set_xlabel("radial frequency bin (0 = DC, 63 = Nyquist corner)")
    ax1.set_ylabel("mean log power")
    ax1.set_title("Radial power spectrum: real vs generators (test set)")
    ax1.legend(fontsize=7, ncol=2)

    weights = pd.read_csv(weights_csv).head(args.top_weights).iloc[::-1]
    colors = ["tab:red" if w > 0 else "tab:blue" for w in weights["weight"]]
    ax2.barh(weights["feature"], weights["weight"], color=colors)
    ax2.axvline(0, color="black", lw=0.8)
    ax2.set_title(
        f"Top {args.top_weights} logistic-regression weights\n"
        "(red pushes toward FAKE, blue toward REAL)"
    )
    ax2.set_xlabel("weight (standardized features)")

    fig.tight_layout()
    out = cfg["paths"]["figures_dir"] / "frequency_spectra.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
