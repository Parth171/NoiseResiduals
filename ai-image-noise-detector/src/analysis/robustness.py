"""Experiment 3/4: robustness of trained detectors to image degradations.

For each detector (RGB CNN, residual CNN, frequency classifier) and each
degradation (clean, JPEG q90/q65/q40, gaussian blur, resize-down-up), the
degradation is applied to the raw test image BEFORE the standard evaluation
pipeline - mimicking what happens to images in the wild - and metrics are
recomputed. No retraining happens here.

Outputs:
    results/metrics/robustness.csv        one row per detector x degradation x scope
    reports/figures/robustness.png        AUROC vs degradation figure

Usage:
    python -m src.analysis.robustness [--splits-csv data/splits/synthbuster_random.csv]
"""

from __future__ import annotations

import argparse
import io
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.datasets.image_dataset import ImageCsvDataset
from src.datasets.transforms import build_transform
from src.features.frequency import FrequencyFeatureExtractor
from src.features.residuals import ResidualTransform
from src.models.rgb_cnn import build_model
from src.training.evaluate import per_generator_metrics
from src.utils.config import PROJECT_ROOT, load_config
from src.utils.reproducibility import resolve_device


# --- degradations (picklable classes: DataLoader workers use spawn on macOS) ---

class JpegCompress:
    def __init__(self, quality: int) -> None:
        self.quality = quality

    def __call__(self, image: Image.Image) -> Image.Image:
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class PilGaussianBlur:
    def __init__(self, radius: float) -> None:
        self.radius = radius

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.filter(ImageFilter.GaussianBlur(self.radius))


class ResizeDownUp:
    """Downscale by ``factor`` then upscale back (web-thumbnail simulation)."""

    def __init__(self, factor: float = 0.5) -> None:
        self.factor = factor

    def __call__(self, image: Image.Image) -> Image.Image:
        w, h = image.size
        small = image.resize(
            (max(1, int(w * self.factor)), max(1, int(h * self.factor))),
            Image.BILINEAR,
        )
        return small.resize((w, h), Image.BILINEAR)


DEGRADATIONS: dict[str, object | None] = {
    "clean": None,
    "jpeg_q90": JpegCompress(90),
    "jpeg_q65": JpegCompress(65),
    "jpeg_q40": JpegCompress(40),
    "gaussian_blur": PilGaussianBlur(1.0),
    "resize_down_up": ResizeDownUp(0.5),
}


def build_eval_transform(base, degradation) -> transforms.Compose:
    """Prepend a degradation to an existing eval transform pipeline."""
    ops = list(base.transforms)
    if degradation is not None:
        ops = [degradation] + ops
    return transforms.Compose(ops)


def load_cnn(experiment: str, config_name: str, device: torch.device):
    cfg = load_config(f"configs/{config_name}.yaml")
    model = build_model(
        arch=cfg["model"]["arch"],
        pretrained=False,
        in_channels=cfg["model"].get("in_channels", 3),
    )
    checkpoint = torch.load(
        PROJECT_ROOT / "results/checkpoints" / experiment / "best.pt",
        map_location=device, weights_only=True,
    )
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), cfg


@torch.no_grad()
def score_cnn(model, loader, device) -> pd.DataFrame:
    scores = []
    for images, _ in tqdm(loader, leave=False):
        logits = model(images.to(device)).squeeze(1)
        scores.extend(torch.sigmoid(logits).cpu().tolist())
    df = loader.dataset.df.copy()
    df["score"] = scores
    return df


def score_frequency(bundle, extractor, loader) -> pd.DataFrame:
    names = bundle["feature_names"]
    frames = []
    for images, _ in tqdm(loader, leave=False):
        feats = pd.DataFrame(extractor(images).numpy(), columns=names)
        frames.append(bundle["classifier"].predict_proba(bundle["scaler"].transform(feats))[:, 1])
    import numpy as np
    df = loader.dataset.df.copy()
    df["score"] = np.concatenate(frames)
    return df


def make_loader(csv_path, transform, cfg) -> DataLoader:
    dataset = ImageCsvDataset(csv_path, "test", transform)
    return DataLoader(
        dataset,
        batch_size=cfg["data"]["batch_size"],
        num_workers=cfg["data"].get("num_workers", 0),
        shuffle=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--splits-csv", default="data/splits/synthbuster_random.csv")
    args = parser.parse_args()
    csv_path = PROJECT_ROOT / args.splits_csv

    device = resolve_device()
    base_cfg = load_config("configs/baseline_rgb.yaml")
    image_size = base_cfg["data"]["image_size"]

    rgb_model, rgb_cfg = load_cnn("baseline_rgb", "baseline_rgb", device)
    res_model, res_cfg = load_cnn("residual_cnn", "residual_cnn", device)
    with open(PROJECT_ROOT / "results/checkpoints/frequency/model.pkl", "rb") as f:
        freq_bundle = pickle.load(f)
    freq_extractor = FrequencyFeatureExtractor()

    residual_transform = ResidualTransform(
        method=res_cfg["input"]["residual_method"],
        kernel_size=res_cfg["input"]["residual_kernel_size"],
        gaussian_sigma=res_cfg["input"]["gaussian_sigma"],
    )
    base_rgb = build_transform(image_size=image_size, train=False, normalize=True)
    base_residual = build_transform(
        image_size=image_size, train=False, normalize=False, residual=residual_transform
    )
    base_plain = build_transform(image_size=image_size, train=False, normalize=False)

    rows = []
    for name, degradation in DEGRADATIONS.items():
        print(f"=== degradation: {name}")

        loader = make_loader(csv_path, build_eval_transform(base_rgb, degradation), base_cfg)
        pred = score_cnn(rgb_model, loader, device)
        rows.append(per_generator_metrics(pred).assign(detector="rgb_cnn", degradation=name))

        loader = make_loader(csv_path, build_eval_transform(base_residual, degradation), base_cfg)
        pred = score_cnn(res_model, loader, device)
        rows.append(per_generator_metrics(pred).assign(detector="residual_cnn", degradation=name))

        loader = make_loader(csv_path, build_eval_transform(base_plain, degradation), base_cfg)
        pred = score_frequency(freq_bundle, freq_extractor, loader)
        rows.append(per_generator_metrics(pred).assign(detector="frequency", degradation=name))

    results = pd.concat(rows, ignore_index=True)
    out_csv = PROJECT_ROOT / "results/metrics/robustness.csv"
    results.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    pooled = results[results["scope"] == "pooled_all_generators"]
    print("\nPooled AUROC by degradation:")
    print(
        pooled.pivot(index="degradation", columns="detector", values="auroc")
        .reindex(DEGRADATIONS.keys())
        .round(4)
        .to_string()
    )

    # --- figure: JPEG decay curve + other degradations ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    jpeg_order = ["clean", "jpeg_q90", "jpeg_q65", "jpeg_q40"]
    xlabels = ["original", "q90", "q65", "q40"]
    for detector, group in pooled.groupby("detector"):
        series = group.set_index("degradation")["auroc"]
        ax1.plot(xlabels, [series[d] for d in jpeg_order], marker="o", label=detector)
    ax1.set_ylim(0.4, 1.02)
    ax1.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax1.set_xlabel("JPEG quality")
    ax1.set_ylabel("pooled test AUROC")
    ax1.set_title("JPEG compression robustness")
    ax1.legend()

    others = ["clean", "gaussian_blur", "resize_down_up"]
    width = 0.25
    for i, (detector, group) in enumerate(pooled.groupby("detector")):
        series = group.set_index("degradation")["auroc"]
        ax2.bar(
            [x + (i - 1) * width for x in range(len(others))],
            [series[d] for d in others],
            width=width, label=detector,
        )
    ax2.set_xticks(range(len(others)), others)
    ax2.set_ylim(0.4, 1.02)
    ax2.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax2.set_title("Blur and resize robustness")
    ax2.legend()

    fig.tight_layout()
    out_fig = PROJECT_ROOT / "reports/figures/robustness.png"
    fig.savefig(out_fig, dpi=150, bbox_inches="tight")
    print(f"Saved {out_fig}")


if __name__ == "__main__":
    main()
