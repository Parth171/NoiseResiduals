"""Run all three trained detectors on every image in this folder.

Drop image files (jpg/png/tiff/webp/...) into individual_tests/ and run:

    ./.venv/bin/python individual_tests/predict.py

Each detector prints a score from 0 (real) to 1 (AI-generated):
- rgb_cnn:       ResNet18 on the RGB image (content + all cues)
- residual_cnn:  ResNet18 on the noise residual (x - blur(x), noise only)
- frequency:     logistic regression on FFT/DCT spectrum features

Note: the models were trained on pristine PNGs (Synthbuster) vs uncompressed
camera crops (RAISE-1k). Scores on compressed/edited internet images are
indicative, not authoritative.
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.transforms import build_transform
from src.features.frequency import FrequencyFeatureExtractor
from src.features.residuals import ResidualTransform
from src.models.rgb_cnn import build_model
from src.utils.config import load_config

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def load_cnn(experiment: str, config_name: str):
    cfg = load_config(f"configs/{config_name}.yaml")
    model = build_model(
        arch=cfg["model"]["arch"],
        pretrained=False,
        in_channels=cfg["model"].get("in_channels", 3),
    )
    ckpt_path = PROJECT_ROOT / "results/checkpoints" / experiment / "best.pt"
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, cfg


def main() -> None:
    folder = Path(__file__).resolve().parent
    images = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        print(f"No images found in {folder}. Drop some image files there and rerun.")
        return

    print("Loading detectors ...")
    rgb_model, _ = load_cnn("baseline_rgb", "baseline_rgb")
    res_model, res_cfg = load_cnn("residual_cnn", "residual_cnn")
    with open(PROJECT_ROOT / "results/checkpoints/frequency/model.pkl", "rb") as f:
        freq_bundle = pickle.load(f)

    to_tensor = build_transform(image_size=224, train=False, normalize=False)
    to_tensor_imagenet = build_transform(image_size=224, train=False, normalize=True)
    residual_transform = ResidualTransform(
        method=res_cfg["input"]["residual_method"],
        kernel_size=res_cfg["input"]["residual_kernel_size"],
        gaussian_sigma=res_cfg["input"]["gaussian_sigma"],
    )
    freq_extractor = FrequencyFeatureExtractor()

    print(f"\n{'image':<40} {'rgb_cnn':>8} {'residual':>9} {'frequency':>10}   verdicts (0=real, 1=AI)")
    print("-" * 100)
    for path in images:
        image = Image.open(path).convert("RGB")
        x = to_tensor(image)

        with torch.no_grad():
            rgb_score = torch.sigmoid(
                rgb_model(to_tensor_imagenet(image).unsqueeze(0))
            ).item()
            res_score = torch.sigmoid(
                res_model(residual_transform(x).unsqueeze(0))
            ).item()

        feats_df = pd.DataFrame(
            freq_extractor(x).numpy(), columns=freq_bundle["feature_names"]
        )
        feats = freq_bundle["scaler"].transform(feats_df)
        freq_score = float(freq_bundle["classifier"].predict_proba(feats)[0, 1])

        verdicts = " / ".join(
            "AI" if s >= 0.5 else "real" for s in (rgb_score, res_score, freq_score)
        )
        print(f"{path.name:<40} {rgb_score:>8.3f} {res_score:>9.3f} {freq_score:>10.3f}   {verdicts}")


if __name__ == "__main__":
    main()
