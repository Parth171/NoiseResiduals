"""Generate a tiny synthetic dataset to smoke-test the data pipeline.

This does NOT pretend to be a realistic benchmark; it only mimics the raw-data
directory convention (real/ and fake/<generator>/) with images whose noise
statistics differ slightly between classes, so the full pipeline (splits,
loading, training loops) can be validated quickly before real data arrives.

"Real" images: smooth random gradients + strong per-pixel Gaussian noise
(camera-sensor-like). "Fake" images: the same construction but blurred, i.e.
suspiciously clean high frequencies - loosely mimicking generator output.

Usage:
    python scripts/make_dummy_data.py [--n-per-class 60] [--size 256]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import PROJECT_ROOT

DUMMY_GENERATORS = ("dummy_gen_a", "dummy_gen_b", "dummy_gen_c")


def _base_image(rng: np.random.Generator, size: int) -> np.ndarray:
    """Random low-frequency color gradient in [0, 255]."""
    yy, xx = np.mgrid[0:size, 0:size] / size
    channels = []
    for _ in range(3):
        a, b, c = rng.uniform(-1, 1, size=3)
        channels.append((a * xx + b * yy + c * xx * yy))
    img = np.stack(channels, axis=-1)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img * 255.0


def make_real(rng: np.random.Generator, size: int) -> Image.Image:
    img = _base_image(rng, size) + rng.normal(0, 12.0, size=(size, size, 3))
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def make_fake(rng: np.random.Generator, size: int, blur_radius: float) -> Image.Image:
    img = _base_image(rng, size) + rng.normal(0, 12.0, size=(size, size, 3))
    pil = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
    return pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-class", type=int, default=60,
                        help="images per class (fakes are split across generators)")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "data/raw/dummy")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    real_dir = args.output / "real"
    real_dir.mkdir(parents=True, exist_ok=True)
    for i in range(args.n_per_class):
        make_real(rng, args.size).save(real_dir / f"real_{i:04d}.png")

    n_per_gen = args.n_per_class // len(DUMMY_GENERATORS)
    # each dummy generator blurs differently, giving generator-specific artifacts
    for g, (gen, blur) in enumerate(zip(DUMMY_GENERATORS, (0.8, 1.2, 1.6))):
        gen_dir = args.output / "fake" / gen
        gen_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_per_gen):
            make_fake(rng, args.size, blur).save(gen_dir / f"{gen}_{i:04d}.png")

    n_fake = n_per_gen * len(DUMMY_GENERATORS)
    print(f"Wrote {args.n_per_class} real and {n_fake} fake images to {args.output}")


if __name__ == "__main__":
    main()
