"""Tests for residual extraction: shapes, behavior on flat vs noisy inputs,
and integration with the transform pipeline."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from PIL import Image

from src.datasets.transforms import build_transform
from src.features.residuals import (
    ResidualTransform,
    compute_gaussian_residual,
    compute_laplacian_residual,
    compute_median_residual,
    compute_sobel_residual,
    compute_srm_residual,
)

ALL_METHODS = ("gaussian", "laplacian", "sobel", "median", "srm")


def test_shapes_and_flat_image():
    x = torch.rand(3, 64, 64)
    flat = torch.full((3, 64, 64), 0.5)

    for fn in (
        compute_gaussian_residual,
        compute_laplacian_residual,
        compute_sobel_residual,
        compute_median_residual,
        compute_srm_residual,
    ):
        out = fn(x)
        assert out.shape == (3, 64, 64), f"{fn.__name__}: bad shape {out.shape}"
        assert out.dtype == torch.float32
        # a constant image has no high-frequency content -> residual ~ 0
        assert fn(flat).abs().max() < 1e-5, f"{fn.__name__}: flat image residual != 0"


def test_noise_sensitivity():
    """A noisy image must produce a larger residual than its smooth original."""
    torch.manual_seed(0)
    yy = torch.linspace(0, 1, 64).view(1, -1, 1).expand(3, 64, 64)
    smooth = yy.contiguous()
    noisy = (smooth + 0.05 * torch.randn_like(smooth)).clamp(0, 1)

    for method in ALL_METHODS:
        t = ResidualTransform(method=method)
        energy_smooth = t(smooth).abs().mean().item()
        energy_noisy = t(noisy).abs().mean().item()
        # sobel also responds to the smooth ramp itself (first derivative),
        # so the margin is modest rather than an order of magnitude
        assert energy_noisy > 1.5 * energy_smooth, (
            f"{method}: noisy residual energy {energy_noisy:.5f} not clearly above "
            f"smooth {energy_smooth:.5f}"
        )


def test_transform_pipeline_integration():
    rng = np.random.default_rng(0)
    pil = Image.fromarray(rng.integers(0, 256, size=(300, 400, 3), dtype=np.uint8))
    transform = build_transform(
        image_size=224, train=False, normalize=False, residual=ResidualTransform("gaussian")
    )
    out = transform(pil)
    assert out.shape == (3, 224, 224)
    # residuals are roughly zero-centered
    assert abs(out.mean().item()) < 0.05


if __name__ == "__main__":
    test_shapes_and_flat_image()
    test_noise_sensitivity()
    test_transform_pipeline_integration()
    print("All residual tests passed.")
