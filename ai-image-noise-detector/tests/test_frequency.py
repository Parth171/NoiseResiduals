"""Tests for frequency feature extraction: shapes, names, and the physical
ordering that sharp/noisy images carry more high-frequency energy than
blurred ones."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torchvision.transforms.functional import gaussian_blur

from src.features.frequency import FrequencyFeatureExtractor


def test_shapes_and_names():
    extractor = FrequencyFeatureExtractor(radial_bins=64)
    names = extractor.feature_names()
    assert len(names) == 64 + 1 + 64
    assert names[0] == "radial_00" and names[64] == "hf_energy_ratio"
    assert names[65] == "dct_00" and names[-1] == "dct_77"

    batch = torch.rand(4, 3, 224, 224)
    feats = extractor(batch)
    assert feats.shape == (4, len(names))
    assert torch.isfinite(feats).all()

    # single image (C, H, W) also works
    single = extractor(torch.rand(3, 224, 224))
    assert single.shape == (1, len(names))


def test_blur_reduces_high_frequency_energy():
    torch.manual_seed(0)
    extractor = FrequencyFeatureExtractor()
    idx = extractor.feature_names().index("hf_energy_ratio")

    noise = torch.rand(2, 3, 224, 224)
    blurred = gaussian_blur(noise, [9, 9], [3.0, 3.0])

    hf_noise = extractor(noise)[:, idx]
    hf_blurred = extractor(blurred)[:, idx]
    assert (hf_noise > 5 * hf_blurred).all(), (hf_noise, hf_blurred)


def test_deterministic():
    extractor = FrequencyFeatureExtractor()
    x = torch.rand(1, 3, 224, 224)
    assert torch.equal(extractor(x), extractor(x))


if __name__ == "__main__":
    test_shapes_and_names()
    test_blur_reduces_high_frequency_energy()
    test_deterministic()
    print("All frequency tests passed.")
