"""Frequency-domain feature extraction for Detector 3.

Each image is summarized as a small named feature vector:

- radial_power_spectrum: the 2D FFT log-power spectrum, azimuthally averaged
  into ``radial_bins`` rings from DC (bin 0) to the Nyquist corner (last bin).
  Natural photos follow a smooth ~1/f falloff; generators often deviate,
  especially at high frequencies.
- high_freq_energy_ratio: fraction of total spectral energy beyond a radius
  threshold - a single number capturing "how much fine detail energy exists".
- dct_stats: mean log-magnitude of each of the 64 coefficients of a blockwise
  8x8 DCT (JPEG's transform), averaged over all blocks.

The extractor is deliberately shallow and fully interpretable: every feature
has a name, so a linear classifier's weights can be read directly.

All computation is batched torch on (B, C, H, W) tensors in [0, 1]; images
are converted to grayscale internally.
"""

from __future__ import annotations

import math

import torch

_EPS = 1e-8


def _dct_matrix(n: int = 8) -> torch.Tensor:
    """Orthonormal DCT-II matrix."""
    k = torch.arange(n).unsqueeze(1).float()
    i = torch.arange(n).unsqueeze(0).float()
    d = torch.cos(math.pi * (i + 0.5) * k / n)
    d[0] *= math.sqrt(1.0 / n)
    d[1:] *= math.sqrt(2.0 / n)
    return d


class FrequencyFeatureExtractor:
    """Config-driven extractor; see configs/frequency.yaml."""

    def __init__(
        self,
        features: tuple[str, ...] = (
            "radial_power_spectrum",
            "high_freq_energy_ratio",
            "dct_stats",
        ),
        radial_bins: int = 64,
        high_freq_radius_fraction: float = 0.5,
        dct_block_size: int = 8,
    ) -> None:
        known = {"radial_power_spectrum", "high_freq_energy_ratio", "dct_stats"}
        unknown = set(features) - known
        if unknown:
            raise ValueError(f"Unknown frequency features: {sorted(unknown)}")
        self.features = tuple(features)
        self.radial_bins = radial_bins
        self.high_freq_radius_fraction = high_freq_radius_fraction
        self.dct_block_size = dct_block_size
        self._dct = _dct_matrix(dct_block_size)
        self._radial_cache: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}

    def feature_names(self) -> list[str]:
        names: list[str] = []
        if "radial_power_spectrum" in self.features:
            names += [f"radial_{i:02d}" for i in range(self.radial_bins)]
        if "high_freq_energy_ratio" in self.features:
            names.append("hf_energy_ratio")
        if "dct_stats" in self.features:
            b = self.dct_block_size
            names += [f"dct_{u}{v}" for u in range(b) for v in range(b)]
        return names

    def _radial_index(self, h: int, w: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Flattened per-pixel ring index and normalized radius, cached per size."""
        if (h, w) not in self._radial_cache:
            cy, cx = h // 2, w // 2
            yy = torch.arange(h).unsqueeze(1).float() - cy
            xx = torch.arange(w).unsqueeze(0).float() - cx
            radius = torch.sqrt(yy**2 + xx**2)
            radius_norm = radius / radius.max()
            ring = (radius_norm * (self.radial_bins - 1)).round().long().flatten()
            self._radial_cache[(h, w)] = (ring, radius_norm.flatten())
        return self._radial_cache[(h, w)]

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.dim() == 3:
            batch = batch.unsqueeze(0)
        gray = batch.mean(dim=1)  # (B, H, W)
        b, h, w = gray.shape

        spectrum = torch.fft.fftshift(torch.fft.fft2(gray), dim=(-2, -1))
        power = spectrum.real**2 + spectrum.imag**2  # (B, H, W)

        parts: list[torch.Tensor] = []

        if "radial_power_spectrum" in self.features or "high_freq_energy_ratio" in self.features:
            ring, radius_norm = self._radial_index(h, w)
            power_flat = power.flatten(1)

        if "radial_power_spectrum" in self.features:
            log_power = torch.log(power_flat + _EPS)
            sums = torch.zeros(b, self.radial_bins).index_add_(1, ring, log_power)
            counts = torch.zeros(self.radial_bins).index_add_(
                0, ring, torch.ones_like(ring, dtype=torch.float)
            )
            parts.append(sums / counts)

        if "high_freq_energy_ratio" in self.features:
            hf_mask = (radius_norm > self.high_freq_radius_fraction).float()
            hf = (power_flat * hf_mask).sum(dim=1)
            total = power_flat.sum(dim=1)
            parts.append((hf / (total + _EPS)).unsqueeze(1))

        if "dct_stats" in self.features:
            k = self.dct_block_size
            if h % k or w % k:
                raise ValueError(f"image size {h}x{w} not divisible by DCT block {k}")
            blocks = (
                gray.unfold(1, k, k).unfold(2, k, k).reshape(b, -1, k, k)
            )  # (B, N, k, k)
            d = self._dct.to(gray.dtype)
            coeffs = d @ blocks @ d.T
            parts.append(torch.log(coeffs.abs() + _EPS).mean(dim=1).flatten(1))

        return torch.cat(parts, dim=1)
