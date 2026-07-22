"""Noise residual extraction.

A residual is what remains after removing an image's low-frequency content:

    residual = image - smoothed(image)      (gaussian, median)

or the direct response of a high-pass filter (laplacian, sobel, SRM). Camera
images carry physical sensor noise in these high frequencies, while generated
images can show unnaturally smooth or structured residuals - this is the
signal the residual CNN classifies.

All functions take a float32 tensor of shape (C, H, W) in [0, 1] and return a
float32 tensor of shape (C, H, W) (SRM: 3 channels regardless of input),
roughly centered on 0. No normalization is applied afterwards; the CNN trains
directly on these values.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

_LAPLACIAN = torch.tensor(
    [[0.0, 1.0, 0.0],
     [1.0, -4.0, 1.0],
     [0.0, 1.0, 0.0]]
)
_SOBEL_X = torch.tensor(
    [[-1.0, 0.0, 1.0],
     [-2.0, 0.0, 2.0],
     [-1.0, 0.0, 1.0]]
)

# Three classic SRM high-pass filters from the steganalysis literature
# (first-order, second-order, and the 5x5 "SQUARE" predictor), normalized so
# outputs are prediction errors on a comparable scale.
_SRM_FIRST_ORDER = torch.tensor(
    [[0.0, 0.0, 0.0],
     [0.0, -1.0, 1.0],
     [0.0, 0.0, 0.0]]
)
_SRM_SECOND_ORDER = torch.tensor(
    [[0.0, 1.0, 0.0],
     [1.0, -4.0, 1.0],
     [0.0, 1.0, 0.0]]
) / 4.0
_SRM_SQUARE_5X5 = torch.tensor(
    [[-1.0,  2.0,  -2.0,  2.0, -1.0],
     [ 2.0, -6.0,   8.0, -6.0,  2.0],
     [-2.0,  8.0, -12.0,  8.0, -2.0],
     [ 2.0, -6.0,   8.0, -6.0,  2.0],
     [-1.0,  2.0,  -2.0,  2.0, -1.0]]
) / 12.0


def _conv_per_channel(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Convolve each channel independently with a 2D kernel (reflect padding)."""
    c = x.shape[0]
    k = kernel.to(x.dtype).expand(c, 1, *kernel.shape)
    pad = kernel.shape[-1] // 2
    xp = F.pad(x.unsqueeze(0), (pad, pad, pad, pad), mode="reflect")
    return F.conv2d(xp, k, groups=c).squeeze(0)


def compute_gaussian_residual(
    x: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0
) -> torch.Tensor:
    """residual = x - gaussian_blur(x)."""
    return x - gaussian_blur(x, [kernel_size, kernel_size], [sigma, sigma])


def compute_laplacian_residual(x: torch.Tensor) -> torch.Tensor:
    """Laplacian high-pass response per channel."""
    return _conv_per_channel(x, _LAPLACIAN)


def compute_sobel_residual(x: torch.Tensor) -> torch.Tensor:
    """Sobel gradient magnitude per channel."""
    gx = _conv_per_channel(x, _SOBEL_X)
    gy = _conv_per_channel(x, _SOBEL_X.T)
    return torch.sqrt(gx**2 + gy**2 + 1e-12)


def compute_median_residual(x: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """residual = x - median_blur(x). Median filtering is a strong denoiser,
    so this residual isolates noise-like content particularly well."""
    pad = kernel_size // 2
    xp = F.pad(x.unsqueeze(0), (pad, pad, pad, pad), mode="reflect")
    patches = xp.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
    c, h, w = x.shape
    median = patches.contiguous().view(1, c, h, w, -1).median(dim=-1).values
    return x - median.squeeze(0)


def compute_srm_residual(x: torch.Tensor) -> torch.Tensor:
    """Apply 3 classic SRM filters to the grayscale image -> 3 channels."""
    gray = x.mean(dim=0, keepdim=True)
    outputs = [
        _conv_per_channel(gray, kernel)
        for kernel in (_SRM_FIRST_ORDER, _SRM_SECOND_ORDER, _SRM_SQUARE_5X5)
    ]
    return torch.cat(outputs, dim=0)


class ResidualTransform:
    """Transform step turning an image tensor into its residual, configured by
    the ``input`` section of a residual experiment config.

    Must stay picklable (no lambdas/closures): DataLoader workers on macOS use
    the "spawn" start method, which pickles transforms into each worker.
    """

    METHODS = ("gaussian", "laplacian", "sobel", "median", "srm")

    def __init__(
        self,
        method: str = "gaussian",
        kernel_size: int = 5,
        gaussian_sigma: float = 1.0,
    ) -> None:
        if method not in self.METHODS:
            raise ValueError(f"Unknown residual method {method!r}; choose from {self.METHODS}")
        self.method = method
        self.kernel_size = kernel_size
        self.gaussian_sigma = gaussian_sigma

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.method == "gaussian":
            return compute_gaussian_residual(x, self.kernel_size, self.gaussian_sigma)
        if self.method == "laplacian":
            return compute_laplacian_residual(x)
        if self.method == "sobel":
            return compute_sobel_residual(x)
        if self.method == "median":
            return compute_median_residual(x, self.kernel_size)
        return compute_srm_residual(x)

    def __repr__(self) -> str:
        return f"ResidualTransform(method={self.method!r})"
