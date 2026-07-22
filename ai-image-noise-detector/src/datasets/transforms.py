"""Image transforms for training and evaluation.

Notes for a forensics project:
- We deliberately avoid horizontal flips and color jitter: they can alter the
  low-level noise statistics that the residual/frequency detectors rely on,
  and we want all three detectors trained on comparable inputs.
- Normalization is optional because ImageNet statistics make sense for the
  pretrained RGB baseline but not for residual maps.
"""

from __future__ import annotations

from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Shorter side is resized to this factor of the crop size before cropping
# (224 -> 256, the standard ImageNet convention).
RESIZE_FACTOR = 256 / 224


def build_transform(
    image_size: int = 224,
    train: bool = False,
    crop: str = "center",
    normalize: bool = True,
    residual=None,
) -> transforms.Compose:
    """Build a transform pipeline producing a float tensor in [0, 1] (or
    ImageNet-normalized if ``normalize=True``).

    Args:
        image_size: output height/width.
        train: whether this is the training pipeline.
        crop: "random" or "center"; random is only used when ``train=True``.
        normalize: apply ImageNet mean/std normalization (RGB models only).
        residual: optional callable (e.g. ``ResidualTransform``) applied to the
            [0, 1] tensor; mutually exclusive with ``normalize``.
    """
    if residual is not None and normalize:
        raise ValueError("residual extraction and ImageNet normalization are exclusive")
    ops: list = [transforms.Resize(round(image_size * RESIZE_FACTOR))]
    if train and crop == "random":
        ops.append(transforms.RandomCrop(image_size))
    else:
        ops.append(transforms.CenterCrop(image_size))
    ops.append(transforms.ToTensor())
    if normalize:
        ops.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    if residual is not None:
        ops.append(residual)
    return transforms.Compose(ops)
