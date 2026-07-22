"""Organize downloaded Synthbuster + RAISE-1k data into the raw-data layout.

Inputs (see download commands in README / Step 3 notes):
    data/downloads/synthbuster.zip       9 generators x 1000 PNGs (Zenodo)
    data/downloads/raise1k_tiff/*.TIF    1000 real camera TIFFs (RAISE-1k)

Outputs:
    data/raw/synthbuster/fake/<generator>/<source_id>.png   (moved from zip)
    data/raw/synthbuster/real/<source_id>.png               (converted TIFFs)

Real-image conversion: center-crop 1024x1024 at native resolution, saved as
lossless PNG. Rationale:
- Fakes are ~1024px natively; reals are ~16MP. Resizing both to network input
  size would downscale the classes by very different factors, handing the
  detector a resize-artifact shortcut instead of generator traces.
- Downscaling averages out the camera sensor noise that the residual detector
  is supposed to exploit; cropping preserves it exactly.
- PNG keeps the pipeline JPEG-free so compression robustness (Experiment 3)
  stays a controlled variable.

Usage:
    python scripts/prepare_synthbuster.py [--crop-size 1024] [--workers 6]
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import PROJECT_ROOT

Image.MAX_IMAGE_PIXELS = None  # RAISE TIFFs exceed PIL's decompression-bomb default

DOWNLOADS = PROJECT_ROOT / "data/downloads"
RAW_ROOT = PROJECT_ROOT / "data/raw/synthbuster"

# normalize Zenodo folder names to snake_case generator ids
GENERATOR_NAME_MAP = {
    "dalle2": "dalle2",
    "dalle3": "dalle3",
    "firefly": "firefly",
    "glide": "glide",
    "midjourney-v5": "midjourney_v5",
    "stable-diffusion-1-3": "stable_diffusion_13",
    "stable-diffusion-1-4": "stable_diffusion_14",
    "stable-diffusion-2": "stable_diffusion_2",
    "stable-diffusion-xl": "stable_diffusion_xl",
}


def extract_fakes(zip_path: Path) -> None:
    """Extract synthbuster.zip and arrange fakes as fake/<generator>/<id>.png."""
    # the zip root is a "synthbuster/" folder; reuse it if already extracted
    extract_dir = DOWNLOADS / "synthbuster"
    if not extract_dir.exists():
        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(DOWNLOADS)
    if not extract_dir.exists():
        raise FileNotFoundError(f"Expected {extract_dir} after extraction")

    # locate generator directories anywhere under the extraction root
    gen_dirs = {
        d.name: d
        for d in extract_dir.rglob("*")
        if d.is_dir() and d.name in GENERATOR_NAME_MAP
    }
    missing = set(GENERATOR_NAME_MAP) - set(gen_dirs)
    if missing:
        found = sorted(d.name for d in extract_dir.rglob("*") if d.is_dir())
        raise FileNotFoundError(
            f"Generator folders not found in zip: {sorted(missing)}; "
            f"directories present: {found}"
        )

    fake_root = RAW_ROOT / "fake"
    for zenodo_name, target_name in GENERATOR_NAME_MAP.items():
        target = fake_root / target_name
        if target.exists():
            continue
        target.mkdir(parents=True, exist_ok=True)
        # move only the images; folders also hold files.txt/generate.py/prompts.txt
        n = 0
        for png in sorted(gen_dirs[zenodo_name].glob("*.png")):
            shutil.move(str(png), str(target / png.name))
            n += 1
        print(f"  fake/{target_name}: {n} images")

    # copy the prompts file next to the data for provenance
    for prompts in extract_dir.rglob("prompts.csv"):
        shutil.copy(prompts, RAW_ROOT / "prompts.csv")
        break


def _convert_one(args: tuple[Path, Path, int]) -> str | None:
    """Center-crop one TIFF to crop_size and save as PNG. Returns error or None."""
    tiff_path, out_path, crop_size = args
    try:
        with Image.open(tiff_path) as img:
            img = img.convert("RGB")
            w, h = img.size
            if min(w, h) < crop_size:
                return f"{tiff_path.name}: too small ({w}x{h})"
            left = (w - crop_size) // 2
            top = (h - crop_size) // 2
            img.crop((left, top, left + crop_size, top + crop_size)).save(
                out_path, format="PNG"
            )
        return None
    except Exception as exc:  # corrupt/truncated downloads must not kill the batch
        return f"{tiff_path.name}: {exc}"


def convert_reals(tiff_dir: Path, crop_size: int, workers: int) -> None:
    real_dir = RAW_ROOT / "real"
    real_dir.mkdir(parents=True, exist_ok=True)

    tasks = []
    for tiff_path in sorted(tiff_dir.glob("*.TIF")):
        out_path = real_dir / f"{tiff_path.stem}.png"
        if not out_path.exists():
            tasks.append((tiff_path, out_path, crop_size))
    if not tasks:
        print("All real images already converted.")
        return

    print(f"Converting {len(tasks)} TIFFs to {crop_size}x{crop_size} PNGs ...")
    errors = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for err in tqdm(pool.map(_convert_one, tasks), total=len(tasks)):
            if err:
                errors.append(err)
    if errors:
        print(f"{len(errors)} conversions failed:")
        for err in errors:
            print(f"  {err}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-size", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--zip", type=Path, default=DOWNLOADS / "synthbuster.zip")
    parser.add_argument("--tiff-dir", type=Path, default=DOWNLOADS / "raise1k_tiff")
    parser.add_argument("--skip-fakes", action="store_true")
    parser.add_argument("--skip-reals", action="store_true")
    args = parser.parse_args()

    if not args.skip_fakes:
        extract_fakes(args.zip)
    if not args.skip_reals:
        convert_reals(args.tiff_dir, args.crop_size, args.workers)

    n_real = len(list((RAW_ROOT / "real").glob("*.png"))) if (RAW_ROOT / "real").exists() else 0
    n_fake = sum(
        1 for _ in (RAW_ROOT / "fake").rglob("*.png")
    ) if (RAW_ROOT / "fake").exists() else 0
    print(f"\ndata/raw/synthbuster: {n_real} real, {n_fake} fake images")


if __name__ == "__main__":
    main()
