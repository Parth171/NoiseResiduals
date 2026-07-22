# Detecting AI-Generated Images Using Noise Residuals and Frequency Artifacts

This repository explores AI-generated image detection using noise residuals and
frequency artifacts. Real camera images contain physical sensor and acquisition
noise, while generated images may contain unnatural statistical patterns. We
compare RGB CNN, residual CNN, and frequency-based classifiers on real/fake
image benchmarks, with a focus on cross-generator generalization and robustness
to JPEG compression, resizing, and blur.

## Detectors

| Detector | Input | Model |
|---|---|---|
| RGB CNN baseline | 224x224 RGB image | ResNet18 |
| Noise Residual CNN | high-pass residual map (e.g. `x - blur(x)`) | ResNet18 |
| Frequency classifier | FFT/DCT summary features | logistic regression / random forest |

## Experiments

1. In-distribution detection (train/val/test on mixed generators)
2. Cross-generator generalization (held-out generators at test time)
3. JPEG compression robustness (q90 / q65 / q40)
4. Resize and blur robustness
5. Visualization and interpretability (residual maps, spectra, per-generator results)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Repository layout

```
configs/            experiment configs (YAML); no paths are hardcoded in code
data/
  raw/              original downloaded images (not committed)
  processed/        cached/preprocessed artifacts
  splits/           metadata CSVs defining train/val/test splits
notebooks/          exploratory analysis and figures
src/
  datasets/         metadata-driven PyTorch datasets and transforms
  features/         residual and frequency feature extraction
  models/           model definitions
  training/         training and evaluation entry points
  analysis/         robustness evaluation and plotting
  utils/            config loading, seeding
reports/
  figures/          generated plots for the paper
  paper.md          research report
results/
  metrics/          metrics and predictions CSVs
  checkpoints/      model checkpoints
scripts/            data download / preparation scripts
tests/              unit and smoke tests
```

## Dataset: Synthbuster + RAISE-1k

- **Fake** (label 1): [Synthbuster](https://zenodo.org/records/10066460) —
  9 generators x 1,000 images (DALL·E 2/3, Firefly, Glide, Midjourney v5,
  SD 1.3/1.4/2/XL), distributed as uncompressed PNGs.
- **Real** (label 0): [RAISE-1k](http://loki.disi.unitn.it/RAISE/) — 1,000
  camera-native TIFFs. We store lossless 1024x1024 center crops (no resizing,
  no JPEG) so sensor noise is preserved and both classes go through comparable
  processing (fakes are ~1024px natively).

To reproduce the download and preparation:

```bash
# fakes (12.4 GB)
curl -L -o data/downloads/synthbuster.zip \
  "https://zenodo.org/records/10066460/files/synthbuster.zip?download=1"
# reals: file list, then ~14 GB of TIFFs
curl -o data/downloads/RAISE_1k_csv.zip "http://loki.disi.unitn.it/RAISE/getFile.php?p=1k"
unzip data/downloads/RAISE_1k_csv.zip -d data/downloads
# (extract the TIFF column into raise1k_tiff_urls.txt, then)
xargs -P 6 -n 1 curl -s --retry 5 -C - --remote-name --output-dir data/downloads/raise1k_tiff \
  < data/downloads/raise1k_tiff_urls.txt
python scripts/prepare_synthbuster.py
```

### Class imbalance and evaluation policy

The dataset is 9,000 fake vs 1,000 real. We handle this explicitly:

- **Training** uses class weighting (or a balanced sampler) so the model
  cannot win by always predicting "fake".
- **Headline metrics are AUROC and Average Precision**, which are robust to
  class ratio. Pooled 9k-vs-1k accuracy is never reported as a main result.
- **Per-generator evaluation**: each generator's fakes are compared against
  the real images separately (a balanced comparison by construction, since
  Synthbuster pairs every real photo with one fake per generator).

### Source-grouped splits (leakage control)

Every Synthbuster fake is generated from a prompt derived from a specific
RAISE-1k photo. Splits are therefore **grouped by `source_id`**: a real photo
and all fakes derived from it always land in the same split, so the test set
shares no content with training. In the cross-generator split, held-out
generator fakes are kept only for test-partition sources (rows excluded by
this policy are marked `split=unused` in the CSV for provenance).

## Reproducibility conventions

- Label convention: `0 = real`, `1 = AI-generated`.
- All dataset splits are materialized as metadata CSVs in `data/splits/` with
  columns `path,label,generator,split,dataset,source_id` and committed to git.
- Every experiment is driven by a YAML config in `configs/` and saves metrics
  to CSV under `results/metrics/`.
- Random seeds are set from the config (`seed` key) for Python, NumPy, and
  PyTorch.
