"""Train Detector 3: frequency features + shallow classifier.

Usage:
    python -m src.training.train_frequency --config configs/frequency.yaml

Feature extraction is the slow part (~2 min for 10k images), so extracted
features are cached to data/processed/ keyed by the splits CSV name; delete
the cache file after changing feature settings.

Outputs mirror the CNN experiments:
    checkpoints/<experiment>/model.pkl              scaler + classifier + names
    metrics/<experiment>/{val,test}_metrics.csv     per-generator metrics
    metrics/<experiment>/{val,test}_predictions.csv per-image scores
    metrics/<experiment>/feature_weights.csv        interpretability (logreg)
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.datasets.image_dataset import ImageCsvDataset
from src.datasets.transforms import build_transform
from src.features.frequency import FrequencyFeatureExtractor
from src.training.evaluate import per_generator_metrics
from src.utils.config import load_config
from src.utils.reproducibility import set_seed


def extract_features(cfg: dict, extractor: FrequencyFeatureExtractor) -> pd.DataFrame:
    """Extract features for every train/val/test row of the splits CSV,
    with caching under data/processed/."""
    csv_path = Path(cfg["paths"]["splits_csv"])
    cache = (
        csv_path.parents[1] / "processed" / f"frequency_features_{csv_path.stem}.csv"
    )
    names = extractor.feature_names()
    if cache.exists():
        cached = pd.read_csv(cache)
        if list(cached.columns[-len(names):]) == names:
            print(f"Loaded cached features: {cache}")
            return cached
        print("Cache exists but feature set changed; re-extracting.")

    transform = build_transform(
        image_size=cfg["data"]["image_size"], train=False, normalize=False
    )
    frames = []
    for split in ("train", "val", "test"):
        dataset = ImageCsvDataset(csv_path, split, transform)
        loader = DataLoader(
            dataset,
            batch_size=cfg["data"]["batch_size"],
            num_workers=cfg["data"].get("num_workers", 0),
            shuffle=False,
        )
        feats = [
            extractor(images).numpy()
            for images, _ in tqdm(loader, desc=f"extract {split}", leave=False)
        ]
        df = dataset.df.copy()
        frames.append(
            pd.concat(
                [df, pd.DataFrame(np.concatenate(feats), columns=names, index=df.index)],
                axis=1,
            )
        )
    out = pd.concat(frames, ignore_index=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    print(f"Cached features to {cache}")
    return out


def build_classifier(cfg: dict, seed: int):
    clf_cfg = cfg["classifier"]
    kind = clf_cfg["type"]
    if kind == "logistic_regression":
        params = clf_cfg.get("logistic_regression", {})
        return LogisticRegression(
            C=params.get("C", 1.0),
            max_iter=params.get("max_iter", 2000),
            class_weight="balanced",
            random_state=seed,
        )
    if kind == "random_forest":
        params = clf_cfg.get("random_forest", {})
        return RandomForestClassifier(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth"),
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(f"Unknown classifier type {kind!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/frequency.yaml")
    parser.add_argument("--experiment-suffix", default=None)
    parser.add_argument("--splits-csv", default=None,
                        help="override paths.splits_csv (e.g. the cross-generator split)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.splits_csv:
        cfg["paths"]["splits_csv"] = Path(args.splits_csv).resolve()
    if args.experiment_suffix:
        cfg["experiment_name"] = f"{cfg['experiment_name']}_{args.experiment_suffix}"
    set_seed(cfg["seed"])
    experiment = cfg["experiment_name"]

    extractor = FrequencyFeatureExtractor(
        features=tuple(cfg["input"]["features"]),
        radial_bins=cfg["input"].get("radial_bins", 64),
        high_freq_radius_fraction=cfg["input"].get("high_freq_radius_fraction", 0.5),
    )
    names = extractor.feature_names()
    all_df = extract_features(cfg, extractor)

    train = all_df[all_df["split"] == "train"]
    scaler = StandardScaler().fit(train[names])
    classifier = build_classifier(cfg, cfg["seed"])
    classifier.fit(scaler.transform(train[names]), train["label"])
    print(f"Trained {cfg['classifier']['type']} on {len(train)} images, {len(names)} features")

    metrics_dir = cfg["paths"]["metrics_dir"] / experiment
    ckpt_dir = cfg["paths"]["checkpoints_dir"] / experiment
    metrics_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(ckpt_dir / "model.pkl", "wb") as f:
        pickle.dump(
            {"scaler": scaler, "classifier": classifier, "feature_names": names,
             "config": args.config},
            f,
        )

    if hasattr(classifier, "coef_"):
        weights = pd.DataFrame({"feature": names, "weight": classifier.coef_[0]})
        weights.reindex(weights["weight"].abs().sort_values(ascending=False).index).to_csv(
            metrics_dir / "feature_weights.csv", index=False
        )

    for split in ("val", "test"):
        subset = all_df[all_df["split"] == split].reset_index(drop=True)
        pred_df = subset[["path", "label", "generator", "split", "dataset", "source_id"]].copy()
        pred_df["score"] = classifier.predict_proba(scaler.transform(subset[names]))[:, 1]
        pred_df.to_csv(metrics_dir / f"{split}_predictions.csv", index=False)
        metrics = per_generator_metrics(pred_df)
        metrics.to_csv(metrics_dir / f"{split}_metrics.csv", index=False)
        if split == "test":
            print("\nTest metrics (per generator vs real):")
            print(
                metrics[["scope", "auroc", "ap", "accuracy", "f1", "n_real", "n_fake"]]
                .to_string(index=False)
            )


if __name__ == "__main__":
    main()
