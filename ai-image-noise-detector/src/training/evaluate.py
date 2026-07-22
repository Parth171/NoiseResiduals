"""Evaluation: scoring a model on a split and computing metrics.

Evaluation policy (see README): AUROC and Average Precision are the headline
metrics because they are insensitive to the 9-fakes-per-real class ratio.
Per-generator rows compare that generator's fakes against the SAME real
images (a balanced comparison on Synthbuster: 150 real vs 150 fake in the
test split). The pooled row over all fakes is supplementary; pooled accuracy
is never a main result.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> pd.DataFrame:
    """Run the model over a loader; return its dataset's metadata frame with a
    ``score`` column (sigmoid of the logit, = predicted P(fake))."""
    model.eval()
    scores: list[np.ndarray] = []
    for images, _ in tqdm(loader, desc="scoring", leave=False):
        logits = model(images.to(device)).squeeze(1)
        scores.append(torch.sigmoid(logits).cpu().numpy())
    df = loader.dataset.df.copy()
    df["score"] = np.concatenate(scores)
    return df


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    return {
        "auroc": roc_auc_score(labels, scores),
        "ap": average_precision_score(labels, scores),
        "accuracy": (preds == labels).mean(),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "n_real": int((labels == 0).sum()),
        "n_fake": int((labels == 1).sum()),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def per_generator_metrics(pred_df: pd.DataFrame) -> pd.DataFrame:
    """One metrics row per generator (its fakes vs the real images), plus a
    pooled row over all fakes."""
    real = pred_df[pred_df["label"] == 0]
    rows = []
    generators = sorted(pred_df.loc[pred_df["label"] == 1, "generator"].unique())
    for generator in generators:
        fake = pred_df[pred_df["generator"] == generator]
        subset = pd.concat([real, fake])
        rows.append(
            {"scope": generator}
            | binary_metrics(subset["label"].to_numpy(), subset["score"].to_numpy())
        )
    rows.append(
        {"scope": "pooled_all_generators"}
        | binary_metrics(pred_df["label"].to_numpy(), pred_df["score"].to_numpy())
    )
    return pd.DataFrame(rows)


def evaluate_and_save(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    output_dir: str | Path,
    prefix: str,
) -> pd.DataFrame:
    """Score a loader, write predictions and per-generator metrics CSVs, and
    return the metrics frame."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pred_df = collect_predictions(model, loader, device)
    pred_df.to_csv(output_dir / f"{prefix}_predictions.csv", index=False)

    metrics_df = per_generator_metrics(pred_df)
    metrics_df.to_csv(output_dir / f"{prefix}_metrics.csv", index=False)
    return metrics_df
