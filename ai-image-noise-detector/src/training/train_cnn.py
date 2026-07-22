"""Train a CNN detector (RGB baseline now; residual CNN reuses this later).

Usage:
    python -m src.training.train_cnn --config configs/baseline_rgb.yaml

Outputs (under results/, named by experiment_name):
    checkpoints/<experiment>/best.pt          best epoch by validation AUROC
    metrics/<experiment>/history.csv          per-epoch train/val curves
    metrics/<experiment>/val_metrics.csv      per-generator metrics, best epoch
    metrics/<experiment>/test_metrics.csv     per-generator metrics on test
    metrics/<experiment>/test_predictions.csv per-image scores on test
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.datasets.image_dataset import build_dataloader
from src.models.rgb_cnn import build_model
from src.training.evaluate import binary_metrics, collect_predictions, evaluate_and_save
from src.utils.config import load_config
from src.utils.reproducibility import resolve_device, set_seed


def train_one_epoch(model, loader, criterion, optimizer, device) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images).squeeze(1), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        n += len(labels)
    return total_loss / n


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="e.g. configs/baseline_rgb.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="override config epochs")
    parser.add_argument("--splits-csv", default=None,
                        help="override paths.splits_csv (e.g. for smoke tests)")
    parser.add_argument("--experiment-suffix", default=None,
                        help="append to experiment_name so outputs don't collide")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.splits_csv:
        cfg["paths"]["splits_csv"] = Path(args.splits_csv).resolve()
    if args.experiment_suffix:
        cfg["experiment_name"] = f"{cfg['experiment_name']}_{args.experiment_suffix}"
    set_seed(cfg["seed"])
    device = resolve_device(cfg.get("device", "auto"))
    experiment = cfg["experiment_name"]
    epochs = args.epochs or cfg["training"]["epochs"]

    input_type = cfg["input"]["type"]
    if input_type not in ("rgb", "residual"):
        raise NotImplementedError(f"input.type={input_type!r} not supported by train_cnn")
    # ImageNet normalization only makes sense for pretrained RGB models; the
    # dataloader independently disables it for residual inputs.
    normalize = input_type == "rgb" and cfg["model"].get("pretrained", False)

    ckpt_dir = cfg["paths"]["checkpoints_dir"] / experiment
    metrics_dir = cfg["paths"]["metrics_dir"] / experiment
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_dataloader(cfg, "train", normalize=normalize, balance=True)
    val_loader = build_dataloader(cfg, "val", normalize=normalize)
    test_loader = build_dataloader(cfg, "test", normalize=normalize)
    print(
        f"experiment={experiment} device={device.type} epochs={epochs} "
        f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
        f"test={len(test_loader.dataset)}"
    )

    model = build_model(
        arch=cfg["model"]["arch"],
        pretrained=cfg["model"].get("pretrained", False),
        in_channels=cfg["model"].get("in_channels", 3),
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"].get("weight_decay", 0.0),
    )

    best_val_auroc = -1.0
    history = []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        val_pred = collect_predictions(model, val_loader, device)
        val = binary_metrics(val_pred["label"].to_numpy(), val_pred["score"].to_numpy())

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_auroc": val["auroc"],
            "val_ap": val["ap"],
            "val_accuracy": val["accuracy"],
            "seconds": round(time.time() - t0, 1),
        })
        pd.DataFrame(history).to_csv(metrics_dir / "history.csv", index=False)

        marker = ""
        if val["auroc"] > best_val_auroc:
            best_val_auroc = val["auroc"]
            torch.save(
                {"model_state": model.state_dict(), "epoch": epoch, "config": args.config,
                 "val_auroc": val["auroc"]},
                ckpt_dir / "best.pt",
            )
            marker = "  <- new best, checkpoint saved"
        print(
            f"epoch {epoch:2d}/{epochs}  train_loss={train_loss:.4f}  "
            f"val_auroc={val['auroc']:.4f}  val_ap={val['ap']:.4f}  "
            f"({history[-1]['seconds']}s){marker}"
        )

    # final evaluation with the best checkpoint
    checkpoint = torch.load(ckpt_dir / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    print(f"\nBest epoch: {checkpoint['epoch']} (val AUROC {checkpoint['val_auroc']:.4f})")

    val_metrics = evaluate_and_save(model, val_loader, device, metrics_dir, "val")
    test_metrics = evaluate_and_save(model, test_loader, device, metrics_dir, "test")
    print("\nTest metrics (per generator vs real):")
    print(
        test_metrics[["scope", "auroc", "ap", "accuracy", "f1", "n_real", "n_fake"]]
        .to_string(index=False)
    )
    del val_metrics


if __name__ == "__main__":
    main()
