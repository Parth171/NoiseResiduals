"""Smoke tests for the config system: defaults merge, overrides win,
paths resolve to absolute locations under the project root."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_config


def test_baseline_rgb_config():
    cfg = load_config("configs/baseline_rgb.yaml")
    assert cfg["experiment_name"] == "baseline_rgb"
    # inherited from default.yaml
    assert cfg["seed"] == 42
    assert cfg["data"]["image_size"] == 224
    # experiment-specific
    assert cfg["model"]["arch"] == "resnet18"
    assert cfg["training"]["select_by"] == "auroc"
    # paths are resolved to absolute project-rooted paths
    for key, path in cfg["paths"].items():
        assert path.is_absolute(), f"paths.{key} is not absolute: {path}"
        assert str(path).startswith(str(PROJECT_ROOT))


def test_residual_config_overrides():
    cfg = load_config("configs/residual_cnn.yaml")
    assert cfg["input"]["type"] == "residual"
    assert cfg["input"]["residual_method"] == "gaussian"
    assert cfg["model"]["pretrained"] is False
    assert cfg["seed"] == 42


def test_frequency_config():
    cfg = load_config("configs/frequency.yaml")
    assert cfg["input"]["type"] == "frequency"
    assert cfg["classifier"]["type"] == "logistic_regression"
    assert "radial_power_spectrum" in cfg["input"]["features"]


if __name__ == "__main__":
    test_baseline_rgb_config()
    test_residual_config_overrides()
    test_frequency_config()
    print("All config tests passed.")
