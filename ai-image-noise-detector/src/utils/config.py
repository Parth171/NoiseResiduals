"""Config loading: experiment YAMLs merged over configs/default.yaml.

Every entry point takes a single experiment config path (e.g.
``configs/baseline_rgb.yaml``). Shared settings live in ``configs/default.yaml``
so nothing is hardcoded in code and experiments only specify what differs.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into ``base`` (override wins)."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(experiment_config: str | Path) -> dict[str, Any]:
    """Load an experiment config merged over the shared defaults.

    Relative paths inside ``cfg["paths"]`` are resolved against the project
    root, so scripts behave the same regardless of the working directory.
    """
    with open(DEFAULT_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f) or {}

    experiment_config = Path(experiment_config)
    if not experiment_config.is_absolute():
        experiment_config = PROJECT_ROOT / experiment_config
    with open(experiment_config) as f:
        overrides = yaml.safe_load(f) or {}

    cfg = _deep_merge(cfg, overrides)

    for key, value in cfg.get("paths", {}).items():
        path = Path(value)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        cfg["paths"][key] = path

    return cfg
