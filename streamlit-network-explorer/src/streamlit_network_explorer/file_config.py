
from __future__ import annotations
from pathlib import Path
import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "data_config.yaml"

def load_config(path: str | None = None) -> dict:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "data" not in cfg:
        raise ValueError("Invalid data_config.yaml: missing 'data' key")
    return cfg
