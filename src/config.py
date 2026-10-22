"""Load the shared YAML config, with optional key=value overrides."""

from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def load_config(path=DEFAULT_CONFIG, overrides=None):
    """e.g. load_config(overrides=["train.epochs=5", "model.name=resnet18"])"""
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg
