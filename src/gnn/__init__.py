"""Efficient and scalable Graph Neural Network components."""

from .config import load_config
from .data import create_dataloaders
from .model import CompatibleGCN

__all__ = [
    "load_config",
    "create_dataloaders",
    "CompatibleGCN",
]
