"""Configuration utilities for the Efficient GNN project."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

import yaml


@dataclass
class ModelConfig:
    """Hyper-parameters for GNN models."""

    model_type: str = "sage"  # "sage", "gcn", or "gat"
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.5
    fanouts: Iterable[int] = (15, 10, 5)  # Only used for SAGE
    use_residual: bool = True
    use_batch_norm: bool = True
    activation: str = "relu"
    gradient_checkpointing: bool = False
    # GCN-specific parameters
    cached: bool = True
    normalize: bool = True
    add_self_loops: bool = True
    improved: bool = False
    # GAT-specific parameters
    heads: int = 8
    attn_dropout: float = 0.0
    concat: bool = True
    negative_slope: float = 0.2
    edge_dim: Optional[int] = None
    fill_value: str = "mean"


@dataclass
class OptimConfig:
    """Optimizer and learning-rate schedule settings."""

    lr: float = 0.003
    weight_decay: float = 5e-4
    betas: Iterable[float] = (0.9, 0.999)
    epochs: int = 40
    warmup_epochs: int = 3


@dataclass
class TrainingConfig:
    """Training loop behaviour."""

    batch_size: int = 1024
    eval_batch_size: int = 4096
    num_workers: int = 8
    use_amp: bool = True
    log_every: int = 25
    patience: int = 10


@dataclass
class DatasetConfig:
    """Dataset selection for node property prediction."""

    name: str = "ogbn-products"
    root: str = "./data"
    dataset_type: str = "ogb"  # "ogb" or "planetoid"


@dataclass
class InferenceConfig:
    """Inference-related options."""

    chunk_size: Optional[int] = 100_000
    num_workers: int = 8


@dataclass
class ExperimentConfig:
    """Top-level configuration dataclass."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    device: str = "cuda"
    seed: int = 42


def _update_dataclass(instance: Any, updates: Mapping[str, Any]) -> None:
    """Recursively apply dictionary updates to nested dataclasses."""

    for key, value in updates.items():
        if not hasattr(instance, key):
            raise KeyError(f"Unknown config field: {key}")
        current = getattr(instance, key)
        if dataclass_is_instance(current) and isinstance(value, Mapping):
            _update_dataclass(current, value)
        else:
            setattr(instance, key, value)


def dataclass_is_instance(obj: Any) -> bool:
    """Return ``True`` when the object is an instance of a dataclass."""

    return hasattr(obj, "__dataclass_fields__")


def load_config(path: Optional[str] = None) -> ExperimentConfig:
    """Load the experiment configuration from YAML or defaults.

    Parameters
    ----------
    path:
        Optional path to a YAML configuration file. When omitted a copy of the
        default configuration is returned.
    """

    config = ExperimentConfig()
    if path is None:
        return config

    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        raw_cfg: Dict[str, Any] = yaml.safe_load(f) or {}

    if not isinstance(raw_cfg, Mapping):
        raise TypeError("Top-level configuration must be a mapping")

    _update_dataclass(config, raw_cfg)
    return config


def config_to_dict(config: ExperimentConfig) -> Dict[str, Any]:
    """Convert a config dataclass into a plain dictionary."""

    def _convert(value: Any) -> Any:
        if dataclass_is_instance(value):
            return {k: _convert(v) for k, v in value.__dict__.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(_convert(v) for v in value)
        return value

    return _convert(config)
