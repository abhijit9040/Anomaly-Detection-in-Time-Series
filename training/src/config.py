"""
Training configuration loader.

Loads YAML config files and provides typed access to all
hyperparameters. Ensures reproducibility by setting seeds
across all libraries (FR-011, US-005).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml


@dataclass
class DatasetConfig:
    name: str = "SMD"
    source: str = ""
    machine_id: str = "machine-1-1"
    version: str = "1.0"


@dataclass
class PreprocessingConfig:
    window_size: int = 30
    step_size: int = 1
    val_split: float = 0.2
    scaler: str = "standard"


@dataclass
class ModelConfig:
    type: str = "lstm_autoencoder"
    hidden_size: int = 64
    latent_size: int = 32
    num_layers: int = 2
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    seed: int = 42
    batch_size: int = 64
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    optimizer: str = "adam"
    scheduler: str = "reduce_on_plateau"
    scheduler_patience: int = 5
    scheduler_factor: float = 0.5
    early_stopping_patience: int = 10
    gradient_clip_value: float = 1.0


@dataclass
class ThresholdConfig:
    method: str = "percentile"
    percentile: float = 95.0
    std_multiplier: float = 3.0


@dataclass
class EvaluationConfig:
    metrics: list[str] = field(
        default_factory=lambda: ["f1_score", "precision", "recall", "nab_score"]
    )


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    threshold: ThresholdConfig = field(default_factory=ThresholdConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)


def load_config(config_path: str | Path = None) -> ExperimentConfig:
    """
    Load experiment configuration from a YAML file.

    Args:
        config_path: Path to YAML config. Defaults to training/configs/default.yaml.

    Returns:
        ExperimentConfig with all settings.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    config = ExperimentConfig(
        dataset=DatasetConfig(**raw.get("dataset", {})),
        preprocessing=PreprocessingConfig(**raw.get("preprocessing", {})),
        model=ModelConfig(**raw.get("model", {})),
        training=TrainingConfig(**raw.get("training", {})),
        threshold=ThresholdConfig(**raw.get("threshold", {})),
        evaluation=EvaluationConfig(**raw.get("evaluation", {})),
    )

    return config


def set_seed(seed: int) -> None:
    """
    Set random seed across all libraries for reproducibility.

    This ensures identical results when re-running with the same
    configuration (US-005 acceptance criterion).

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass  # PyTorch not installed — skip


if __name__ == "__main__":
    # Quick test: load and print default config
    cfg = load_config()
    print(f"Dataset: {cfg.dataset.name} / {cfg.dataset.machine_id}")
    print(f"Window size: {cfg.preprocessing.window_size}")
    print(f"Hidden size: {cfg.model.hidden_size}")
    print(f"Seed: {cfg.training.seed}")
    print(f"Threshold method: {cfg.threshold.method} @ {cfg.threshold.percentile}th percentile")
