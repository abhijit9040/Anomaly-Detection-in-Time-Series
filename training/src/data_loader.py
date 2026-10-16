"""
SMD (Server Machine Dataset) Data Loader.

Loads train/test data and labels for the Server Machine Dataset.
Documents exact dataset schema as required by PRD §6.

Dataset Structure:
    ServerMachineDataset/
    ├── train/          # Normal operation data for training
    │   └── machine-X-Y.txt   # Space-separated, 38 features per line
    ├── test/           # Contains both normal and anomalous data
    │   └── machine-X-Y.txt
    └── test_label/     # Binary labels (0=normal, 1=anomaly)
        └── machine-X-Y.txt

Each machine has 38 sensor features sampled at regular intervals.
The train set contains ONLY normal data. The test set contains
both normal and anomalous segments with corresponding labels.

Source: https://github.com/NetManAIOps/OmniAnomaly
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default data directory relative to project root
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "ServerMachineDataset"

# All machine IDs in the SMD dataset
MACHINE_IDS = [
    f"machine-{group}-{idx}"
    for group in range(1, 4)
    for idx in range(1, 9)
] + [
    "machine-3-9",
    "machine-3-10",
    "machine-3-11",
]

# Number of sensor features per machine
NUM_FEATURES = 38


def load_smd_machine(
    machine_id: str,
    data_dir: str | Path = None,
) -> dict:
    """
    Load train data, test data, and test labels for a single SMD machine.

    Args:
        machine_id: Machine identifier, e.g. "machine-1-1".
        data_dir: Path to the ServerMachineDataset directory.
                  Defaults to data/raw/ServerMachineDataset/.

    Returns:
        Dictionary with keys:
            - "train": np.ndarray of shape (n_train, 38)
            - "test": np.ndarray of shape (n_test, 38)
            - "test_labels": np.ndarray of shape (n_test,) — binary
            - "machine_id": str
            - "num_features": int
            - "n_train": int
            - "n_test": int
            - "anomaly_ratio": float — fraction of test data labeled anomalous
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    data_dir = Path(data_dir)

    train_path = data_dir / "train" / f"{machine_id}.txt"
    test_path = data_dir / "test" / f"{machine_id}.txt"
    label_path = data_dir / "test_label" / f"{machine_id}.txt"

    # Validate paths exist
    for path, name in [(train_path, "train"), (test_path, "test"), (label_path, "test_label")]:
        if not path.exists():
            raise FileNotFoundError(
                f"{name} file not found: {path}\n"
                f"Run scripts/download_dataset.py first to download the SMD dataset."
            )

    # Load data — space-separated text files
    train_data = np.loadtxt(train_path, delimiter=",")
    test_data = np.loadtxt(test_path, delimiter=",")
    test_labels = np.loadtxt(label_path, delimiter=",").astype(int)

    # Validate shapes
    assert train_data.ndim == 2 and train_data.shape[1] == NUM_FEATURES, (
        f"Expected train shape (n, {NUM_FEATURES}), got {train_data.shape}"
    )
    assert test_data.ndim == 2 and test_data.shape[1] == NUM_FEATURES, (
        f"Expected test shape (n, {NUM_FEATURES}), got {test_data.shape}"
    )
    assert len(test_labels) == len(test_data), (
        f"Label count ({len(test_labels)}) != test count ({len(test_data)})"
    )

    anomaly_ratio = test_labels.sum() / len(test_labels)

    logger.info(
        "Loaded %s: train=%d, test=%d, features=%d, anomaly_ratio=%.3f",
        machine_id,
        len(train_data),
        len(test_data),
        train_data.shape[1],
        anomaly_ratio,
    )

    return {
        "train": train_data,
        "test": test_data,
        "test_labels": test_labels,
        "machine_id": machine_id,
        "num_features": train_data.shape[1],
        "n_train": len(train_data),
        "n_test": len(test_data),
        "anomaly_ratio": anomaly_ratio,
    }


def get_dataset_info(data_dir: str | Path = None) -> dict:
    """
    Get a summary of the entire SMD dataset.

    PRD §6 requires documenting:
    - Number of sensors/features
    - Row count
    - Sampling info
    - Anomaly label format
    - Class distribution

    Returns:
        Dictionary with dataset metadata.
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    data_dir = Path(data_dir)

    info = {
        "name": "Server Machine Dataset (SMD)",
        "source": "https://github.com/NetManAIOps/OmniAnomaly",
        "num_features": NUM_FEATURES,
        "feature_names": [f"sensor_{i}" for i in range(NUM_FEATURES)],
        "label_format": "Binary (0=normal, 1=anomaly)",
        "machines": [],
    }

    if not data_dir.exists():
        logger.warning("Dataset directory not found: %s", data_dir)
        info["status"] = "not_downloaded"
        return info

    for machine_id in sorted(MACHINE_IDS):
        train_path = data_dir / "train" / f"{machine_id}.txt"
        test_path = data_dir / "test" / f"{machine_id}.txt"
        label_path = data_dir / "test_label" / f"{machine_id}.txt"

        if not all(p.exists() for p in [train_path, test_path, label_path]):
            continue

        train_data = np.loadtxt(train_path, delimiter=",")
        test_labels = np.loadtxt(label_path, delimiter=",").astype(int)
        test_data = np.loadtxt(test_path, delimiter=",")

        info["machines"].append({
            "machine_id": machine_id,
            "n_train": len(train_data),
            "n_test": len(test_data),
            "anomaly_count": int(test_labels.sum()),
            "anomaly_ratio": float(test_labels.sum() / len(test_labels)),
        })

    info["total_machines"] = len(info["machines"])
    info["status"] = "available" if info["machines"] else "not_downloaded"

    return info


def load_as_dataframe(
    machine_id: str,
    data_dir: str | Path = None,
    split: str = "train",
) -> pd.DataFrame:
    """
    Load SMD data as a pandas DataFrame for EDA.

    Args:
        machine_id: Machine identifier.
        data_dir: Dataset directory path.
        split: "train" or "test".

    Returns:
        DataFrame with columns sensor_0 through sensor_37,
        plus 'label' column for test data.
    """
    data = load_smd_machine(machine_id, data_dir)
    feature_names = [f"sensor_{i}" for i in range(NUM_FEATURES)]

    if split == "train":
        df = pd.DataFrame(data["train"], columns=feature_names)
    elif split == "test":
        df = pd.DataFrame(data["test"], columns=feature_names)
        df["label"] = data["test_labels"]
    else:
        raise ValueError(f"Invalid split: {split}. Use 'train' or 'test'.")

    df["timestep"] = range(len(df))
    return df


if __name__ == "__main__":
    # Quick test: print dataset info
    logging.basicConfig(level=logging.INFO)
    info = get_dataset_info()
    print(f"Dataset: {info['name']}")
    print(f"Status: {info['status']}")
    print(f"Features: {info['num_features']}")
    if info["machines"]:
        print(f"Machines found: {info['total_machines']}")
        for m in info["machines"][:3]:
            print(f"  {m['machine_id']}: train={m['n_train']}, test={m['n_test']}, "
                  f"anomaly_ratio={m['anomaly_ratio']:.3f}")
