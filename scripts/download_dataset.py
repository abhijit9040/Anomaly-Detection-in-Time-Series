"""
Download the Server Machine Dataset (SMD) from GitHub.

The SMD dataset is hosted in the OmniAnomaly repository:
https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset

This script downloads:
- train/ directory (normal operation data)
- test/ directory (normal + anomalous data)
- test_label/ directory (binary anomaly labels)

Usage:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --output data/raw
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

# GitHub raw content base URL for the SMD dataset
BASE_URL = (
    "https://raw.githubusercontent.com/NetManAIOps/OmniAnomaly/"
    "master/ServerMachineDataset"
)

# All machine IDs in the dataset
MACHINE_IDS = [
    f"machine-{group}-{idx}"
    for group in range(1, 4)
    for idx in range(1, 9)
] + [
    "machine-3-9",
    "machine-3-10",
    "machine-3-11",
]

SPLITS = ["train", "test", "test_label"]


def download_file(url: str, dest: Path) -> bool:
    """Download a single file from URL to destination."""
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, str(dest))
        size_kb = dest.stat().st_size / 1024
        logger.info("  ✓ Downloaded: %s (%.1f KB)", dest.name, size_kb)
        return True
    except Exception as e:
        logger.error("  ✗ Failed: %s — %s", url, e)
        return False


def download_smd_dataset(output_dir: str | Path = None) -> dict:
    """
    Download the complete SMD dataset.

    Args:
        output_dir: Base output directory. The dataset will be saved to
                    {output_dir}/ServerMachineDataset/.

    Returns:
        Summary dict with download statistics.
    """
    if output_dir is None:
        # Default: project_root/data/raw
        output_dir = Path(__file__).parent.parent / "data" / "raw"
    output_dir = Path(output_dir)

    dataset_dir = output_dir / "ServerMachineDataset"

    logger.info("=" * 60)
    logger.info("Downloading Server Machine Dataset (SMD)")
    logger.info("Source: %s", BASE_URL)
    logger.info("Output: %s", dataset_dir)
    logger.info("Machines: %d", len(MACHINE_IDS))
    logger.info("=" * 60)

    total = 0
    success = 0
    failed = 0

    for split in SPLITS:
        logger.info("\n── %s ──", split)
        for machine_id in MACHINE_IDS:
            url = f"{BASE_URL}/{split}/{machine_id}.txt"
            dest = dataset_dir / split / f"{machine_id}.txt"

            total += 1
            if dest.exists():
                logger.info("  ⊘ Skipped (exists): %s", dest.name)
                success += 1
                continue

            if download_file(url, dest):
                success += 1
            else:
                failed += 1

    logger.info("\n" + "=" * 60)
    logger.info("Download complete: %d/%d succeeded, %d failed", success, total, failed)
    logger.info("Dataset saved to: %s", dataset_dir)
    logger.info("=" * 60)

    return {
        "total": total,
        "success": success,
        "failed": failed,
        "output_dir": str(dataset_dir),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download the Server Machine Dataset (SMD)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: data/raw/)",
    )
    args = parser.parse_args()

    result = download_smd_dataset(args.output)

    if result["failed"] > 0:
        logger.error("Some files failed to download. Please retry.")
        sys.exit(1)
    else:
        logger.info("All files downloaded successfully!")
        sys.exit(0)
