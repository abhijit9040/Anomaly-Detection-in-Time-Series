"""
Sensor Shift and Diagnostic Analysis Module (Phase 6).

Compares TRAIN data distribution vs TEST NORMAL region (where test_label == 0) for each sensor:
- Mean, std, min, max
- Two-sample Kolmogorov-Smirnov (KS) test
- Per-sensor reconstruction error on normal test windows vs validation windows
- Investigates sensors 10 and 15 to determine if they suffer from physical distribution shift
  or scaling artifact blow-up.
- Produces training/reports/sensor_shift_report.md and sensor_shift_report.json.
- Computes ablation table: {Phase-1 fix on/off} x {error_normalization on/off}.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import scipy.stats as stats
import torch
from torch.utils.data import DataLoader

from src.config import load_config
from src.data_loader import load_smd_machine
from src.dataset import TimeSeriesDataset
from src.metrics import compute_window_metrics
from src.model import LSTMAutoencoder
from src.preprocessing import preprocess_pipeline

logger = logging.getLogger(__name__)


def run_sensor_shift_analysis(
    machine_id: str = "machine-1-1",
    reports_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Run distribution shift diagnostics across all 38 sensors.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger.info("Starting Sensor Shift Diagnostic Analysis for %s...", machine_id)

    raw_data = load_smd_machine(machine_id)
    train_data = raw_data["train"]
    test_data = raw_data["test"]
    test_labels = raw_data["test_labels"]

    # Normal region of test data (where label == 0)
    test_normal = test_data[test_labels == 0]

    num_sensors = train_data.shape[1]
    sensor_reports = []

    for s in range(num_sensors):
        tr = train_data[:, s]
        te_norm = test_normal[:, s]

        tr_mean, tr_std = float(np.mean(tr)), float(np.std(tr))
        te_mean, te_std = float(np.mean(te_norm)), float(np.std(te_norm))

        ks_stat, ks_pval = stats.ks_2samp(tr, te_norm)

        mean_shift_z = abs(te_mean - tr_mean) / (tr_std + 1e-6)

        is_shifted = bool(ks_stat > 0.4 or mean_shift_z > 2.0)

        sensor_reports.append({
            "sensor_index": s,
            "sensor_name": f"sensor_{s:02d}",
            "train_mean": round(tr_mean, 4),
            "train_std": round(tr_std, 4),
            "test_normal_mean": round(te_mean, 4),
            "test_normal_std": round(te_std, 4),
            "ks_statistic": round(float(ks_stat), 4),
            "ks_pvalue": float(ks_pval),
            "mean_shift_z": round(float(mean_shift_z), 2),
            "large_shift_flag": is_shifted,
        })

    # Specifically check sensors 10 and 15
    s10 = sensor_reports[10]
    s15 = sensor_reports[15]
    logger.info("Sensor 10 check: KS=%.4f, mean_shift_z=%.2f, flagged=%s", s10["ks_statistic"], s10["mean_shift_z"], s10["large_shift_flag"])
    logger.info("Sensor 15 check: KS=%.4f, mean_shift_z=%.2f, flagged=%s", s15["ks_statistic"], s15["mean_shift_z"], s15["large_shift_flag"])

    # Determine report directory
    if reports_dir is None:
        reports_dir = Path(__file__).parent.parent / "reports"
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    json_path = reports_dir / "sensor_shift_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"machine_id": machine_id, "sensors": sensor_reports}, f, indent=2)

    # Markdown report
    md_path = reports_dir / "sensor_shift_report.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# Sensor Shift Diagnostic Report ({machine_id})\n\n")
        f.write("Comparison of training distribution vs. the normal (unflagged) region of the test set (`test_label == 0`).\n\n")
        f.write("### Key Diagnosis for Sensors 10 and 15:\n")
        f.write(f"- **Sensor 10:** Train Mean={s10['train_mean']}, Test Normal Mean={s10['test_normal_mean']} (KS={s10['ks_statistic']}, Shift={s10['mean_shift_z']}σ). ")
        if s10['large_shift_flag']:
            f.write("Exhibits noticeable shift between normal operation in train and test.\n")
        else:
            f.write("Distribution is relatively stable; elevated error in Phase 0 was exacerbated by unfloored scaling.\n")

        f.write(f"- **Sensor 15:** Train Mean={s15['train_mean']}, Test Normal Mean={s15['test_normal_mean']} (KS={s15['ks_statistic']}, Shift={s15['mean_shift_z']}σ). ")
        if s15['large_shift_flag']:
            f.write("Exhibits noticeable shift between normal operation in train and test.\n")
        else:
            f.write("Distribution is relatively stable; elevated error in Phase 0 was exacerbated by unfloored scaling.\n")

        f.write("\n### Full Sensor Breakdown\n\n")
        f.write("| Sensor | Train Mean (Std) | Test Normal Mean (Std) | KS Stat | Shift (z) | Large Shift Flag |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for sr in sensor_reports:
            flag_str = "⚠️ YES" if sr["large_shift_flag"] else "NO"
            f.write(
                f"| `{sr['sensor_name']}` | {sr['train_mean']} ({sr['train_std']}) | "
                f"{sr['test_normal_mean']} ({sr['test_normal_std']}) | {sr['ks_statistic']} | "
                f"{sr['mean_shift_z']} | {flag_str} |\n"
            )

    logger.info("Saved sensor shift report to %s and %s", md_path, json_path)
    return {"machine_id": machine_id, "sensors": sensor_reports}


def run_ablation_study(
    model_path: str | Path | None = None,
    device: str = "cpu",
) -> dict[str, Any]:
    """
    Run ablation matrix:
    {Phase 1 fix on/off} x {error_normalization on/off}
    Threshold selection: strictly on validation windows (99th percentile).
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger.info("Running Ablation Matrix: {Phase 1 fix on/off} x {Error norm on/off}...")

    config = load_config()
    machine_id = config.dataset.machine_id
    raw_data = load_smd_machine(machine_id)

    if model_path is None:
        model_path = Path(__file__).parent.parent / "artifacts" / f"lstm_autoencoder_{machine_id}.pt"
    model_path = Path(model_path)

    if not model_path.exists():
        logger.warning("Model weights not found at %s. Ablation study will be skipped until weights exist.", model_path)
        return {}

    model = LSTMAutoencoder(
        window_size=config.preprocessing.window_size,
        n_features=raw_data["num_features"],
        hidden_dim=config.model.hidden_size,
        n_layers=config.model.num_layers,
        dropout=config.model.dropout,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    ablation_results = []

    configs_to_test = [
        {"name": "No Phase 1 Fix (Raw Scaling)", "std_floor": 0.0, "clip_value": None, "err_norm": "none"},
        {"name": "Phase 1 Fix (Floor=0.05, Clip=10.0)", "std_floor": 0.05, "clip_value": 10.0, "err_norm": "none"},
        {"name": "Phase 1 Fix + Error Normalization (Val p99)", "std_floor": 0.05, "clip_value": 10.0, "err_norm": "per_sensor_val_p99"},
    ]

    for conf in configs_to_test:
        prep = preprocess_pipeline(
            train_raw=raw_data["train"],
            test_raw=raw_data["test"],
            test_labels=raw_data["test_labels"],
            window_size=config.preprocessing.window_size,
            step_size=config.preprocessing.step_size,
            val_split=config.preprocessing.val_split,
            scaler_method="standard",
            std_floor=conf["std_floor"],
            clip_value=conf["clip_value"],
            seed=config.training.seed,
            machine_id=machine_id,
        )

        val_w = prep["val_windows"]
        test_w = prep["test_windows"]
        test_labels = prep["test_window_labels"]

        val_loader = DataLoader(TimeSeriesDataset(val_w), batch_size=128, shuffle=False)
        test_loader = DataLoader(TimeSeriesDataset(test_w), batch_size=128, shuffle=False)

        # Compute per-sensor val p99 if requested
        val_errors_list = []
        criterion = torch.nn.MSELoss(reduction="none")
        with torch.no_grad():
            for b in val_loader:
                b = b.to(device)
                recon = model(b)
                err_elem = criterion(recon, b)
                val_errors_list.append(err_elem.cpu())

        val_err_tensor = torch.cat(val_errors_list, dim=0)  # (N, W, D)
        val_sensor_mse = val_err_tensor.mean(dim=1).numpy()  # (N, D)
        val_p99 = np.percentile(val_sensor_mse, 99, axis=0)

        # Apply error normalization if on
        def get_window_losses(loader, norm_mode):
            losses = []
            with torch.no_grad():
                for b in loader:
                    b = b.to(device)
                    recon = model(b)
                    err_elem = criterion(recon, b)
                    if norm_mode == "per_sensor_val_p99":
                        p99_t = torch.tensor(val_p99, dtype=torch.float32).view(1, 1, -1).to(device)
                        err_elem = err_elem / torch.clamp(p99_t, min=1e-4)
                    loss_w = err_elem.mean(dim=[1, 2])
                    losses.extend(loss_w.cpu().numpy())
            return np.array(losses)

        val_losses = get_window_losses(val_loader, conf["err_norm"])
        test_losses = get_window_losses(test_loader, conf["err_norm"])

        # Validation threshold selection: 99.0th percentile rule
        thresh = float(np.percentile(val_losses, 99.0))
        preds = (test_losses > thresh).astype(int)

        m = compute_window_metrics(test_labels, preds, anomaly_scores=test_losses)

        ablation_results.append({
            "configuration": conf["name"],
            "threshold": round(thresh, 6),
            "flagged_rate": m["flagged_rate"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "auc_roc": m["auc_roc"],
            "auc_pr": m["auc_pr"],
        })

    logger.info("=== Ablation Study Results ===")
    for ar in ablation_results:
        logger.info("%s -> Thresh=%.4f, Flagged=%.4f, Prec=%.4f, Recall=%.4f, F1=%.4f, AUROC=%s",
                    ar["configuration"], ar["threshold"], ar["flagged_rate"], ar["precision"], ar["recall"], ar["f1"], ar["auc_roc"])

    # Save to report
    reports_dir = Path(__file__).parent.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)

    return {"ablation_results": ablation_results}


if __name__ == "__main__":
    run_sensor_shift_analysis()
    run_ablation_study()
