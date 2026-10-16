"""
Evaluation script for the LSTM Autoencoder.

Features:
- Loads the best model weights and the fitted scaler.
- Computes reconstruction errors (MSE) per window.
- Selects the anomaly threshold strictly on the VALIDATION SET (no leakage).
- Evaluates the model on the unseen TEST SET using the selected threshold.
- Outputs Precision, Recall, and F1-score to compare against the baseline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from src.config import load_config
from src.data_loader import load_smd_machine
from src.preprocessing import preprocess_pipeline
from src.model import LSTMAutoencoder
from src.dataset import TimeSeriesDataset

logger = logging.getLogger(__name__)

def get_reconstruction_errors(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion: nn.Module, 
    device: torch.device
) -> np.ndarray:
    """
    Computes the reconstruction error for each window in the dataloader.
    Returns an array of errors of shape (n_samples,).
    """
    model.eval()
    errors = []
    
    # We don't reduce the loss globally, we want it per-window
    criterion.reduction = 'none' 
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            reconstruction = model(batch)
            
            # Loss per element: (batch_size, window_size, n_features)
            loss_per_element = criterion(reconstruction, batch)
            
            # Aggregate loss per window (mean over time and features)
            loss_per_window = torch.mean(loss_per_element, dim=[1, 2])
            
            errors.extend(loss_per_window.cpu().numpy())
            
    return np.array(errors)

def find_best_threshold_f1(scores: np.ndarray, labels: np.ndarray, num_steps: int = 200) -> float:
    """
    Find the threshold that maximizes F1 score.
    """
    min_score = np.min(scores)
    max_score = np.max(scores)
    thresholds = np.linspace(min_score, max_score, num_steps)
    
    best_f1 = 0.0
    best_thresh = min_score
    
    for thresh in thresholds:
        preds = (scores > thresh).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            
    return best_thresh

def evaluate_model():
    """Main evaluation orchestrator."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    
    # 1. Setup & Config
    config = load_config()
    machine_id = config.dataset.machine_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    artifacts_dir = Path(__file__).parent.parent / "artifacts"
    model_path = artifacts_dir / f"lstm_autoencoder_{machine_id}.pt"
    
    if not model_path.exists():
        logger.error(f"Model not found at {model_path}. Run train.py first.")
        return
        
    # 2. Data Loading & Preprocessing
    logger.info(f"Loading data for machine: {machine_id}")
    raw_data = load_smd_machine(machine_id)
    
    # Note: test_labels inside preprocessing is returned for window_labels
    prep_result = preprocess_pipeline(
        train_raw=raw_data["train"],
        test_raw=raw_data["test"],
        test_labels=raw_data["test_labels"],
        window_size=config.preprocessing.window_size,
        step_size=config.preprocessing.step_size,
        val_split=config.preprocessing.val_split,
        scaler_method=config.preprocessing.scaler,
        seed=config.training.seed
    )
    
    val_dataset = TimeSeriesDataset(prep_result["val_windows"])
    test_dataset = TimeSeriesDataset(prep_result["test_windows"])
    test_labels = prep_result["test_window_labels"]
    
    # We need validation labels to pick a threshold dynamically.
    # But wait, SMD train sets ONLY have normal data (no anomalies).
    # If the validation set has no anomalies, we can't optimize F1 on it!
    # Instead, we will use a statistical threshold (e.g. mean + 3*std of val errors)
    # per PRD guidelines for unsupervised anomaly detection.
    
    val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)
    
    # 3. Model Initialization & Loading
    model = LSTMAutoencoder(
        window_size=config.preprocessing.window_size,
        n_features=raw_data["num_features"],
        hidden_dim=config.model.hidden_size,
        n_layers=config.model.num_layers,
        dropout=config.model.dropout
    ).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    logger.info("Loaded pre-trained model weights.")
    
    criterion = nn.MSELoss()
    
    # 4. Validation Phase (Threshold Selection)
    logger.info("Calculating reconstruction errors for Validation set...")
    val_errors = get_reconstruction_errors(model, val_loader, criterion, device)
    
    # Since Val set contains ONLY normal data, we set threshold statistically
    # Threshold = mean(val_errors) + z * std(val_errors)
    z_multiplier = config.threshold.std_multiplier
    val_mean = np.mean(val_errors)
    val_std = np.std(val_errors)
    threshold = val_mean + (z_multiplier * val_std)
    
    logger.info(f"Validation Errors - Mean: {val_mean:.6f}, Std: {val_std:.6f}")
    logger.info(f"Selected Anomaly Threshold (Mean + {z_multiplier}*Std): {threshold:.6f}")
    
    # 5. Test Phase (Evaluation)
    logger.info("Calculating reconstruction errors for Test set...")
    test_errors = get_reconstruction_errors(model, test_loader, criterion, device)
    
    predictions = (test_errors > threshold).astype(int)
    
    precision = precision_score(test_labels, predictions, zero_division=0)
    recall = recall_score(test_labels, predictions, zero_division=0)
    f1 = f1_score(test_labels, predictions, zero_division=0)
    cm = confusion_matrix(test_labels, predictions)
    
    logger.info("=== Test Set Evaluation Results ===")
    logger.info(f"Precision: {precision:.4f}")
    logger.info(f"Recall:    {recall:.4f}")
    logger.info(f"F1 Score:  {f1:.4f}")
    logger.info("Confusion Matrix:")
    logger.info(f"\n{cm}")
    
    # 6. Save Report
    report = {
        "machine_id": machine_id,
        "method": "LSTM_Autoencoder",
        "threshold": float(threshold),
        "threshold_strategy": f"Val Mean + {z_multiplier}*Std",
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }
    
    out_path = artifacts_dir / "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved evaluation results to {out_path}")

if __name__ == "__main__":
    evaluate_model()
