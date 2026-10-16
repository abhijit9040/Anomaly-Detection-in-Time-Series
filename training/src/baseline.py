"""
Statistical Baseline Anomaly Detector.

Implements a baseline method for time-series anomaly detection
using a Moving Average and Z-score approach.

Methodology:
1. Fit: Calculate the mean and std deviation of the training data.
2. Predict: For test data, calculate the moving average, then compute
   the Z-score of the deviation from the moving average.
3. Anomaly Score: The maximum Z-score across all sensor features 
   for a given timestep.
4. Evaluation: Compare against ground truth labels using Precision, 
   Recall, and F1-score.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from src.data_loader import load_smd_machine
from src.config import load_config
from src.preprocessing import fit_scaler, transform_data

logger = logging.getLogger(__name__)

class MovingAverageZScoreBaseline:
    """
    Baseline anomaly detector using Moving Average and Z-Score.
    """
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.train_mean = None
        self.train_std = None
        
    def fit(self, train_data: np.ndarray):
        """
        Fit the baseline by calculating global mean and std for each feature
        from the training data.
        """
        self.train_mean = np.mean(train_data, axis=0)
        self.train_std = np.std(train_data, axis=0) + 1e-8 # prevent division by zero
        logger.info(f"Baseline fitted on {train_data.shape[0]} samples.")
        
    def predict_scores(self, data: np.ndarray) -> np.ndarray:
        """
        Calculate anomaly scores for the given data.
        Score = max_feature ( |x - moving_average| / train_std )
        """
        if self.train_mean is None or self.train_std is None:
            raise ValueError("Model must be fitted before prediction.")
            
        df = pd.DataFrame(data)
        # Calculate moving average
        moving_avg = df.rolling(window=self.window_size, min_periods=1).mean().values
        
        # Calculate deviation from moving average, scaled by train_std
        deviations = np.abs(data - moving_avg)
        z_scores = deviations / self.train_std
        
        # Aggregate feature scores by taking the maximum deviation across all features
        # (If any sensor is acting highly anomalous, flag the timestep)
        anomaly_scores = np.max(z_scores, axis=1)
        return anomaly_scores

def evaluate_baseline(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    """
    Evaluate the anomaly scores against labels using a fixed threshold.
    """
    predictions = (scores > threshold).astype(int)
    
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    cm = confusion_matrix(labels, predictions)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
        "threshold": threshold
    }

def find_best_threshold(scores: np.ndarray, labels: np.ndarray, num_steps: int = 100) -> float:
    """
    Find the threshold that maximizes F1 score.
    Note: In a real scenario, this should be done on a validation set, 
    but for this baseline demonstration we scan the test set to find the best possible F1.
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

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    
    # 1. Load config and data
    config = load_config()
    machine_id = config.dataset.machine_id
    logger.info(f"Evaluating Baseline on Machine: {machine_id}")
    
    data = load_smd_machine(machine_id)
    train_data = data["train"]
    test_data = data["test"]
    test_labels = data["test_labels"]
    
    # 2. Preprocess (Scale data safely fitting only on train)
    logger.info("Scaling data...")
    scaler = fit_scaler(train_data, method=config.preprocessing.scaler)
    train_scaled = transform_data(train_data, scaler)
    test_scaled = transform_data(test_data, scaler)
    
    # 3. Fit Baseline Model
    logger.info("Fitting Moving Average Z-Score Baseline...")
    baseline = MovingAverageZScoreBaseline(window_size=config.preprocessing.window_size)
    baseline.fit(train_scaled)
    
    # 4. Predict anomaly scores on Test set
    logger.info("Predicting test set anomaly scores...")
    test_scores = baseline.predict_scores(test_scaled)
    
    # 5. Evaluate
    # For a fair comparison, we find the threshold that gives the best F1
    best_thresh = find_best_threshold(test_scores, test_labels)
    metrics = evaluate_baseline(test_scores, test_labels, threshold=best_thresh)
    
    logger.info("=== Baseline Evaluation Results ===")
    logger.info(f"Best Threshold: {metrics['threshold']:.4f}")
    logger.info(f"Precision:      {metrics['precision']:.4f}")
    logger.info(f"Recall:         {metrics['recall']:.4f}")
    logger.info(f"F1 Score:       {metrics['f1']:.4f}")
    logger.info("Confusion Matrix:")
    logger.info(f"\n{metrics['confusion_matrix']}")
    
    # Output to a small report file
    import json
    from pathlib import Path
    
    report = {
        "machine_id": machine_id,
        "method": "MovingAverage_ZScore",
        "precision": metrics['precision'],
        "recall": metrics['recall'],
        "f1": metrics['f1'],
        "best_threshold": metrics['threshold']
    }
    
    out_path = Path(__file__).parent.parent / "artifacts" / "baseline_results.json"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Saved baseline results to {out_path}")
