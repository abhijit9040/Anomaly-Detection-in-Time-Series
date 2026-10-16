"""
Training loop for the LSTM Autoencoder.

Features:
- Handles data loading and batching using DataLoader.
- Minimizes MSE reconstruction loss.
- Early stopping based on validation loss to prevent overfitting.
- Saves the best model weights based on validation performance.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import load_config
from src.data_loader import load_smd_machine
from src.preprocessing import preprocess_pipeline
from src.model import LSTMAutoencoder
from src.dataset import TimeSeriesDataset

logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion: nn.Module, 
    optimizer: torch.optim.Optimizer,
    device: torch.device
) -> float:
    """Trains the model for one epoch."""
    model.train()
    epoch_loss = 0.0
    
    for batch in dataloader:
        batch = batch.to(device)
        
        optimizer.zero_grad()
        # The target for an autoencoder is the input itself
        reconstruction = model(batch)
        loss = criterion(reconstruction, batch)
        
        loss.backward()
        # Gradient clipping to prevent exploding gradients in LSTMs
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += loss.item() * batch.size(0)
        
    return epoch_loss / len(dataloader.dataset)


def validate_epoch(
    model: nn.Module, 
    dataloader: DataLoader, 
    criterion: nn.Module,
    device: torch.device
) -> float:
    """Evaluates the model on the validation set."""
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            reconstruction = model(batch)
            loss = criterion(reconstruction, batch)
            val_loss += loss.item() * batch.size(0)
            
    return val_loss / len(dataloader.dataset)


def train_model():
    """Main training orchestrator."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    
    # 1. Configuration Setup
    config = load_config()
    machine_id = config.dataset.machine_id
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 2. Data Loading & Preprocessing
    logger.info(f"Loading data for machine: {machine_id}")
    raw_data = load_smd_machine(machine_id)
    
    logger.info("Executing preprocessing pipeline (leakage-safe)...")
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
    
    train_dataset = TimeSeriesDataset(prep_result["train_windows"])
    val_dataset = TimeSeriesDataset(prep_result["val_windows"])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.training.batch_size, 
        shuffle=True, # Shuffle training data
        drop_last=True # Drop last incomplete batch
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.training.batch_size, 
        shuffle=False
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    # 3. Model Initialization
    num_features = raw_data["num_features"]
    model = LSTMAutoencoder(
        window_size=config.preprocessing.window_size,
        n_features=num_features,
        hidden_dim=config.model.hidden_size,
        n_layers=config.model.num_layers,
        dropout=config.model.dropout
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # 4. Training Loop with Early Stopping
    logger.info(f"Starting training for up to {config.training.epochs} epochs...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    artifacts_dir = Path(__file__).parent.parent / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = artifacts_dir / f"lstm_autoencoder_{machine_id}.pt"
    scaler_save_path = artifacts_dir / f"scaler_{machine_id}.pkl"
    
    for epoch in range(1, config.training.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        logger.info(f"Epoch {epoch:03d}/{config.training.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"  -> Best model saved! (Val Loss improved)")
        else:
            patience_counter += 1
            if patience_counter >= config.training.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch} epochs.")
                break
                
    # Save the scaler so the backend can use it for inference later
    import pickle
    with open(scaler_save_path, "wb") as f:
        pickle.dump(prep_result["scaler"], f)
    logger.info(f"Saved fitted scaler to {scaler_save_path}")
    logger.info("Training complete.")


if __name__ == "__main__":
    train_model()
