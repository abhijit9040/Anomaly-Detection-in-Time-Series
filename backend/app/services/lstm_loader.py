"""
LSTM model loader helper for the backend.
Mirrors the training/src/model.py architecture but lives inside the backend package.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return h_n[-1, :, :]


class LSTMDecoder(nn.Module):
    def __init__(self, window_size: int, n_features: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.window_size = window_size
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.output_layer = nn.Linear(hidden_dim, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_repeated = x.unsqueeze(1).repeat(1, self.window_size, 1)
        lstm_out, _ = self.lstm(x_repeated)
        return self.output_layer(lstm_out)


class LSTMAutoencoder(nn.Module):
    def __init__(self, window_size: int, n_features: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.encoder = LSTMEncoder(n_features, hidden_dim, n_layers, dropout)
        self.decoder = LSTMDecoder(window_size, n_features, hidden_dim, n_layers, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def build_lstm_autoencoder(
    window_size: int = 30,
    n_features: int = 38,
    hidden_dim: int = 64,
    n_layers: int = 2,
    dropout: float = 0.1,
) -> LSTMAutoencoder:
    """Build the LSTM autoencoder with default SMD hyperparameters."""
    return LSTMAutoencoder(
        window_size=window_size,
        n_features=n_features,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        dropout=dropout,
    )
