"""
LSTM Autoencoder PyTorch Architecture.

Implements the LSTM-based Encoder-Decoder architecture for 
Multi-sensor Anomaly Detection, grounded in the ICML 2016 paper.

Architecture:
- Encoder: Reads the input sequence (window_size, num_features) and 
  compresses it into a fixed-length hidden representation.
- Decoder: Reconstructs the input sequence from the hidden representation.
- Objective: Minimize the reconstruction error (MSE) between the input 
  window and the reconstructed window.
"""

from __future__ import annotations

import torch
import torch.nn as nn

class LSTMEncoder(nn.Module):
    def __init__(self, n_features: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sequence of shape (batch_size, window_size, n_features)
        Returns:
            hidden: The final hidden state of the LSTM of shape (batch_size, hidden_dim)
        """
        # _ is the outputs at each step
        # hidden consists of (h_n, c_n), each of shape (n_layers, batch_size, hidden_dim)
        _, (h_n, _) = self.lstm(x)
        
        # Take the hidden state from the last layer
        return h_n[-1, :, :]

class LSTMDecoder(nn.Module):
    def __init__(self, window_size: int, n_features: int, hidden_dim: int, n_layers: int, dropout: float):
        super().__init__()
        self.window_size = window_size
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0
        )
        
        self.output_layer = nn.Linear(hidden_dim, n_features)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: The hidden state from the encoder of shape (batch_size, hidden_dim)
        Returns:
            reconstruction: The reconstructed sequence (batch_size, window_size, n_features)
        """
        # Repeat the hidden state across the window size dimension
        # Shape becomes (batch_size, window_size, hidden_dim)
        x_repeated = x.unsqueeze(1).repeat(1, self.window_size, 1)
        
        # Pass through decoder LSTM
        lstm_out, _ = self.lstm(x_repeated)
        
        # Pass through linear layer to get original features
        reconstruction = self.output_layer(lstm_out)
        
        return reconstruction

class LSTMAutoencoder(nn.Module):
    """
    Full LSTM Encoder-Decoder model.
    """
    def __init__(
        self, 
        window_size: int, 
        n_features: int, 
        hidden_dim: int = 64, 
        n_layers: int = 2, 
        dropout: float = 0.2
    ):
        super().__init__()
        self.encoder = LSTMEncoder(
            n_features=n_features, 
            hidden_dim=hidden_dim, 
            n_layers=n_layers, 
            dropout=dropout
        )
        self.decoder = LSTMDecoder(
            window_size=window_size, 
            n_features=n_features, 
            hidden_dim=hidden_dim, 
            n_layers=n_layers, 
            dropout=dropout
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sequence of shape (batch_size, window_size, n_features)
        Returns:
            Reconstructed sequence of shape (batch_size, window_size, n_features)
        """
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

if __name__ == "__main__":
    # Quick sanity check for the model architecture
    batch_size = 32
    window_size = 30
    n_features = 38
    
    model = LSTMAutoencoder(
        window_size=window_size, 
        n_features=n_features, 
        hidden_dim=64, 
        n_layers=2
    )
    
    # Dummy input
    x = torch.randn(batch_size, window_size, n_features)
    output = model(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    
    assert x.shape == output.shape, "Output shape must match input shape for Autoencoder"
    print("Architecture verified successfully.")
