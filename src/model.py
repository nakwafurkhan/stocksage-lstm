"""The StockSage LSTM architecture.

A straightforward, well-regularised recurrent regressor: a stacked LSTM whose
final hidden state is mapped to ``output_size`` future closing prices by a
single linear layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StockLSTM(nn.Module):
    """Stacked LSTM for next-step (or multi-step) close-price regression.

    Parameters
    ----------
    input_size:   Number of features per timestep (12 technical indicators).
    hidden_size:  LSTM hidden units per layer.
    num_layers:   Number of stacked LSTM layers.
    dropout:      Dropout applied between LSTM layers (ignored if num_layers==1).
    output_size:  Forecast horizon. 1 = next-day close, 30 = multi-step head.
    """

    def __init__(
        self,
        input_size: int = 12,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_size: int = 1,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, input_size) -> (batch, output_size)."""
        out, _ = self.lstm(x)          # (batch, seq_len, hidden_size)
        last_step = out[:, -1, :]      # (batch, hidden_size) — final timestep
        return self.fc(last_step)      # (batch, output_size)

    def summary(self) -> int:
        """Print and return the total parameter count."""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(
            "StockLSTM("
            f"input_size={self.input_size}, hidden_size={self.hidden_size}, "
            f"num_layers={self.num_layers}, output_size={self.output_size})"
        )
        print(f"  parameters: {total:,} total | {trainable:,} trainable")
        return total


if __name__ == "__main__":
    model = StockLSTM()
    model.summary()
    # Quick shape sanity check: batch=4, seq_len=60, features=12.
    dummy = torch.randn(4, 60, 12)
    print("  forward output shape:", tuple(model(dummy).shape))
