"""StockSage — LSTM stock-price forecasting pipeline.

Modules
-------
data_loader    Download historical OHLCV data from Yahoo Finance (yfinance).
preprocessing  Clean data and engineer 12 technical indicators, then scale.
model          The StockLSTM PyTorch architecture.
train          Train a model for a single ticker with early stopping.
predict        Autoregressive multi-step forecasting with confidence bounds.
"""

__version__ = "1.0.0"
