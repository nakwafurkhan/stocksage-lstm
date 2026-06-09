"""Autoregressive multi-step forecasting with simple confidence bounds.

The trained model predicts the next-day scaled close from a window of the 12
scaled indicators. To forecast ``horizon`` days we feed each prediction back in:
append the predicted close to the price history, recompute the 12 indicators on
the extended series (future days approximated as O=H=L=Close with the last known
volume), re-scale, slide the window forward, and repeat.

Public API
----------
    predict(ticker, last_n_days_df, horizon=30) -> DataFrame
        columns: date, predicted_close, lower_bound, upper_bound

    predict_from_processed(ticker, horizon=30) -> DataFrame
        convenience wrapper that loads the tail of data/processed/{TICKER}.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.model import StockLSTM
from src.preprocessing import (
    FEATURE_COLS,
    PROCESSED_DIR,
    SAVED_DIR,
    SCALER_COLS,
    add_indicators,
    inverse_transform_close,
)

CONFIDENCE = 0.05  # ±5% band around the point forecast
_RAW_COLS = ["Open", "High", "Low", "Close", "Volume"]


def load_model_and_scaler(ticker: str):
    """Load the best checkpoint + scaler for a ticker.

    Raises ``FileNotFoundError`` with an actionable message if either artifact
    is missing (the dashboard surfaces this to the user).
    """
    ckpt_path = SAVED_DIR / f"{ticker}_lstm_best.pt"
    scaler_path = SAVED_DIR / f"{ticker}_scaler.pkl"
    if not ckpt_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            f"Model not found for {ticker}. Train it first with: "
            f"python src/train.py --ticker {ticker}"
        )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = StockLSTM(
        input_size=ckpt["input_size"],
        hidden_size=ckpt["hidden_size"],
        num_layers=ckpt["num_layers"],
        output_size=ckpt["output_size"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    scaler = joblib.load(scaler_path)
    return model, scaler, int(ckpt["seq_len"])


def _scale_feature_row(scaler, feat_values: np.ndarray, close_usd: float) -> np.ndarray:
    """Scale one freshly computed indicator row, returning the 12 scaled feats."""
    row = pd.DataFrame(
        [np.concatenate([feat_values, [close_usd]])], columns=SCALER_COLS
    )
    scaled = scaler.transform(row)[0]
    return scaled[: len(FEATURE_COLS)]  # FEATURE_COLS come first in SCALER_COLS


def predict(
    ticker: str,
    last_n_days_df: pd.DataFrame,
    horizon: int = 30,
    seq_len: int | None = None,
) -> pd.DataFrame:
    """Forecast ``horizon`` future closes from recent processed feature rows.

    ``seq_len`` optionally overrides the window length used per step (the LSTM
    accepts variable-length sequences); defaults to the checkpoint's seq_len.
    """
    model, scaler, ckpt_seq_len = load_model_and_scaler(ticker)
    seq_len = int(seq_len) if seq_len else ckpt_seq_len

    df = last_n_days_df.copy().reset_index(drop=True)
    missing = [c for c in _RAW_COLS + FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"input df is missing columns: {missing}")
    if len(df) < seq_len:
        raise ValueError(f"need >= {seq_len} rows of history, got {len(df)}")

    raw = df[_RAW_COLS].astype(float).copy()
    feat_window = df[FEATURE_COLS].to_numpy(dtype=np.float32)[-seq_len:]
    last_volume = float(raw["Volume"].iloc[-1])

    if "Date" in df.columns:
        last_date = pd.to_datetime(df["Date"].iloc[-1])
    else:
        last_date = pd.Timestamp.today().normalize()

    preds_usd: list[float] = []
    for _ in range(horizon):
        x = torch.from_numpy(feat_window[-seq_len:][None, ...].astype(np.float32))
        with torch.no_grad():
            scaled_next = float(model(x).numpy().reshape(-1)[0])
        close_usd = float(inverse_transform_close(scaler, [scaled_next])[0])
        preds_usd.append(close_usd)

        # Append a synthetic future bar and recompute indicators on the extended
        # series (documented approximation for unknown future O/H/L/Volume).
        raw.loc[len(raw)] = {
            "Open": close_usd, "High": close_usd, "Low": close_usd,
            "Close": close_usd, "Volume": last_volume,
        }
        feat_values = add_indicators(raw).iloc[-1][FEATURE_COLS].to_numpy(dtype=float)
        scaled_feats = _scale_feature_row(scaler, feat_values, close_usd)
        feat_window = np.vstack([feat_window, scaled_feats.astype(np.float32)])

    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1), periods=horizon
    )
    point = np.asarray(preds_usd)
    return pd.DataFrame(
        {
            "date": future_dates,
            "predicted_close": np.round(point, 2),
            "lower_bound": np.round(point * (1 - CONFIDENCE), 2),
            "upper_bound": np.round(point * (1 + CONFIDENCE), 2),
        }
    )


def predict_from_processed(
    ticker: str, horizon: int = 30, seq_len: int | None = None
) -> pd.DataFrame:
    """Load the tail of the processed CSV and forecast forward."""
    proc_path = PROCESSED_DIR / f"{ticker}.csv"
    if not proc_path.exists():
        raise FileNotFoundError(
            f"{proc_path} not found. Run "
            f"`python -m src.preprocessing --ticker {ticker}` first."
        )
    df = pd.read_csv(proc_path)
    _, _, ckpt_seq_len = load_model_and_scaler(ticker)
    window = int(seq_len) if seq_len else ckpt_seq_len
    return predict(ticker, df.tail(max(window, 60)), horizon=horizon, seq_len=window)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Forecast future closes for a ticker.")
    p.add_argument("--ticker", required=True)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--seq_len", type=int, default=None,
                   help="Override the input window length (default: model's).")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    forecast = predict_from_processed(
        a.ticker.upper(), horizon=a.horizon, seq_len=a.seq_len
    )
    print(forecast.to_string(index=False))
