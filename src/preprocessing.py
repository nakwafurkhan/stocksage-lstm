"""Clean raw OHLCV data, engineer 12 technical indicators, and scale.

Pipeline (per ticker)
---------------------
1. Load ``data/raw/{TICKER}.csv``.
2. Clean: drop mostly-null rows, forward-fill short gaps, drop split/error days.
3. Engineer exactly 12 technical indicators (see ``FEATURE_COLS``).
4. Fit a ``MinMaxScaler`` on the **first 70%** of rows only (no look-ahead),
   then transform every row.
5. Save the scaler to ``models/saved/{TICKER}_scaler.pkl`` and the processed
   frame to ``data/processed/{TICKER}.csv``.

Indicators are implemented with only ``pandas``/``numpy`` — no TA-Lib.

Usage
-----
    python -m src.preprocessing                 # all tickers
    python -m src.preprocessing --ticker AAPL   # one ticker
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = _ROOT / "data" / "raw"
PROCESSED_DIR = _ROOT / "data" / "processed"
SAVED_DIR = _ROOT / "models" / "saved"

# The 12 engineered features, in a fixed, documented order.
FEATURE_COLS = [
    "SMA_10", "SMA_50", "EMA_12", "EMA_26", "MACD", "MACD_signal",
    "RSI_14", "BB_upper", "BB_lower", "ATR_14", "OBV", "ROC_10",
]

# The scaler covers the 12 features plus the Close target (13 columns).
TARGET_COL = "Close"
SCALER_COLS = FEATURE_COLS + [TARGET_COL]
CLOSE_IDX = SCALER_COLS.index(TARGET_COL)

TRAIN_FRACTION = 0.70
MAX_DAILY_MOVE = 0.15  # drop days with > 15% single-day move (split/data error)


# ─────────────────────────────────────────────────────────────────────────────
# Indicator math
# ─────────────────────────────────────────────────────────────────────────────
def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # When there are no losses in the window, RSI is defined as 100.
    rsi = rsi.where(avg_loss != 0.0, 100.0)
    return rsi


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window).mean()


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Append the 12 technical indicators to a copy of ``df``.

    ``df`` must contain at least: High, Low, Close, Volume. The returned frame
    carries warm-up ``NaN`` rows at the top (drop them before scaling/training).
    """
    out = df.copy()
    close, high, low, volume = out["Close"], out["High"], out["Low"], out["Volume"]

    # Trend / moving averages
    out["SMA_10"] = close.rolling(10).mean()
    out["SMA_50"] = close.rolling(50).mean()
    out["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    out["EMA_26"] = close.ewm(span=26, adjust=False).mean()

    # MACD and its signal line
    out["MACD"] = out["EMA_12"] - out["EMA_26"]
    out["MACD_signal"] = out["MACD"].ewm(span=9, adjust=False).mean()

    # Momentum
    out["RSI_14"] = _rsi(close, 14)
    out["ROC_10"] = (close / close.shift(10) - 1.0) * 100.0

    # Volatility — Bollinger Bands (20, 2σ)
    sma_20 = close.rolling(20).mean()
    std_20 = close.rolling(20).std()
    out["BB_upper"] = sma_20 + 2.0 * std_20
    out["BB_lower"] = sma_20 - 2.0 * std_20

    # Volatility / volume
    out["ATR_14"] = _atr(high, low, close, 14)
    out["OBV"] = _obv(close, volume)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Cleaning
# ─────────────────────────────────────────────────────────────────────────────
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the documented cleaning steps to raw OHLCV data."""
    out = df.copy()
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"])
        out = out.sort_values("Date").reset_index(drop=True)

    numeric = ["Open", "High", "Low", "Close", "Volume"]
    out[numeric] = out[numeric].apply(pd.to_numeric, errors="coerce")

    # 1. Drop rows where more than 10% of columns are null.
    null_frac = out[numeric].isna().mean(axis=1)
    out = out[null_frac <= 0.10].reset_index(drop=True)

    # 2. Forward-fill gaps of up to 3 consecutive NaN days.
    out[numeric] = out[numeric].ffill(limit=3)
    out = out.dropna(subset=numeric).reset_index(drop=True)

    # 3. Flag IQR outliers on Close (for logging) and remove >15% single-day
    #    moves, which almost always indicate an unadjusted split or a data error.
    q1, q3 = out["Close"].quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    n_iqr = int(((out["Close"] < lo) | (out["Close"] > hi)).sum())

    daily_move = out["Close"].pct_change().abs()
    keep = (daily_move <= MAX_DAILY_MOVE) | daily_move.isna()
    n_jumps = int((~keep).sum())
    out = out[keep].reset_index(drop=True)

    out.attrs["n_iqr_flagged"] = n_iqr
    out.attrs["n_jump_removed"] = n_jumps
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Scaling helpers
# ─────────────────────────────────────────────────────────────────────────────
def inverse_transform_close(scaler: MinMaxScaler, scaled_close: np.ndarray) -> np.ndarray:
    """Map scaled Close values back to real dollars using a fitted scaler.

    The scaler was fit on ``SCALER_COLS`` (13 columns); we reconstruct a dummy
    matrix, drop the prediction into the Close column, and inverse-transform.
    """
    scaled_close = np.asarray(scaled_close, dtype=float).reshape(-1)
    dummy = np.zeros((scaled_close.shape[0], len(SCALER_COLS)))
    dummy[:, CLOSE_IDX] = scaled_close
    return scaler.inverse_transform(dummy)[:, CLOSE_IDX]


# ─────────────────────────────────────────────────────────────────────────────
# Per-ticker driver
# ─────────────────────────────────────────────────────────────────────────────
def process_ticker(ticker: str, save: bool = True) -> pd.DataFrame:
    """Run the full pipeline for one ticker and (optionally) persist outputs."""
    raw_path = RAW_DIR / f"{ticker}.csv"
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} not found. Run `python -m src.data_loader` first."
        )

    df = pd.read_csv(raw_path)
    df = clean(df)
    df = add_indicators(df).dropna().reset_index(drop=True)
    if len(df) < 100:
        raise ValueError(f"{ticker}: only {len(df)} usable rows after cleaning.")

    # Fit the scaler on the first 70% of rows ONLY (prevents look-ahead leakage).
    n_train = int(len(df) * TRAIN_FRACTION)
    scaler = MinMaxScaler()
    scaler.fit(df.loc[: n_train - 1, SCALER_COLS])
    scaled = scaler.transform(df[SCALER_COLS])
    scaled = pd.DataFrame(scaled, columns=SCALER_COLS, index=df.index)

    # Keep raw OHLCV alongside the scaled features. Training ignores the extra
    # columns, but inference needs them to recompute indicators step-by-step
    # during autoregressive forecasting.
    processed = pd.DataFrame({"Date": df["Date"]})
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        processed[col] = df[col].to_numpy()
    for col in FEATURE_COLS:
        processed[col] = scaled[col]
    processed["Close_scaled"] = scaled[TARGET_COL]

    if save:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        SAVED_DIR.mkdir(parents=True, exist_ok=True)
        processed.to_csv(PROCESSED_DIR / f"{ticker}.csv", index=False)
        joblib.dump(scaler, SAVED_DIR / f"{ticker}_scaler.pkl")

    print(
        f"  {ticker:<6} rows={len(processed):>4}  train_rows={n_train:>4}  "
        f"iqr_flagged={df.attrs.get('n_iqr_flagged', 0)}  "
        f"jumps_removed={df.attrs.get('n_jump_removed', 0)}"
    )
    return processed


def process_all(tickers: list[str] | None = None) -> None:
    from src.data_loader import TICKERS  # local import avoids a hard cycle

    tickers = tickers or TICKERS
    print(f"Preprocessing {len(tickers)} tickers -> {PROCESSED_DIR}")
    for ticker in tickers:
        try:
            process_ticker(ticker)
        except Exception as exc:  # noqa: BLE001
            print(f"  {ticker:<6} SKIPPED ({exc})")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean + engineer features for tickers.")
    p.add_argument("--ticker", help="Process a single ticker (default: all).")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.ticker:
        process_ticker(args.ticker.upper())
    else:
        process_all()
