"""Train a StockLSTM for one ticker, with early stopping and $-denominated MAE.

Example
-------
    python src/train.py --ticker AAPL --seq_len 60 --epochs 100 --output_size 1

Artifacts written to ``models/saved/``:
    {TICKER}_lstm_best.pt   best checkpoint (weights + config)
    {TICKER}_metrics.json   {"ticker", "mae", "test_samples", "epochs_trained", ...}
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from src.model import StockLSTM
from src.preprocessing import (
    FEATURE_COLS,
    PROCESSED_DIR,
    SAVED_DIR,
    inverse_transform_close,
)

# ── Fixed training configuration (matches the project spec) ──────────────
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
SCHED_PATIENCE = 5
SCHED_FACTOR = 0.5
EARLY_STOP_PATIENCE = 10
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15  # remaining 0.15 is the test set

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_sequences(
    df: pd.DataFrame, seq_len: int, output_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding window: seq_len days of features -> next ``output_size`` closes.

    X: (samples, seq_len, 12)   y: (samples, output_size)
    """
    features = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    target = df["Close_scaled"].to_numpy(dtype=np.float32)

    x_list, y_list = [], []
    last = len(df) - seq_len - output_size + 1
    for i in range(last):
        x_list.append(features[i : i + seq_len])
        y_list.append(target[i + seq_len : i + seq_len + output_size])
    if not x_list:
        raise ValueError("Not enough rows to build a single sequence.")
    return np.stack(x_list), np.stack(y_list)


def time_split(n: int) -> tuple[slice, slice, slice]:
    """Time-ordered 70/15/15 split (no shuffling)."""
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n)


def _loader(x: np.ndarray, y: np.ndarray, shuffle: bool = False) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


def _epoch_loss(model, loader, criterion, optimizer=None) -> float:
    """Run one epoch. If ``optimizer`` is given, train; otherwise evaluate."""
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            preds = model(xb)
            loss = criterion(preds, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * xb.size(0)
            count += xb.size(0)
    return total / max(count, 1)


def train(
    ticker: str,
    seq_len: int = 60,
    epochs: int = 100,
    output_size: int = 1,
) -> dict:
    set_seed()
    proc_path = PROCESSED_DIR / f"{ticker}.csv"
    scaler_path = SAVED_DIR / f"{ticker}_scaler.pkl"
    if not proc_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(
            f"Missing processed data/scaler for {ticker}. "
            f"Run `python -m src.preprocessing --ticker {ticker}` first."
        )

    df = pd.read_csv(proc_path)
    scaler = joblib.load(scaler_path)

    X, y = build_sequences(df, seq_len, output_size)
    tr, va, te = time_split(len(X))
    print(
        f"[{ticker}] sequences={len(X)}  "
        f"train={tr.stop - tr.start} val={va.stop - va.start} test={te.stop - te.start} "
        f"(seq_len={seq_len}, output_size={output_size}, device={DEVICE})"
    )

    train_loader = _loader(X[tr], y[tr])
    val_loader = _loader(X[va], y[va])

    model = StockLSTM(input_size=len(FEATURE_COLS), output_size=output_size).to(DEVICE)
    model.summary()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, patience=SCHED_PATIENCE, factor=SCHED_FACTOR)

    SAVED_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = SAVED_DIR / f"{ticker}_lstm_best.pt"

    best_val = float("inf")
    best_epoch = 0
    epochs_run = 0
    stale = 0

    for epoch in range(1, epochs + 1):
        epochs_run = epoch
        train_loss = _epoch_loss(model, train_loader, criterion, optimizer)
        val_loss = _epoch_loss(model, val_loader, criterion)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val, best_epoch, stale = val_loss, epoch, 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_size": len(FEATURE_COLS),
                    "hidden_size": model.hidden_size,
                    "num_layers": model.num_layers,
                    "output_size": output_size,
                    "seq_len": seq_len,
                    "ticker": ticker,
                },
                ckpt_path,
            )
        else:
            stale += 1

        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:>3}/{epochs}  "
                  f"train={train_loss:.6f}  val={val_loss:.6f}  best={best_val:.6f}")

        if stale >= EARLY_STOP_PATIENCE:
            print(f"  early stop at epoch {epoch} (no val improvement for "
                  f"{EARLY_STOP_PATIENCE} epochs; best @ {best_epoch}).")
            break

    # ── Evaluate the best checkpoint on the held-out test set ────────────
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X[te]).to(DEVICE)).cpu().numpy()
    actual = y[te]

    pred_usd = inverse_transform_close(scaler, preds.reshape(-1))
    true_usd = inverse_transform_close(scaler, actual.reshape(-1))
    mae = float(np.mean(np.abs(pred_usd - true_usd)))
    rmse = float(np.sqrt(np.mean((pred_usd - true_usd) ** 2)))

    metrics = {
        "ticker": ticker,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "test_samples": int(te.stop - te.start),
        "epochs_trained": epochs_run,
        "best_epoch": best_epoch,
        "seq_len": seq_len,
        "output_size": output_size,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(SAVED_DIR / f"{ticker}_metrics.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(f"[{ticker}] TEST MAE = ${mae:,.2f}  RMSE = ${rmse:,.2f}  "
          f"(best epoch {best_epoch}) -> {ckpt_path.name}")
    return metrics


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a StockLSTM for one ticker.")
    p.add_argument("--ticker", required=True, help="e.g. AAPL")
    p.add_argument("--seq_len", type=int, default=60)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--output_size", type=int, default=1,
                   help="1 = next-day close; 30 = multi-step head")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse_args()
    train(a.ticker.upper(), seq_len=a.seq_len, epochs=a.epochs, output_size=a.output_size)
