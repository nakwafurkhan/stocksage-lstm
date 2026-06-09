"""Download historical OHLCV data from Yahoo Finance via ``yfinance``.

Data source
-----------
`yfinance` pulls free historical Open/High/Low/Close/Volume data directly from
Yahoo Finance. No API key is required. With ``auto_adjust=True`` the prices are
split- and dividend-adjusted.

Size note
---------
Five years of **daily** OHLCV is small: roughly 1,250 rows per ticker, so each
CSV is ~100 KB and all 10 tickers together are only a few megabytes — not
gigabytes. (You would only approach multi-GB sizes with intraday/tick data.)

Usage
-----
As a script (downloads every ticker in ``TICKERS``)::

    python -m src.data_loader

As a module::

    from src.data_loader import download_all
    download_all(["AAPL", "MSFT"])
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

# Resolve <repo>/data/raw regardless of where the script is invoked from.
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# Ten liquid, large-cap S&P 500 names spanning several sectors.
TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "SPY",
    "NVDA", "TSLA", "META", "JPM", "NFLX",
]

OHLCV_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance may return a MultiIndex column frame for a single ticker."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def download_ticker(
    ticker: str,
    period: str = "5y",
    interval: str = "1d",
    out_dir: Path = RAW_DIR,
) -> Path:
    """Download one ticker and write ``data/raw/{TICKER}.csv``.

    Returns the path to the written CSV. Raises ``ValueError`` if Yahoo returns
    no rows for the symbol.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    df = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if df is None or df.empty:
        raise ValueError(f"no data returned for {ticker!r}")

    df = _flatten_columns(df).reset_index()

    # Keep a stable, documented column order.
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{ticker}: missing expected columns {missing}")
    df = df[OHLCV_COLUMNS]

    out_path = out_dir / f"{ticker}.csv"
    df.to_csv(out_path, index=False)
    return out_path


def download_all(
    tickers: list[str] | None = None,
    period: str = "5y",
    interval: str = "1d",
    out_dir: Path = RAW_DIR,
) -> dict[str, str]:
    """Download every ticker, logging per-ticker success/failure.

    Returns a mapping of ``ticker -> "ok" | "error: ..."`` so callers can react
    programmatically instead of parsing stdout.
    """
    tickers = tickers or TICKERS
    results: dict[str, str] = {}

    print(f"[{_timestamp()}] Downloading {len(tickers)} tickers "
          f"(period={period}, interval={interval}) -> {out_dir}")

    for ticker in tickers:
        try:
            path = download_ticker(ticker, period, interval, out_dir)
            rows = sum(1 for _ in open(path)) - 1  # minus header
            results[ticker] = "ok"
            print(f"[{_timestamp()}]  OK   {ticker:<6} {rows:>5} rows -> {path}")
        except Exception as exc:  # noqa: BLE001 — log and continue per ticker
            results[ticker] = f"error: {exc}"
            print(f"[{_timestamp()}]  FAIL {ticker:<6} {exc}")

    ok = sum(1 for v in results.values() if v == "ok")
    print(f"[{_timestamp()}] Done: {ok}/{len(tickers)} succeeded.")
    return results


if __name__ == "__main__":
    outcome = download_all()
    # Non-zero exit if every download failed (useful in CI / cron).
    if all(v != "ok" for v in outcome.values()):
        sys.exit(1)
