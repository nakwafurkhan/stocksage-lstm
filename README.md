# ⚡ StockSage — ML Price Predictor

[![Python](https://img.shields.io/badge/python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-22c55e?style=flat-square)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/nakwafurkhan/stocksage-lstm?style=flat-square&color=00D4FF)](https://github.com/nakwafurkhan/stocksage-lstm/stargazers)

> An LSTM-based stock-price forecaster trained on 5 years of Yahoo Finance data.
> It predicts future S&P 500 closing prices from 12 engineered technical
> indicators and serves them through an interactive Plotly Dash dashboard.
> Held-out accuracy is reported as mean absolute error (MAE) **in real dollars** —
> see [Results](#results) once you've trained a ticker.

---

## Live demo

The project ships two front-ends:

| Surface | What it is | URL |
|---------|-----------|-----|
| 🖥️ **Dashboard** (Plotly Dash) | Interactive forecasting app, deployable to Render | _add your `https://stocksage-xxxxx.onrender.com` URL after deploying_ |
| 🌐 **Landing page** (static) | Animated project overview, served by GitHub Pages | `https://nakwafurkhan.github.io/stocksage-lstm/` _(live once Pages is enabled)_ |

See [Deployment](#deployment) for both.

---

## What it does

StockSage learns short-term price dynamics from historical market data. For each
ticker it downloads five years of daily OHLCV bars from Yahoo Finance, engineers
twelve classic technical indicators (moving averages, MACD, RSI, Bollinger Bands,
ATR, OBV, and more), and trains a two-layer LSTM to predict the next closing
price from a sliding window of recent days.

At inference time the model rolls forward autoregressively to produce a multi-day
forecast with a simple ±5% confidence band. Everything is wrapped in a dark-themed
Plotly Dash dashboard where you can switch tickers and tune the input window and
forecast horizon with sliders — no code required. **This is an educational project,
not financial advice.**

---

## Architecture

```text
 yfinance API
     │  src/data_loader.py        download 5y daily OHLCV
     ▼
 data/raw/*.csv
     │  src/preprocessing.py      clean + 12 indicators + MinMaxScaler
     ▼
 data/processed/*.csv  +  models/saved/<TICKER>_scaler.pkl
     │  src/train.py              StockLSTM (2-layer, hidden=128), early stopping
     ▼
 models/saved/<TICKER>_lstm_best.pt  +  <TICKER>_metrics.json
     │  src/predict.py            autoregressive 30-day forecast + ±5% band
     ▼
 app/dashboard.py                 Plotly Dash  →  deploy on Render
```

---

## Results

Test-set MAE is the mean absolute error between the predicted and true closing
price on the **last 15%** of each ticker's history (the model never sees it during
training). The table below is generated from `models/saved/<TICKER>_metrics.json`.

<!-- RESULTS:START -->
| Ticker | MAE ($) | Test samples | Epochs trained |
|--------|---------|--------------|----------------|
| AAPL   | —       | —            | —              |
| MSFT   | —       | —            | —              |
| SPY    | —       | —            | —              |
| GOOGL  | —       | —            | —              |
<!-- RESULTS:END -->

> The dashes are placeholders. After training, run
> `python scripts/update_readme_results.py` to fill this table in automatically
> from your real metrics files.

---

## Quick start

```bash
# 1 · clone
git clone https://github.com/nakwafurkhan/stocksage-lstm.git
cd stocksage-lstm

# 2 · install pinned dependencies (use a fresh virtualenv)
pip install -r requirements.txt

# 3 · download data and build features for all tickers
python -m src.data_loader
python -m src.preprocessing

# 4 · train a ticker (writes a checkpoint + metrics.json)
python src/train.py --ticker AAPL --seq_len 60 --epochs 100 --output_size 1

# 5 · launch the dashboard  →  http://localhost:8050
python app/dashboard.py
```

---

## Project structure

```text
stocksage-lstm/
├── data/
│   ├── raw/                  # CSVs downloaded from yfinance (git-ignored)
│   └── processed/            # cleaned, feature-engineered CSVs (git-ignored)
├── models/
│   └── saved/                # .pt checkpoints + .pkl scalers + .json metrics
├── notebooks/
│   └── exploration.ipynb     # EDA and prototyping
├── src/
│   ├── data_loader.py        # download OHLCV from Yahoo Finance
│   ├── preprocessing.py      # cleaning + 12 indicators + scaling
│   ├── model.py              # StockLSTM architecture
│   ├── train.py              # training loop, early stopping, MAE eval
│   └── predict.py            # autoregressive multi-step inference
├── app/
│   ├── dashboard.py          # Plotly Dash app (dark theme, no login)
│   └── assets/custom.css     # dashboard styling
├── scripts/
│   └── update_readme_results.py  # regenerate the Results table from metrics
├── index.html                # animated landing page (GitHub Pages)
├── requirements.txt
├── Procfile                  # Render start command
├── README.md
└── LICENSE
```

---

## The 12 technical indicators

All indicators are implemented with **only pandas/NumPy** (no TA-Lib).

| Feature | Formula | Why it's useful |
|---------|---------|-----------------|
| `SMA_10`  | 10-day mean of Close | Short-term trend |
| `SMA_50`  | 50-day mean of Close | Medium-term trend |
| `EMA_12`  | 12-day exponential MA | Faster-reacting trend |
| `EMA_26`  | 26-day exponential MA | Slower trend baseline |
| `MACD`    | EMA_12 − EMA_26 | Trend momentum |
| `MACD_signal` | 9-day EMA of MACD | Momentum crossovers |
| `RSI_14`  | 100 − 100 / (1 + avg gain / avg loss), 14d | Overbought / oversold |
| `BB_upper` | SMA_20 + 2·σ_20 | Upper volatility envelope |
| `BB_lower` | SMA_20 − 2·σ_20 | Lower volatility envelope |
| `ATR_14`  | 14-day mean of True Range | Absolute volatility |
| `OBV`     | Cumulative sign(ΔClose)·Volume | Volume-confirmed pressure |
| `ROC_10`  | (Close / Close₋₁₀ − 1) · 100 | 10-day momentum |

---

## Tech stack

| Tool | Version | Purpose |
|------|---------|---------|
| PyTorch | 2.2.0 | LSTM model + training |
| NumPy | 1.26.4 | Numerics |
| pandas | 2.1.4 | Data wrangling + indicators |
| scikit-learn | 1.4.0 | MinMaxScaler |
| joblib | 1.3.2 | Scaler persistence |
| yfinance | 0.2.38 | Yahoo Finance data |
| Dash | 2.16.1 | Dashboard framework |
| dash-bootstrap-components | 1.5.0 | Layout + theming |
| Plotly | 5.20.0 | Charts |
| gunicorn | 21.2.0 | Production WSGI (optional) |

---

## Data source

Historical data comes from **Yahoo Finance** via the
[`yfinance`](https://pypi.org/project/yfinance/) Python library — **no API key
required**. With `auto_adjust=True`, prices are split- and dividend-adjusted.

```bash
python -m src.data_loader          # downloads all 10 tickers into data/raw/
```

**Size note:** five years of *daily* OHLCV is compact — roughly 1,250 rows (~100 KB)
per ticker, so all 10 tickers together are only a **few megabytes**. (You would only
reach multi-GB volumes with intraday/tick data, which this project does not use.)

Default tickers: `AAPL, MSFT, GOOGL, AMZN, SPY, NVDA, TSLA, META, JPM, NFLX`.

---

## Deployment

### Dashboard → Render

1. Push this repo to GitHub.
2. On [render.com](https://render.com), create **New → Web Service** and connect the repo.
3. Settings:
   - **Environment:** Python 3
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `python app/dashboard.py`
4. Render injects a `PORT` env var; `dashboard.py` already binds to it (see `Procfile`).
5. After the build succeeds, copy the public `https://stocksage-xxxxx.onrender.com`
   URL into the [Live demo](#live-demo) table above and the repo description.

> ⚠️ Commit trained `.pt`/`.pkl` files (or run training in the build) if you want the
> deployed dashboard to serve predictions — they are git-ignored by default.

### Landing page → GitHub Pages

1. **Settings → Pages → Source:** *Deploy from a branch* → `main` → `/ (root)`.
2. GitHub serves `index.html` at `https://nakwafurkhan.github.io/stocksage-lstm/`.

---

## Disclaimer

⚠️ **Educational purposes only. Not financial advice.** Past performance and model
forecasts do not predict future results. Do not trade on these outputs.

---

## License

Released under the [MIT License](LICENSE).
