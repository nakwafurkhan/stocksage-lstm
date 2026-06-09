"""StockSage — interactive Plotly Dash dashboard (dark theme, no auth).

Publicly accessible. Pick a ticker, tune the input window and forecast horizon
with the sliders, and hit *Run Prediction* to call the trained LSTM.

Run locally:
    python app/dashboard.py        # -> http://localhost:8050

On Render the server binds to $PORT automatically (see Procfile).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html

# Make `src` importable when launched as `python app/dashboard.py`.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.predict import predict  # noqa: E402
from src.preprocessing import PROCESSED_DIR, SAVED_DIR  # noqa: E402

# ── Theme constants (kept in sync with assets/custom.css) ────────────────
NAVY = "#050B18"
PANEL = "#0A1628"
CYAN = "#00D4FF"
GOLD = "#FFB547"
GREEN = "#00FF88"
RED = "#FF5C7A"

TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "SPY"]
HISTORY_DAYS = 180

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="StockSage — ML Price Predictor",
    update_title=None,
)
server = app.server  # exposed for gunicorn / WSGI hosts


# ── Data helpers ──────────────────────────────────────────────────────────
def load_history(ticker: str) -> pd.DataFrame | None:
    path = PROCESSED_DIR / f"{ticker}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["Date"])
    return df


def load_mae(ticker: str) -> str:
    path = SAVED_DIR / f"{ticker}_metrics.json"
    if not path.exists():
        return "—"
    try:
        return f"${json.load(open(path))['mae']:,.2f}"
    except Exception:  # noqa: BLE001
        return "—"


def _base_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PANEL,
        title=dict(text=title, font=dict(size=18, color="#E6F1FF")),
        font=dict(family="JetBrains Mono, monospace", color="#9FB3C8"),
        margin=dict(l=50, r=30, t=60, b=40),
        legend=dict(orientation="h", y=1.02, x=0, bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="#16263a", zeroline=False),
        yaxis=dict(gridcolor="#16263a", zeroline=False, tickprefix="$"),
        hovermode="x unified",
    )
    return fig


def placeholder_figure(msg: str = "Select options, then click Run Prediction") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False,
                       font=dict(size=16, color="#5b7088"), x=0.5, y=0.5, xref="paper", yref="paper")
    return _base_layout(fig, "StockSage forecast")


def build_figure(ticker: str, hist: pd.DataFrame, pred: pd.DataFrame | None) -> go.Figure:
    fig = go.Figure()
    h = hist.tail(HISTORY_DAYS)
    fig.add_trace(go.Scatter(
        x=h["Date"], y=h["Close"], name="Historical",
        mode="lines", line=dict(color=CYAN, width=2),
    ))

    if pred is not None and not pred.empty:
        # Connect the forecast to the last observed close for visual continuity.
        anchor_date, anchor_close = h["Date"].iloc[-1], h["Close"].iloc[-1]
        px = [anchor_date, *pred["date"]]
        upper = [anchor_close, *pred["upper_bound"]]
        lower = [anchor_close, *pred["lower_bound"]]
        mid = [anchor_close, *pred["predicted_close"]]

        fig.add_trace(go.Scatter(x=px, y=upper, mode="lines",
                                 line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=px, y=lower, mode="lines", line=dict(width=0),
                                 fill="tonexty", fillcolor="rgba(255,181,71,0.15)",
                                 name="±5% band", hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=px, y=mid, name="Predicted", mode="lines",
                                 line=dict(color=GOLD, width=2, dash="dash")))

    return _base_layout(fig, f"{ticker} — price & forecast")


# ── UI components ──────────────────────────────────────────────────────────
def metric_card(card_id: str, label: str) -> dbc.Col:
    return dbc.Col(
        html.Div(
            [html.Div(label, className="metric-label"),
             html.Div("—", id=card_id, className="metric-value")],
            className="metric-card",
        ),
        xs=6, md=3,
    )


controls = html.Div(
    [
        html.Div([
            html.Span("⚡", className="logo-mark"),
            html.Span("StockSage", className="logo-text"),
        ], className="brand"),
        html.Div([
            html.Label("Ticker", className="ctl-label"),
            dcc.Dropdown(id="ticker-dd", options=[{"label": t, "value": t} for t in TICKERS],
                         value="AAPL", clearable=False, className="ticker-dd"),
        ], className="ctl"),
        html.Div([
            html.Label("Sequence length (days)", className="ctl-label"),
            dcc.Slider(id="seq-slider", min=30, max=90, value=60, step=None,
                       marks={30: "30", 45: "45", 60: "60", 90: "90"}),
        ], className="ctl ctl-slider"),
        html.Div([
            html.Label("Horizon (days)", className="ctl-label"),
            dcc.Slider(id="horizon-slider", min=7, max=30, value=30, step=None,
                       marks={7: "7", 14: "14", 30: "30"}),
        ], className="ctl ctl-slider"),
        dbc.Button("Run Prediction", id="run-btn", n_clicks=0, className="run-btn"),
    ],
    className="control-bar",
)

app.layout = dbc.Container(
    [
        controls,
        html.Div(id="status"),
        dcc.Loading(
            dcc.Graph(id="price-chart", figure=placeholder_figure(), config={"displayModeBar": False}),
            type="dot", color=CYAN,
        ),
        dbc.Row(
            [
                metric_card("m-mae", "Test MAE"),
                metric_card("m-last", "Last close"),
                metric_card("m-pred", "Forecast (end)"),
                metric_card("m-change", "% change"),
            ],
            className="metrics-row g-3",
        ),
        html.Div(
            "Educational project — not financial advice.",
            className="footer-note",
        ),
    ],
    fluid=True,
    className="app-shell",
)


# ── Callback ───────────────────────────────────────────────────────────────
@app.callback(
    Output("price-chart", "figure"),
    Output("m-mae", "children"),
    Output("m-last", "children"),
    Output("m-pred", "children"),
    Output("m-change", "children"),
    Output("status", "children"),
    Input("run-btn", "n_clicks"),
    State("ticker-dd", "value"),
    State("seq-slider", "value"),
    State("horizon-slider", "value"),
    prevent_initial_call=True,
)
def run_prediction(_clicks, ticker, seq_len, horizon):
    hist = load_history(ticker)
    if hist is None:
        msg = (f"No processed data for {ticker}. Run: "
               f"python -m src.preprocessing --ticker {ticker}")
        return placeholder_figure(msg), "—", "—", "—", "—", dbc.Alert(msg, color="warning")

    last_close = float(hist["Close"].iloc[-1])
    try:
        pred = predict(ticker, hist.tail(max(int(seq_len), 90)), horizon=int(horizon), seq_len=int(seq_len))
    except FileNotFoundError as exc:
        # Model not trained yet — show history and the exact command to fix it.
        fig = build_figure(ticker, hist, None)
        return (fig, load_mae(ticker), f"${last_close:,.2f}", "—", "—",
                dbc.Alert(str(exc), color="warning", className="status-alert"))

    fig = build_figure(ticker, hist, pred)
    end_price = float(pred["predicted_close"].iloc[-1])
    pct = (end_price / last_close - 1.0) * 100.0
    pct_txt = html.Span(f"{pct:+.2f}%", style={"color": GREEN if pct >= 0 else RED})

    ok = dbc.Alert(
        f"Forecast for {ticker}: {horizon} business days ahead "
        f"(window {seq_len} days).",
        color="info", className="status-alert",
    )
    return (fig, load_mae(ticker), f"${last_close:,.2f}",
            f"${end_price:,.2f}", pct_txt, ok)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(host="0.0.0.0", port=port, debug=False)
