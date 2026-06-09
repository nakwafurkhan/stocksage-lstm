"""Regenerate the README Results table from real training metrics.

Scans ``models/saved/*_metrics.json`` and rewrites the markdown table between
the ``<!-- RESULTS:START -->`` / ``<!-- RESULTS:END -->`` markers in README.md.
Run it after training one or more tickers:

    python scripts/update_readme_results.py

This keeps the README honest: numbers come straight from the JSON your training
runs produced — never hand-edited.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / "models" / "saved"
README = ROOT / "README.md"

START = "<!-- RESULTS:START -->"
END = "<!-- RESULTS:END -->"

# Canonical rows shown even before training (as dashes), in this order.
DEFAULT_ORDER = ["AAPL", "MSFT", "SPY", "GOOGL"]


def load_metrics() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for path in sorted(SAVED.glob("*_metrics.json")):
        try:
            data = json.loads(path.read_text())
            out[data.get("ticker", path.stem.split("_")[0])] = data
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  ! skipping {path.name}: {exc}")
    return out


def build_table(metrics: dict[str, dict]) -> str:
    # Show the default tickers first, then any extra trained tickers.
    tickers = DEFAULT_ORDER + [t for t in sorted(metrics) if t not in DEFAULT_ORDER]
    best = min(metrics.values(), key=lambda m: m.get("mae", float("inf")), default=None)

    rows = [
        "| Ticker | MAE ($) | Test samples | Epochs trained |",
        "|--------|---------|--------------|----------------|",
    ]
    for t in tickers:
        m = metrics.get(t)
        if not m:
            rows.append(f"| {t:<6} | —       | —            | —              |")
            continue
        star = " ★" if best and m is best else ""
        rows.append(
            f"| {t:<6} | {m['mae']:.2f}{star} | {m['test_samples']} | {m['epochs_trained']} |"
        )
    return "\n".join(rows)


def main() -> None:
    metrics = load_metrics()
    if not metrics:
        print("No *_metrics.json found in models/saved/. Train a ticker first:")
        print("  python src/train.py --ticker AAPL")
        return

    text = README.read_text()
    table = build_table(metrics)
    replacement = f"{START}\n{table}\n{END}"
    new_text, n = re.subn(
        re.escape(START) + r".*?" + re.escape(END),
        replacement,
        text,
        count=1,
        flags=re.DOTALL,
    )
    if n == 0:
        raise SystemExit("Could not find RESULTS markers in README.md")

    README.write_text(new_text)
    trained = ", ".join(sorted(metrics))
    print(f"Updated README results table from {len(metrics)} ticker(s): {trained}")


if __name__ == "__main__":
    main()
