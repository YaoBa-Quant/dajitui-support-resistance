from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from indicators import calculate_support_resistance


def build_synthetic_df(rows: int = 120) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=rows, freq="B")
    x = np.arange(rows, dtype=float)
    base = 100 + np.sin(x / 3.0) * 1.5
    high = base + 4
    low = base - 4
    close = base.copy()
    open_ = close + np.cos(x / 5.0) * 0.2
    volume = np.full(rows, 1_000_000.0)

    for i in range(8, rows, 15):
        high[i] = 110.0
    for i in range(4, rows, 15):
        low[i] = 90.0

    return pd.DataFrame(
        {
            "date": dates,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def test_support_resistance_on_known_extrema():
    df = build_synthetic_df()
    result = calculate_support_resistance(df, n_window=100, extrema_window=3)
    assert "pressure_line" in result and "support_line" in result
    assert len(result["high_points"]) >= 2
    assert len(result["low_points"]) >= 2
    assert result["pressure_slope"] > result["support_slope"]
    assert result["pattern"] in {"喇叭形", "其他形态", "上升三角形", "对称三角形", "下降三角形", "矩形", "旗形"}
