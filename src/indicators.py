from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from sklearn.linear_model import QuantileRegressor


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def normalize_prices(prices: pd.Series) -> pd.Series:
    first = float(prices.iloc[0])
    if abs(first) < 1e-12:
        return prices.copy()
    return prices / first * 100.0


def find_local_points(series: pd.Series, window: int, mode: str) -> np.ndarray:
    values = series.to_numpy()
    if len(values) < window * 2 + 1:
        return np.array([], dtype=int)
    comparator = np.greater if mode == "max" else np.less
    return argrelextrema(values, comparator, order=window)[0]


def _fit_quantile_line(x: np.ndarray, y: np.ndarray, quantile: float) -> Tuple[float, float]:
    if len(x) < 2:
        return 0.0, float(np.median(y)) if len(y) else 0.0
    model = QuantileRegressor(quantile=quantile, alpha=0.0, solver="highs")
    model.fit(x.reshape(-1, 1), y)
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    return slope, intercept


def identify_pattern(pressure_slope: float, support_slope: float) -> str:
    patterns = {
        "上升三角形": ((-0.05, 0.05), (0.05, 0.15)),
        "对称三角形": ((-0.15, -0.05), (0.05, 0.15)),
        "下降三角形": ((-0.05, 0.05), (-0.15, -0.05)),
        "矩形": ((-0.05, 0.05), (-0.05, 0.05)),
        "旗形": ((-0.15, -0.05), (-0.15, -0.05)),
        "喇叭形": ((0.05, np.inf), (-np.inf, -0.05)),
    }
    for name, (p_range, s_range) in patterns.items():
        if p_range[0] <= pressure_slope <= p_range[1] and s_range[0] <= support_slope <= s_range[1]:
            return name
    return "其他形态"


def identify_channel_pattern(pressure_slope: float, support_slope: float) -> str:
    if pressure_slope > 0 and support_slope > 0:
        return "上升通道收敛" if pressure_slope < support_slope else "上升通道发散"
    if -0.15 < pressure_slope < 0.05 and -0.15 < support_slope < 0.05:
        return "横盘"
    if pressure_slope < 0 and support_slope < 0:
        return "下降通道收敛" if pressure_slope < support_slope else "下降通道发散"
    return "其他通道"


def calculate_breakout_returns(
    normalized_prices: np.ndarray,
    pressure_line: np.ndarray,
    hold_days: List[int] | None = None,
) -> Dict[str, Dict[str, float]]:
    hold_days = hold_days or [5, 10, 20]
    breakout_indices: List[int] = []
    for i in range(1, len(normalized_prices)):
        if normalized_prices[i - 1] <= pressure_line[i - 1] and normalized_prices[i] > pressure_line[i]:
            breakout_indices.append(i)

    ret: Dict[str, Dict[str, float]] = {}
    for days in hold_days:
        period_returns: List[float] = []
        for idx in breakout_indices:
            if idx + days < len(normalized_prices):
                r = (normalized_prices[idx + days] - normalized_prices[idx]) / max(normalized_prices[idx], 1e-8)
                period_returns.append(float(r))
        if period_returns:
            arr = np.array(period_returns, dtype=float)
            ret[f"持有{days}天"] = {
                "平均收益": float(arr.mean()),
                "胜率": float((arr > 0).mean()),
                "样本数": int(len(arr)),
            }
    return ret


def calculate_false_breakdown_rebound(
    normalized_prices: np.ndarray,
    support_line: np.ndarray,
    hold_days: List[int] | None = None,
) -> Dict[str, Dict[str, float]]:
    hold_days = hold_days or [5, 10, 20]
    breakdown_indices: List[int] = []
    for i in range(1, len(normalized_prices)):
        if normalized_prices[i - 1] >= support_line[i - 1] and normalized_prices[i] < support_line[i]:
            breakdown_indices.append(i)

    ret: Dict[str, Dict[str, float]] = {}
    for days in hold_days:
        period_returns: List[float] = []
        back_above_support = 0
        for idx in breakdown_indices:
            if idx + days < len(normalized_prices):
                r = (normalized_prices[idx + days] - normalized_prices[idx]) / max(normalized_prices[idx], 1e-8)
                period_returns.append(float(r))
                if normalized_prices[idx + days] > support_line[idx + days]:
                    back_above_support += 1
        if period_returns:
            arr = np.array(period_returns, dtype=float)
            sample_n = int(len(arr))
            ret[f"持有{days}天"] = {
                "平均收益": float(arr.mean()),
                "胜率": float((arr > 0).mean()),
                "假跌破概率": float(back_above_support / max(sample_n, 1)),
                "样本数": sample_n,
            }
    return ret


def calculate_support_resistance(
    df: pd.DataFrame,
    n_window: int = 60,
    extrema_window: int = 5,
) -> Dict[str, object]:
    if len(df) < n_window:
        raise ValueError(f"数据长度({len(df)})小于回溯期({n_window})。")

    recent = df.tail(n_window).copy().reset_index(drop=True)
    normalized_close = normalize_prices(recent["close"])
    x_range = np.arange(len(normalized_close), dtype=float)

    high_idx = find_local_points(normalized_close, window=extrema_window, mode="max")
    low_idx = find_local_points(normalized_close, window=extrema_window, mode="min")
    if len(high_idx) < 2 or len(low_idx) < 2:
        raise ValueError("局部高低点数量不足，无法拟合分位数回归线。")

    high_points = pd.DataFrame({"index": high_idx, "price": normalized_close.iloc[high_idx].to_numpy(dtype=float)})
    low_points = pd.DataFrame({"index": low_idx, "price": normalized_close.iloc[low_idx].to_numpy(dtype=float)})

    pressure_slope, pressure_intercept = _fit_quantile_line(
        high_points["index"].to_numpy(dtype=float),
        high_points["price"].to_numpy(dtype=float),
        quantile=0.9,
    )
    support_slope, support_intercept = _fit_quantile_line(
        low_points["index"].to_numpy(dtype=float),
        low_points["price"].to_numpy(dtype=float),
        quantile=0.1,
    )

    pressure_line = pressure_intercept + pressure_slope * x_range
    support_line = support_intercept + support_slope * x_range

    pattern = identify_pattern(pressure_slope, support_slope)
    channel_pattern = identify_channel_pattern(pressure_slope, support_slope)
    breakout_stats = calculate_breakout_returns(normalized_close.to_numpy(dtype=float), pressure_line)
    false_breakdown_stats = calculate_false_breakdown_rebound(normalized_close.to_numpy(dtype=float), support_line)

    width = np.maximum(pressure_line - support_line, 1e-8)
    position = np.clip((normalized_close.to_numpy(dtype=float) - support_line) / width, 0.0, 1.0)
    up_prob = sigmoid((position[-1] - 0.5) * 2.0 + pressure_slope * 2.0)
    down_prob = 1.0 - up_prob
    strength = abs(up_prob - down_prob)

    return {
        "recent_df": recent,
        "normalized_prices": normalized_close.to_numpy(dtype=float),
        "pressure_slope": float(pressure_slope),
        "pressure_intercept": float(pressure_intercept),
        "support_slope": float(support_slope),
        "support_intercept": float(support_intercept),
        "pressure_line": pressure_line,
        "support_line": support_line,
        "high_points": high_points,
        "low_points": low_points,
        "pattern": pattern,
        "channel_pattern": channel_pattern,
        "breakout_returns": breakout_stats,
        "false_breakdown_rebound": false_breakdown_stats,
        "breakout_up": float(up_prob),
        "breakout_down": float(down_prob),
        "strength": float(strength),
    }
