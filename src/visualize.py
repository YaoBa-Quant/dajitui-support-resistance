from __future__ import annotations

from pathlib import Path
from typing import Dict

from matplotlib import font_manager
import matplotlib.pyplot as plt
import pandas as pd


def _choose_font_and_language() -> str:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "PingFang SC",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
    ]
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in installed:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return "zh"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return "en"


def _safe_pattern_text(result: Dict[str, object], lang: str) -> str:
    if lang == "zh":
        return f"形态: {result['pattern']} | 通道: {result['channel_pattern']}"
    return "Pattern: n/a | Channel: n/a"


def plot_support_resistance(
    df: pd.DataFrame,
    result: Dict[str, object],
    stock_code: str,
    output_path: Path,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    lang = _choose_font_and_language()

    normalized_prices = result["normalized_prices"]
    pressure_line = result["pressure_line"]
    support_line = result["support_line"]
    high_points = result["high_points"]
    low_points = result["low_points"]
    breakout_up = result["breakout_up"]
    breakout_down = result["breakout_down"]
    strength = result["strength"]

    fig, ax = plt.subplots(figsize=(13, 7))

    if lang == "zh":
        price_label = "价格"
        pressure_label = "压力线"
        support_label = "支撑线"
        high_label = "局部高点"
        low_label = "局部低点"
        title = (
            f"{stock_code} 压力线与支撑线识别（最近{len(df)}日）\n"
            f"向上突破概率: {breakout_up:.2%} | 向下跌破概率: {breakout_down:.2%} | 强度: {strength:.2%}"
        )
        y_label = "标准化价格"
    else:
        price_label = "Price"
        pressure_label = "Resistance Line"
        support_label = "Support Line"
        high_label = "Local Highs"
        low_label = "Local Lows"
        title = (
            f"{stock_code} Support/Resistance (Last {len(df)} Days)\n"
            f"Breakout Up: {breakout_up:.2%} | Breakdown Down: {breakout_down:.2%} | Strength: {strength:.2%}"
        )
        y_label = "Normalized Price"

    ax.plot(df["date"], normalized_prices, color="black", linewidth=1.5, label=price_label)
    ax.plot(df["date"], pressure_line, color="red", linestyle="--", linewidth=1.6, label=pressure_label)
    ax.plot(df["date"], support_line, color="green", linestyle="--", linewidth=1.6, label=support_label)
    ax.scatter(
        df["date"].iloc[high_points["index"].to_numpy(dtype=int)],
        high_points["price"],
        color="red",
        s=25,
        label=high_label,
        zorder=5,
    )
    ax.scatter(
        df["date"].iloc[low_points["index"].to_numpy(dtype=int)],
        low_points["price"],
        color="green",
        s=25,
        label=low_label,
        zorder=5,
    )
    ax.fill_between(df["date"], pressure_line, support_line, color="gray", alpha=0.18)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel(y_label)
    ax.legend(loc="best")
    fig.autofmt_xdate()

    ax.text(
        0.01,
        0.02,
        _safe_pattern_text(result, lang),
        transform=ax.transAxes,
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "gray"},
    )

    plt.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)
    return output_path
