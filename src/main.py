from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

from data_loader import find_data_file, load_stock_data, normalize_stock_code
from indicators import calculate_support_resistance
from visualize import plot_support_resistance


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    default_data = project_root / "data"
    default_output = project_root / "outputs"

    parser = argparse.ArgumentParser(description="基于技术形态的支撑位/压力位识别工具")
    parser.add_argument("--code", type=str, default=None, help="股票代码，如 000001.SZ / sh600519")
    parser.add_argument("--data-dir", type=Path, default=default_data, help="行情数据目录")
    parser.add_argument("--window", type=int, default=120, help="最近N日窗口，默认120")
    parser.add_argument("--extrema-window", type=int, default=5, help="局部高低点窗口，默认5")
    parser.add_argument("--end-date", type=str, default=None, help="分析截止日期，如 2026-03-31")
    parser.add_argument("--history-scan", action="store_true", help="回溯过去5年逐日扫描形态并统计变化")
    parser.add_argument("--output-dir", type=Path, default=default_output, help="图片输出目录")
    return parser


def _input_code_if_needed(code: str | None) -> str:
    if code:
        return code.strip()
    user_input = input("请输入股票代码（如 000001.SZ / sh600519）: ").strip()
    if not user_input:
        raise ValueError("股票代码不能为空。")
    return user_input


def _print_result_table(code: str, result: dict) -> None:
    summary = pd.DataFrame(
        [
            {"指标": "股票代码", "值": code},
            {"指标": "压力线斜率", "值": f"{result['pressure_slope']:.6f}"},
            {"指标": "支撑线斜率", "值": f"{result['support_slope']:.6f}"},
            {"指标": "持续形态", "值": result["pattern"]},
            {"指标": "通道形态", "值": result["channel_pattern"]},
            {"指标": "向上突破概率", "值": f"{result['breakout_up']:.2%}"},
            {"指标": "向下跌破概率", "值": f"{result['breakout_down']:.2%}"},
            {"指标": "方向强度", "值": f"{result['strength']:.2%}"},
        ]
    )

    print("\n=== 支撑/压力位综合结果 ===")
    print(tabulate(summary, headers="keys", tablefmt="github", showindex=False))

    stats = result.get("breakout_returns", {})
    if stats:
        rows = []
        for k, v in stats.items():
            rows.append({"持有周期": k, "平均收益": f"{v['平均收益']:.2%}", "胜率": f"{v['胜率']:.2%}", "样本数": v["样本数"]})
        print("\n=== 突破压力线后的收益统计 ===")
        print(tabulate(pd.DataFrame(rows), headers="keys", tablefmt="github", showindex=False))

    fb_stats = result.get("false_breakdown_rebound", {})
    if fb_stats:
        rows = []
        for k, v in fb_stats.items():
            rows.append(
                {
                    "持有周期": k,
                    "平均收益": f"{v['平均收益']:.2%}",
                    "胜率(做多)": f"{v['胜率']:.2%}",
                    "假跌破概率": f"{v['假跌破概率']:.2%}",
                    "样本数": v["样本数"],
                }
            )
        print("\n=== 下破支撑线后反弹统计（做多视角） ===")
        print(tabulate(pd.DataFrame(rows), headers="keys", tablefmt="github", showindex=False))


def _build_segments(history_df: pd.DataFrame) -> pd.DataFrame:
    if history_df.empty:
        return pd.DataFrame(columns=["开始日期", "结束日期", "持续形态", "通道形态", "连续天数"])

    segments: List[Dict[str, object]] = []
    start_idx = 0
    for i in range(1, len(history_df) + 1):
        hit_end = i == len(history_df)
        changed = False if hit_end else (
            history_df.iloc[i]["pattern"] != history_df.iloc[start_idx]["pattern"]
            or history_df.iloc[i]["channel_pattern"] != history_df.iloc[start_idx]["channel_pattern"]
        )
        if hit_end or changed:
            start_row = history_df.iloc[start_idx]
            end_row = history_df.iloc[i - 1]
            segments.append(
                {
                    "开始日期": start_row["date"].date().isoformat(),
                    "结束日期": end_row["date"].date().isoformat(),
                    "持续形态": start_row["pattern"],
                    "通道形态": start_row["channel_pattern"],
                    "连续天数": int(i - start_idx),
                }
            )
            start_idx = i
    return pd.DataFrame(segments)


def _build_change_points(history_df: pd.DataFrame) -> pd.DataFrame:
    if len(history_df) <= 1:
        return pd.DataFrame(columns=["变化日期", "旧持续形态", "新持续形态", "旧通道形态", "新通道形态"])

    changes: List[Dict[str, str]] = []
    for i in range(1, len(history_df)):
        prev_row = history_df.iloc[i - 1]
        curr_row = history_df.iloc[i]
        if prev_row["pattern"] != curr_row["pattern"] or prev_row["channel_pattern"] != curr_row["channel_pattern"]:
            changes.append(
                {
                    "变化日期": curr_row["date"].date().isoformat(),
                    "旧持续形态": prev_row["pattern"],
                    "新持续形态": curr_row["pattern"],
                    "旧通道形态": prev_row["channel_pattern"],
                    "新通道形态": curr_row["channel_pattern"],
                }
            )
    return pd.DataFrame(changes)


def _run_history_scan(
    df: pd.DataFrame,
    code: str,
    output_dir: Path,
    window: int,
    extrema_window: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    end_dt = df["date"].max()
    start_dt = end_dt - pd.DateOffset(years=5)
    eligible_idx = df.index[df["date"] >= start_dt].tolist()
    if not eligible_idx:
        raise ValueError("过去5年没有可扫描数据。")

    results: List[Dict[str, object]] = []
    for pos in eligible_idx:
        if pos + 1 < window:
            continue
        current_date = df.iloc[pos]["date"]
        source_window = df.iloc[pos - window + 1 : pos + 1]
        try:
            res = calculate_support_resistance(source_window, n_window=window, extrema_window=extrema_window)
            results.append(
                {
                    "date": current_date,
                    "pattern": res["pattern"],
                    "channel_pattern": res["channel_pattern"],
                    "pressure_slope": round(float(res["pressure_slope"]), 6),
                    "support_slope": round(float(res["support_slope"]), 6),
                    "breakout_up": round(float(res["breakout_up"]), 6),
                    "breakout_down": round(float(res["breakout_down"]), 6),
                }
            )
        except Exception:
            continue

    history_df = pd.DataFrame(results)
    if history_df.empty:
        raise ValueError("过去5年逐日扫描未得到有效形态结果，请检查window/extrema-window参数。")

    segments_df = _build_segments(history_df)
    changes_df = _build_change_points(history_df)

    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / f"{code}_pattern_history_5y.csv"
    segments_path = output_dir / f"{code}_pattern_segments_5y.csv"
    changes_path = output_dir / f"{code}_pattern_changes_5y.csv"
    prob_chart_path = output_dir / f"{code}_breakout_up_5y.png"
    history_df.to_csv(history_path, index=False, encoding="utf-8-sig")
    segments_df.to_csv(segments_path, index=False, encoding="utf-8-sig")
    changes_df.to_csv(changes_path, index=False, encoding="utf-8-sig")

    # 绘制向上突破概率时间序列
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(history_df["date"], history_df["breakout_up"], color="#1f77b4", linewidth=1.4, label="Breakout Up")
    ax.set_title(f"{code} 向上突破概率（过去5年逐日）")
    ax.set_xlabel("Date")
    ax.set_ylabel("Breakout Up Probability")
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="best")
    fig.autofmt_xdate()
    plt.tight_layout()
    fig.savefig(prob_chart_path, dpi=140)
    plt.close(fig)

    print("\n=== 过去5年逐日形态扫描 ===")
    print(f"覆盖交易日: {len(history_df)}")
    print(f"变化次数: {len(changes_df)}")
    print(f"明细文件: {history_path}")
    print(f"连续区间: {segments_path}")
    print(f"变化拐点: {changes_path}")
    print(f"突破概率图: {prob_chart_path}")

    print("\n=== 最近10个变化拐点 ===")
    if changes_df.empty:
        print("无变化拐点。")
    else:
        print(tabulate(changes_df.tail(10), headers="keys", tablefmt="github", showindex=False))

    print("\n=== 最近10段连续区间 ===")
    print(tabulate(segments_df.tail(10), headers="keys", tablefmt="github", showindex=False))
    return history_df, segments_df, changes_df


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    raw_code = _input_code_if_needed(args.code)
    normalized = normalize_stock_code(raw_code)
    file_path = find_data_file(args.data_dir, raw_code)

    df = load_stock_data(file_path)
    if args.end_date:
        end_dt = pd.to_datetime(args.end_date, errors="coerce")
        if pd.isna(end_dt):
            raise ValueError(f"无法解析 --end-date: {args.end_date}")
        df = df[df["date"] <= end_dt].reset_index(drop=True)
        if df.empty:
            raise ValueError(f"截止到 {args.end_date} 后无可用数据。")

    if args.history_scan:
        _run_history_scan(
            df=df,
            code=normalized,
            output_dir=args.output_dir,
            window=args.window,
            extrema_window=args.extrema_window,
        )
        return

    result = calculate_support_resistance(
        df,
        n_window=args.window,
        extrema_window=args.extrema_window,
    )

    _print_result_table(normalized, result)

    img_path = args.output_dir / f"{normalized}_support_resistance.png"
    out = plot_support_resistance(result["recent_df"], result, normalized, img_path)
    print(f"\n图像已保存: {out}")


if __name__ == "__main__":
    main()
