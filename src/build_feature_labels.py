from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_loader import load_stock_data
from indicators import calculate_support_resistance


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成全市场逐日形态特征与未来收益标签")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data", help="行情数据目录")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs", help="输出目录")
    parser.add_argument("--window", type=int, default=240, help="回看窗口，默认240")
    parser.add_argument("--extrema-window", type=int, default=5, help="局部高低点窗口，默认5")
    parser.add_argument("--horizons", type=str, default="5,10,20,60", help="前瞻收益周期，逗号分隔")
    parser.add_argument("--limit", type=int, default=0, help="仅处理前N个文件，0表示全部")
    parser.add_argument("--start-date", type=str, default=None, help="样本起始日期，如 2020-01-01")
    parser.add_argument("--end-date", type=str, default=None, help="样本结束日期，如 2025-12-31")
    return parser


def _parse_horizons(value: str) -> list[int]:
    items = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        horizon = int(chunk)
        if horizon <= 0:
            raise ValueError(f"horizon 必须为正整数，收到: {horizon}")
        items.append(horizon)
    if not items:
        raise ValueError("至少需要一个前瞻收益周期。")
    return sorted(set(items))


def _progress_bar(current: int, total: int, width: int = 26) -> str:
    total = max(total, 1)
    filled = int(width * current / total)
    return f"[{'#' * filled}{'-' * (width - filled)}] {current}/{total}"


def _append_csv(df: pd.DataFrame, file_path: Path) -> None:
    if df.empty:
        return
    write_header = (not file_path.exists()) or file_path.stat().st_size == 0
    df.to_csv(file_path, mode="a", header=write_header, index=False, encoding="utf-8-sig")


def _build_one_symbol_samples(
    file_path: Path,
    window: int,
    extrema_window: int,
    horizons: list[int],
    start_dt: pd.Timestamp | None,
    end_dt: pd.Timestamp | None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    df = load_stock_data(file_path)
    code = file_path.stem
    if start_dt is not None:
        df = df[df["date"] >= start_dt].reset_index(drop=True)
    if end_dt is not None:
        df = df[df["date"] <= end_dt].reset_index(drop=True)

    if len(df) < window:
        raise ValueError(f"数据长度({len(df)})小于回看窗口({window})。")

    samples: list[dict[str, object]] = []
    success_count = 0
    failed_windows = 0
    for end_idx in range(window - 1, len(df)):
        window_df = df.iloc[end_idx - window + 1 : end_idx + 1].copy()
        try:
            result = calculate_support_resistance(window_df, n_window=window, extrema_window=extrema_window)
            latest_close = float(window_df["close"].iloc[-1])
            latest_date = window_df["date"].iloc[-1]
            latest_norm_price = float(result["normalized_prices"][-1])
            latest_support = float(result["support_line"][-1])
            latest_pressure = float(result["pressure_line"][-1])
            channel_width = latest_pressure - latest_support
            distance_to_support = (latest_norm_price - latest_support) / max(latest_support, 1e-8)
            distance_to_pressure = (latest_pressure - latest_norm_price) / max(abs(latest_pressure), 1e-8)
            position_in_channel = (latest_norm_price - latest_support) / max(channel_width, 1e-8)

            row: dict[str, object] = {
                "date": latest_date.date().isoformat(),
                "code": code,
                "close": latest_close,
                "pattern": result["pattern"],
                "channel_pattern": result["channel_pattern"],
                "pressure_slope": float(result["pressure_slope"]),
                "support_slope": float(result["support_slope"]),
                "breakout_up": float(result["breakout_up"]),
                "breakout_down": float(result["breakout_down"]),
                "strength": float(result["strength"]),
                "latest_support_norm": latest_support,
                "latest_pressure_norm": latest_pressure,
                "distance_to_support": distance_to_support,
                "distance_to_pressure": distance_to_pressure,
                "channel_width": channel_width,
                "position_in_channel": position_in_channel,
                "n_high_points": int(len(result["high_points"])),
                "n_low_points": int(len(result["low_points"])),
            }
            for horizon in horizons:
                future_idx = end_idx + horizon
                if future_idx < len(df):
                    future_close = float(df["close"].iloc[future_idx])
                    row[f"ret_fwd_{horizon}d"] = future_close / max(latest_close, 1e-8) - 1.0
                    row[f"label_up_{horizon}d"] = int(future_close > latest_close)
                else:
                    row[f"ret_fwd_{horizon}d"] = pd.NA
                    row[f"label_up_{horizon}d"] = pd.NA
            samples.append(row)
            success_count += 1
        except Exception:
            failed_windows += 1
            continue

    meta = {
        "code": code,
        "rows_total": int(len(df)),
        "samples_built": int(success_count),
        "failed_windows": int(failed_windows),
        "first_date": df["date"].iloc[0].date().isoformat(),
        "last_date": df["date"].iloc[-1].date().isoformat(),
    }
    return pd.DataFrame(samples), meta


def main() -> None:
    args = build_parser().parse_args()
    horizons = _parse_horizons(args.horizons)
    start_dt = pd.to_datetime(args.start_date) if args.start_date else None
    end_dt = pd.to_datetime(args.end_date) if args.end_date else None

    files = sorted(args.data_dir.glob("*.csv"))
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"在 {args.data_dir} 未找到任何 CSV 文件。")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    feature_path = args.output_dir / f"feature_labels_{stamp}.csv"
    quality_path = args.output_dir / f"feature_labels_quality_{stamp}.csv"

    print("开始生成逐日特征与标签...")
    print(f"数据目录: {args.data_dir}")
    print(f"回看窗口: {args.window} | extrema-window: {args.extrema_window}")
    print(f"前瞻周期: {horizons}")
    print("-" * 90, flush=True)

    if feature_path.exists():
        feature_path.unlink()
    if quality_path.exists():
        quality_path.unlink()

    total_files = len(files)
    ok_count = 0
    total_sample_rows = 0
    label_non_null_counts = {f"ret_fwd_{h}d": 0 for h in horizons}
    for idx, file_path in enumerate(files, start=1):
        code = file_path.stem
        try:
            panel_df, meta = _build_one_symbol_samples(
                file_path=file_path,
                window=args.window,
                extrema_window=args.extrema_window,
                horizons=horizons,
                start_dt=start_dt,
                end_dt=end_dt,
            )
            quality_row = pd.DataFrame([{"status": "ok", "message": "", **meta}])
            _append_csv(quality_row, quality_path)
            if not panel_df.empty:
                _append_csv(panel_df, feature_path)
                total_sample_rows += len(panel_df)
                for label_col in label_non_null_counts:
                    label_non_null_counts[label_col] += int(panel_df[label_col].notna().sum())
            ok_count += 1
            print(
                f"{_progress_bar(idx, total_files)} {code} | 样本 {meta['samples_built']} 条 | 失败窗口 {meta['failed_windows']} 条",
                flush=True,
            )
        except Exception as exc:
            quality_row = pd.DataFrame(
                [
                    {
                        "code": code,
                        "status": "error",
                        "message": str(exc),
                        "rows_total": pd.NA,
                        "samples_built": 0,
                        "failed_windows": pd.NA,
                        "first_date": pd.NA,
                        "last_date": pd.NA,
                    }
                ]
            )
            _append_csv(quality_row, quality_path)
            print(f"{_progress_bar(idx, total_files)} {code} | 处理失败 | {exc}", flush=True)
        if idx == 1 or idx % 50 == 0 or idx == total_files:
            print(f"已保存进度 -> 特征: {feature_path} | 质量: {quality_path}", flush=True)

    print("-" * 90)
    print(f"生成完成: {feature_path}")
    print(f"质量报告: {quality_path}")
    print(f"总股票数: {total_files}")
    print(f"成功股票数: {ok_count}")
    print(f"样本总行数: {total_sample_rows}")

    if total_sample_rows > 0:
        coverage = {
            label_col: round(non_null_count / total_sample_rows * 100.0, 2)
            for label_col, non_null_count in label_non_null_counts.items()
        }
        print(f"标签覆盖率(%): {coverage}")


if __name__ == "__main__":
    main()
