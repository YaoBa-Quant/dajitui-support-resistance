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
    parser = argparse.ArgumentParser(description="批量筛选上升通道发散且靠近支撑位的股票")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data", help="行情数据目录")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs", help="结果输出目录")
    parser.add_argument("--window", type=int, default=240, help="回看窗口，默认240")
    parser.add_argument("--extrema-window", type=int, default=5, help="局部高低点窗口，默认5")
    parser.add_argument("--max-dist", type=float, default=0.02, help="距支撑位最大距离，默认0.02表示2%%")
    parser.add_argument("--limit", type=int, default=0, help="仅扫描前N个文件，0表示全部")
    parser.add_argument("--top-k", type=int, default=50, help="终端展示前K条命中结果")
    return parser


def _progress_bar(current: int, total: int, width: int = 26) -> str:
    total = max(total, 1)
    filled = int(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {current}/{total}"


def _safe_pct(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.2%}"


def _print_status(index: int, total: int, message: str) -> None:
    print(f"{_progress_bar(index, total)} {message}", flush=True)


def main() -> None:
    args = build_parser().parse_args()
    files = sorted(args.data_dir.glob("*.csv"))
    if args.limit > 0:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"在 {args.data_dir} 未找到任何 CSV 文件。")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_result_path = args.output_dir / f"screener_all_{stamp}.csv"
    hit_result_path = args.output_dir / f"screener_hits_{stamp}.csv"

    print("开始扫描...")
    print(f"数据目录: {args.data_dir}")
    print(f"筛选条件: 上升通道发散 且 距支撑位 <= {args.max_dist:.2%}")
    print(f"回看窗口: {args.window} | extrema-window: {args.extrema_window}")
    print("-" * 90, flush=True)

    rows: list[dict[str, object]] = []
    total = len(files)
    for idx, file_path in enumerate(files, start=1):
        code = file_path.stem
        row: dict[str, object] = {
            "code": code,
            "file_name": file_path.name,
            "latest_date": None,
            "pattern": None,
            "channel_pattern": None,
            "distance_to_support": None,
            "distance_to_support_pct": None,
            "breakout_up": None,
            "strength": None,
            "matched": False,
            "status": "error",
            "message": "",
        }
        try:
            df = load_stock_data(file_path)
            result = calculate_support_resistance(df, n_window=args.window, extrema_window=args.extrema_window)
            latest_price = float(result["normalized_prices"][-1])
            latest_support = float(result["support_line"][-1])
            dist = (latest_price - latest_support) / max(latest_support, 1e-8)
            matched = result["channel_pattern"] == "上升通道发散" and 0 <= dist <= args.max_dist

            row.update(
                {
                    "latest_date": result["recent_df"]["date"].iloc[-1].date().isoformat(),
                    "pattern": result["pattern"],
                    "channel_pattern": result["channel_pattern"],
                    "distance_to_support": dist,
                    "distance_to_support_pct": dist * 100.0,
                    "breakout_up": float(result["breakout_up"]),
                    "strength": float(result["strength"]),
                    "matched": matched,
                    "status": "ok",
                }
            )

            if result["channel_pattern"] == "上升通道发散":
                message = (
                    f"{code} | 上升通道发散 | 距支撑位 {_safe_pct(dist)}"
                    f" | {'命中筛选' if matched else '未命中'}"
                )
            else:
                message = f"{code} | {result['channel_pattern']} | 不满足通道条件"
            row["message"] = message
            _print_status(idx, total, message)
        except Exception as exc:
            row["message"] = str(exc)
            _print_status(idx, total, f"{code} | 处理失败 | {exc}")
        rows.append(row)

    all_df = pd.DataFrame(rows)
    hit_df = all_df[all_df["matched"] == True].copy()
    if not hit_df.empty:
        hit_df = hit_df.sort_values("distance_to_support", ascending=True).reset_index(drop=True)

    all_df.to_csv(all_result_path, index=False, encoding="utf-8-sig")
    hit_df.to_csv(hit_result_path, index=False, encoding="utf-8-sig")

    print("-" * 90)
    print(f"扫描完成: {total} 只")
    print(f"命中数量: {len(hit_df)}")
    print(f"全量结果: {all_result_path}")
    print(f"命中结果: {hit_result_path}")

    if hit_df.empty:
        print("未发现满足条件的股票。")
        return

    view_cols = [
        "code",
        "latest_date",
        "pattern",
        "channel_pattern",
        "distance_to_support_pct",
        "breakout_up",
        "strength",
    ]
    preview = hit_df[view_cols].head(args.top_k).copy()
    preview["distance_to_support_pct"] = preview["distance_to_support_pct"].map(lambda x: f"{x:.2f}%")
    preview["breakout_up"] = preview["breakout_up"].map(lambda x: f"{x:.2%}")
    preview["strength"] = preview["strength"].map(lambda x: f"{x:.2%}")
    print("\n命中结果预览:")
    print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
