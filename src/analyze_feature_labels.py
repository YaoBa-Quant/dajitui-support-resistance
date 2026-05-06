from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="评估形态/通道/通道距离与未来收益(5/10/20/60日)的关系")
    parser.add_argument("--input-file", type=Path, default=None, help="特征标签文件，默认读取最新 feature_labels_*.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs", help="输出目录")
    parser.add_argument("--horizons", type=str, default="5,10,20,60", help="前瞻收益周期，逗号分隔")
    parser.add_argument("--chunksize", type=int, default=200_000, help="分块读取行数")
    parser.add_argument("--limit-codes", type=int, default=0, help="仅处理前N个股票，0表示全部")
    parser.add_argument("--bins", type=int, default=5, help="分位分箱数量(用于位置/距离分箱统计)")
    parser.add_argument("--ret-min", type=float, default=-0.95, help="过滤未来收益下限(异常值置空)")
    parser.add_argument("--ret-max", type=float, default=10.0, help="过滤未来收益上限(异常值置空)")
    return parser


def _parse_horizons(value: str) -> list[int]:
    items: list[int] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        h = int(chunk)
        if h <= 0:
            raise ValueError(f"horizons 必须为正整数，收到: {h}")
        items.append(h)
    if not items:
        raise ValueError("至少需要一个 horizons。")
    return sorted(set(items))


def _find_latest_feature_file(output_dir: Path) -> Path:
    candidates = sorted(
        [p for p in output_dir.glob("feature_labels_*.csv") if "quality" not in p.name.lower()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"在 {output_dir} 未找到 feature_labels_*.csv。")
    return candidates[0]


def _clean_ret(ret: pd.Series, ret_min: float, ret_max: float) -> pd.Series:
    s = pd.to_numeric(ret, errors="coerce").replace([np.inf, -np.inf], np.nan)
    s = s.where((s >= ret_min) & (s <= ret_max), np.nan)
    return s


def _safe_spearman_corr(x: pd.Series, y: pd.Series) -> float | None:
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < 30:
        return None
    v = float(df.iloc[:, 0].corr(df.iloc[:, 1], method="spearman"))
    if np.isnan(v):
        return None
    return v


def _safe_mean(x: pd.Series) -> float | None:
    x = pd.to_numeric(x, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return None
    return float(x.mean())


def _safe_win_rate(ret: pd.Series) -> float | None:
    ret = pd.to_numeric(ret, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if ret.empty:
        return None
    return float((ret > 0).mean())


def _quantile_bin_stats(x: pd.Series, y: pd.Series, bins: int) -> dict[int, dict[str, float]] | None:
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < max(60, bins * 20):
        return None
    xs = pd.to_numeric(df.iloc[:, 0], errors="coerce").replace([np.inf, -np.inf], np.nan)
    ys = pd.to_numeric(df.iloc[:, 1], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df = pd.DataFrame({"x": xs, "y": ys}).dropna()
    if len(df) < max(60, bins * 20):
        return None
    try:
        q = pd.qcut(df["x"], q=bins, labels=False, duplicates="drop")
    except ValueError:
        return None
    df["bin"] = q.astype("Int64")
    stats: dict[int, dict[str, float]] = {}
    for b, g in df.groupby("bin", sort=True):
        if pd.isna(b):
            continue
        b = int(b) + 1
        stats[b] = {
            "n": float(len(g)),
            "mean_ret": float(g["y"].mean()),
            "win_rate": float((g["y"] > 0).mean()),
        }
    if not stats:
        return None
    return stats


def main() -> None:
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    horizons = _parse_horizons(args.horizons)
    input_file = args.input_file or _find_latest_feature_file(args.output_dir)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = args.output_dir / f"relationship_summary_{stamp}.csv"
    pattern_path = args.output_dir / f"relationship_pattern_{stamp}.csv"
    channel_path = args.output_dir / f"relationship_channel_{stamp}.csv"
    bins_path = args.output_dir / f"relationship_bins_{stamp}.csv"

    numeric_features = [
        "pressure_slope",
        "support_slope",
        "breakout_up",
        "breakout_down",
        "strength",
        "distance_to_support",
        "distance_to_pressure",
        "channel_width",
        "position_in_channel",
        "n_high_points",
        "n_low_points",
    ]

    corr_values: dict[tuple[int, str], list[float]] = defaultdict(list)
    corr_codes: dict[tuple[int, str], int] = defaultdict(int)

    pattern_stats: dict[tuple[int, str], dict[str, float]] = defaultdict(lambda: {"sum": 0.0, "pos": 0.0, "n": 0.0})
    channel_stats: dict[tuple[int, str], dict[str, float]] = defaultdict(lambda: {"sum": 0.0, "pos": 0.0, "n": 0.0})

    bin_stats: dict[tuple[int, str, int], dict[str, float]] = defaultdict(lambda: {"sum": 0.0, "pos": 0.0, "n": 0.0})
    binned_features = ["position_in_channel", "distance_to_support", "distance_to_pressure"]

    use_cols = ["date", "code", "pattern", "channel_pattern", *numeric_features]
    for h in horizons:
        use_cols.append(f"ret_fwd_{h}d")
    use_cols = sorted(set(use_cols))

    processed_codes = 0
    carry: pd.DataFrame | None = None
    current_code: str | None = None

    for chunk in pd.read_csv(input_file, usecols=use_cols, chunksize=args.chunksize):
        chunk["code"] = chunk["code"].astype(str)
        if carry is not None and not carry.empty:
            chunk = pd.concat([carry, chunk], ignore_index=True)
            carry = None

        if chunk.empty:
            continue

        codes = chunk["code"].tolist()
        last_code = codes[-1] if codes else None
        if last_code is not None:
            tail_mask = chunk["code"] == last_code
            tail_df = chunk.loc[tail_mask].copy()
            body_df = chunk.loc[~tail_mask].copy()
            carry = tail_df
        else:
            body_df = chunk
            carry = None

        if body_df.empty:
            continue

        for code, g in body_df.groupby("code", sort=False):
            current_code = code
            processed_codes += 1
            for h in horizons:
                y_col = f"ret_fwd_{h}d"
                if y_col not in g.columns:
                    continue
                y = _clean_ret(g[y_col], args.ret_min, args.ret_max)

                for feat in numeric_features:
                    if feat not in g.columns:
                        continue
                    v = _safe_spearman_corr(g[feat], y)
                    if v is None:
                        continue
                    corr_values[(h, feat)].append(v)
                    corr_codes[(h, feat)] += 1

                if "pattern" in g.columns:
                    for p, pg in g.groupby("pattern", sort=False):
                        yy = _clean_ret(pg[y_col], args.ret_min, args.ret_max)
                        m = _safe_mean(yy)
                        w = _safe_win_rate(yy)
                        n = float(yy.notna().sum())
                        if m is None or w is None or n <= 0:
                            continue
                        row = pattern_stats[(h, str(p))]
                        row["sum"] += m * n
                        row["pos"] += w * n
                        row["n"] += n

                if "channel_pattern" in g.columns:
                    for p, pg in g.groupby("channel_pattern", sort=False):
                        yy = _clean_ret(pg[y_col], args.ret_min, args.ret_max)
                        m = _safe_mean(yy)
                        w = _safe_win_rate(yy)
                        n = float(yy.notna().sum())
                        if m is None or w is None or n <= 0:
                            continue
                        row = channel_stats[(h, str(p))]
                        row["sum"] += m * n
                        row["pos"] += w * n
                        row["n"] += n

                for feat in binned_features:
                    if feat not in g.columns:
                        continue
                    qstats = _quantile_bin_stats(g[feat], y, bins=args.bins)
                    if not qstats:
                        continue
                    for b, s in qstats.items():
                        n = float(s["n"])
                        if n <= 0:
                            continue
                        row = bin_stats[(h, feat, int(b))]
                        row["sum"] += float(s["mean_ret"]) * n
                        row["pos"] += float(s["win_rate"]) * n
                        row["n"] += n

            if args.limit_codes and processed_codes >= args.limit_codes:
                break

        if args.limit_codes and processed_codes >= args.limit_codes:
            break

    if carry is not None and not carry.empty:
        for code, g in carry.groupby("code", sort=False):
            processed_codes += 1
            for h in horizons:
                y_col = f"ret_fwd_{h}d"
                if y_col not in g.columns:
                    continue
                y = _clean_ret(g[y_col], args.ret_min, args.ret_max)
                for feat in numeric_features:
                    v = _safe_spearman_corr(g[feat], y)
                    if v is None:
                        continue
                    corr_values[(h, feat)].append(v)
                    corr_codes[(h, feat)] += 1

                if "pattern" in g.columns:
                    for p, pg in g.groupby("pattern", sort=False):
                        yy = _clean_ret(pg[y_col], args.ret_min, args.ret_max)
                        m = _safe_mean(yy)
                        w = _safe_win_rate(yy)
                        n = float(yy.notna().sum())
                        if m is None or w is None or n <= 0:
                            continue
                        row = pattern_stats[(h, str(p))]
                        row["sum"] += m * n
                        row["pos"] += w * n
                        row["n"] += n

                if "channel_pattern" in g.columns:
                    for p, pg in g.groupby("channel_pattern", sort=False):
                        yy = _clean_ret(pg[y_col], args.ret_min, args.ret_max)
                        m = _safe_mean(yy)
                        w = _safe_win_rate(yy)
                        n = float(yy.notna().sum())
                        if m is None or w is None or n <= 0:
                            continue
                        row = channel_stats[(h, str(p))]
                        row["sum"] += m * n
                        row["pos"] += w * n
                        row["n"] += n

                for feat in binned_features:
                    qstats = _quantile_bin_stats(g[feat], y, bins=args.bins)
                    if not qstats:
                        continue
                    for b, s in qstats.items():
                        n = float(s["n"])
                        if n <= 0:
                            continue
                        row = bin_stats[(h, feat, int(b))]
                        row["sum"] += float(s["mean_ret"]) * n
                        row["pos"] += float(s["win_rate"]) * n
                        row["n"] += n

            if args.limit_codes and processed_codes >= args.limit_codes:
                break

    summary_rows: list[dict[str, object]] = []
    for (h, feat), vals in sorted(corr_values.items(), key=lambda x: (x[0][0], x[0][1])):
        arr = np.array(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        summary_rows.append(
            {
                "horizon": h,
                "feature": feat,
                "ic_mean": float(np.mean(arr)),
                "ic_median": float(np.median(arr)),
                "ic_std": float(np.std(arr, ddof=1)) if arr.size >= 2 else 0.0,
                "pct_pos": float(np.mean(arr > 0.0)),
                "n_codes": int(corr_codes[(h, feat)]),
            }
        )
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False, encoding="utf-8-sig")

    pattern_rows: list[dict[str, object]] = []
    for (h, p), s in sorted(pattern_stats.items(), key=lambda x: (x[0][0], x[0][1])):
        n = float(s["n"])
        if n <= 0:
            continue
        pattern_rows.append(
            {
                "horizon": h,
                "pattern": p,
                "mean_ret": float(s["sum"] / n),
                "win_rate": float(s["pos"] / n),
                "n_obs": int(n),
            }
        )
    pd.DataFrame(pattern_rows).sort_values(["horizon", "mean_ret"], ascending=[True, False]).to_csv(
        pattern_path, index=False, encoding="utf-8-sig"
    )

    channel_rows: list[dict[str, object]] = []
    for (h, p), s in sorted(channel_stats.items(), key=lambda x: (x[0][0], x[0][1])):
        n = float(s["n"])
        if n <= 0:
            continue
        channel_rows.append(
            {
                "horizon": h,
                "channel_pattern": p,
                "mean_ret": float(s["sum"] / n),
                "win_rate": float(s["pos"] / n),
                "n_obs": int(n),
            }
        )
    pd.DataFrame(channel_rows).sort_values(["horizon", "mean_ret"], ascending=[True, False]).to_csv(
        channel_path, index=False, encoding="utf-8-sig"
    )

    bin_rows: list[dict[str, object]] = []
    for (h, feat, b), s in sorted(bin_stats.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        n = float(s["n"])
        if n <= 0:
            continue
        bin_rows.append(
            {
                "horizon": h,
                "feature": feat,
                "bin": int(b),
                "mean_ret": float(s["sum"] / n),
                "win_rate": float(s["pos"] / n),
                "n_obs": int(n),
            }
        )
    pd.DataFrame(bin_rows).to_csv(bins_path, index=False, encoding="utf-8-sig")

    print(f"输入文件: {input_file}")
    print(f"处理股票数: {processed_codes}")
    print(f"相关性汇总: {summary_path}")
    print(f"形态统计: {pattern_path}")
    print(f"通道统计: {channel_path}")
    print(f"分箱统计: {bins_path}")


if __name__ == "__main__":
    main()

