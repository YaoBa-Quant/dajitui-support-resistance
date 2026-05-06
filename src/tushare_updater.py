from __future__ import annotations

from datetime import date, timedelta
import os
from pathlib import Path
from typing import Any

import pandas as pd

from data_loader import load_stock_data, normalize_stock_code


def _normalize_to_ts_code(stock_code: str) -> str:
    normalized = normalize_stock_code(stock_code)
    market = normalized[:2].upper()
    code = normalized[2:]
    return f"{code}.{market}"


def _resolve_token(token: str | None) -> str:
    value = (token or os.getenv("TUSHARE_TOKEN", "")).strip()
    if not value:
        raise ValueError("缺少 Tushare Token，请在页面输入，或设置环境变量 TUSHARE_TOKEN。")
    return value


def _pick_existing_column(columns: list[str], candidates: list[str], fallback: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return fallback


def _latest_non_empty(series: pd.Series, fallback: str = "") -> str:
    cleaned = series.dropna().astype(str).str.strip()
    cleaned = cleaned[cleaned != ""]
    if cleaned.empty:
        return fallback
    return cleaned.iloc[-1]


def _build_append_frame(raw_df: pd.DataFrame, daily_df: pd.DataFrame, normalized_code: str) -> pd.DataFrame:
    columns = list(raw_df.columns)
    date_col = _pick_existing_column(columns, ["日期", "date", "交易日期"], "日期")
    time_col = _pick_existing_column(columns, ["时间", "time"], "时间")
    code_col = _pick_existing_column(columns, ["代码", "code"], "代码")
    name_col = _pick_existing_column(columns, ["名称", "name"], "名称")
    open_col = _pick_existing_column(columns, ["开盘", "开盘价", "open"], "开盘")
    high_col = _pick_existing_column(columns, ["最高", "最高价", "high"], "最高")
    low_col = _pick_existing_column(columns, ["最低", "最低价", "low"], "最低")
    close_col = _pick_existing_column(columns, ["收盘", "收盘价", "close"], "收盘")
    volume_col = _pick_existing_column(columns, ["总手", "成交量", "volume", "vol"], "总手")
    amount_col = _pick_existing_column(columns, ["金额", "amount"], "金额")

    default_time = _latest_non_empty(raw_df[time_col], "15:00") if time_col in raw_df.columns else "15:00"
    default_name = _latest_non_empty(raw_df[name_col], normalized_code) if name_col in raw_df.columns else normalized_code

    append_df = pd.DataFrame(
        {
            date_col: pd.to_datetime(daily_df["trade_date"], format="%Y%m%d").dt.strftime("%Y-%m-%d"),
            time_col: default_time,
            code_col: normalized_code,
            name_col: default_name,
            open_col: daily_df["open"].astype(float),
            high_col: daily_df["high"].astype(float),
            low_col: daily_df["low"].astype(float),
            close_col: daily_df["close"].astype(float),
            volume_col: daily_df["vol"].astype(float),
            amount_col: daily_df["amount"].astype(float),
        }
    )

    for col in columns:
        if col not in append_df.columns:
            append_df[col] = pd.NA
    return append_df[columns]


def update_stock_data_via_tushare(
    file_path: Path,
    stock_code: str,
    token: str | None = None,
    end_date: date | None = None,
) -> dict[str, Any]:
    if file_path.suffix.lower() != ".csv":
        raise ValueError("当前仅支持更新 CSV 数据文件。")

    resolved_token = _resolve_token(token)
    try:
        import tushare as ts
    except ImportError as exc:
        raise ImportError("未安装 tushare，请先执行 `pip install tushare`。") from exc

    normalized_code = normalize_stock_code(stock_code)
    ts_code = _normalize_to_ts_code(stock_code)
    clean_df = load_stock_data(file_path)
    raw_df = pd.read_csv(file_path)

    old_last_date = pd.Timestamp(clean_df["date"].max()).date()
    target_end_date = end_date or date.today()
    next_start_date = old_last_date + timedelta(days=1)
    if next_start_date > target_end_date:
        return {
            "normalized_code": normalized_code,
            "ts_code": ts_code,
            "rows_added": 0,
            "old_last_date": old_last_date,
            "new_last_date": old_last_date,
            "file_path": file_path,
        }

    pro = ts.pro_api(resolved_token)
    daily_df = pro.daily(
        ts_code=ts_code,
        start_date=next_start_date.strftime("%Y%m%d"),
        end_date=target_end_date.strftime("%Y%m%d"),
        fields="ts_code,trade_date,open,high,low,close,vol,amount",
    )

    if daily_df is None or daily_df.empty:
        return {
            "normalized_code": normalized_code,
            "ts_code": ts_code,
            "rows_added": 0,
            "old_last_date": old_last_date,
            "new_last_date": old_last_date,
            "file_path": file_path,
        }

    daily_df = daily_df.sort_values("trade_date").reset_index(drop=True)
    append_df = _build_append_frame(raw_df, daily_df, normalized_code)
    date_col = _pick_existing_column(list(raw_df.columns), ["日期", "date", "交易日期"], "日期")

    merged_df = pd.concat([raw_df, append_df], ignore_index=True)
    merged_df["_date_key"] = pd.to_datetime(merged_df[date_col], errors="coerce")
    merged_df = (
        merged_df.dropna(subset=["_date_key"])
        .drop_duplicates(subset=["_date_key"], keep="last")
        .sort_values("_date_key")
        .drop(columns=["_date_key"])
        .reset_index(drop=True)
    )
    merged_df.to_csv(file_path, index=False, encoding="utf-8-sig")

    new_last_date = pd.Timestamp(pd.to_datetime(merged_df[date_col]).max()).date()
    return {
        "normalized_code": normalized_code,
        "ts_code": ts_code,
        "rows_added": int(len(append_df)),
        "old_last_date": old_last_date,
        "new_last_date": new_last_date,
        "file_path": file_path,
    }
