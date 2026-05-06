from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd


COLUMN_ALIASES: Dict[str, str] = {
    "date": "date",
    "日期": "date",
    "datetime": "date",
    "交易日期": "date",
    "open": "open",
    "开盘": "open",
    "开盘价": "open",
    "high": "high",
    "最高": "high",
    "最高价": "high",
    "low": "low",
    "最低": "low",
    "最低价": "low",
    "close": "close",
    "收盘": "close",
    "收盘价": "close",
    "volume": "volume",
    "vol": "volume",
    "成交量": "volume",
    "总手": "volume",
    "代码": "code",
    "code": "code",
}


def normalize_stock_code(raw_code: str) -> str:
    code = raw_code.strip().lower()
    if not code:
        raise ValueError("股票代码不能为空。")

    if code.startswith(("sh", "sz", "bj")):
        return code

    if "." in code:
        body, market = code.split(".", 1)
        market = market.lower()
        market_prefix = {"sh": "sh", "sz": "sz", "bj": "bj"}.get(market)
        if market_prefix is None:
            raise ValueError(f"无法识别市场后缀: {raw_code}")
        return f"{market_prefix}{body}"

    if code.startswith(("6", "5")):
        return f"sh{code}"
    if code.startswith(("0", "3")):
        return f"sz{code}"
    if code.startswith(("4", "8", "9")):
        return f"bj{code}"
    raise ValueError(f"无法推断市场: {raw_code}")


def find_data_file(data_dir: Path, raw_code: str) -> Path:
    normalized = normalize_stock_code(raw_code)
    candidates = [
        data_dir / f"{normalized}.csv",
        data_dir / f"{normalized}.json",
        data_dir / f"{raw_code.strip().lower()}.csv",
        data_dir / f"{raw_code.strip().lower()}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"在 {data_dir} 未找到股票 {raw_code} 对应文件。"
        f"已尝试: {', '.join(str(p.name) for p in candidates)}"
    )


def _rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        key = str(col).strip()
        if key in COLUMN_ALIASES:
            rename_map[col] = COLUMN_ALIASES[key]
    return df.rename(columns=rename_map)


def load_stock_data(file_path: Path) -> pd.DataFrame:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(file_path)
    elif suffix == ".json":
        df = pd.read_json(file_path)
    else:
        raise ValueError(f"仅支持 CSV/JSON，收到: {file_path.name}")

    df = _rename_columns(df)
    required = {"date", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"数据缺少必需字段: {missing}")

    keep_cols = ["date", "open", "high", "low", "close", "volume"]
    if "code" in df.columns:
        keep_cols.append("code")
    df = df[keep_cols].copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    df = df[(df["open"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["close"] > 0)].copy()
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
    if df.empty:
        raise ValueError(f"{file_path.name} 清洗后为空。")
    return df
