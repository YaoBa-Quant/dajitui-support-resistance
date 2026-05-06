from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="基于形态特征训练 LightGBM 模型")
    parser.add_argument("--input-file", type=Path, default=None, help="特征标签文件，默认自动读取最新 feature_labels_*.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs", help="训练结果输出目录")
    parser.add_argument("--task", choices=["classification", "regression"], default="classification", help="任务类型")
    parser.add_argument("--horizon", type=int, default=20, help="目标周期，如 5/10/20/60")
    parser.add_argument("--train-end-date", type=str, default=None, help="训练集结束日期，如 2023-12-31")
    parser.add_argument("--valid-end-date", type=str, default=None, help="验证集结束日期，如 2024-12-31")
    parser.add_argument("--num-leaves", type=int, default=31, help="LightGBM num_leaves")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="LightGBM learning_rate")
    parser.add_argument("--n-estimators", type=int, default=300, help="LightGBM n_estimators")
    parser.add_argument("--min-data-in-leaf", type=int, default=50, help="LightGBM min_child_samples")
    parser.add_argument("--feature-fraction", type=float, default=0.9, help="LightGBM colsample_bytree")
    parser.add_argument("--bagging-fraction", type=float, default=0.9, help="LightGBM subsample")
    parser.add_argument("--bagging-freq", type=int, default=1, help="LightGBM subsample_freq")
    parser.add_argument("--random-state", type=int, default=42, help="随机种子")
    return parser


def _load_lightgbm():
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("未安装 lightgbm，请先执行 `pip install lightgbm` 或 `pip install -r requirements.txt`。") from exc
    return lgb


def _find_latest_feature_file(output_dir: Path) -> Path:
    candidates = sorted(
        [p for p in output_dir.glob("feature_labels_*.csv") if "quality" not in p.name.lower()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"在 {output_dir} 未找到 feature_labels_*.csv。")
    return candidates[0]


def _target_column(task: str, horizon: int) -> str:
    return f"label_up_{horizon}d" if task == "classification" else f"ret_fwd_{horizon}d"


def _safe_rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(mean_squared_error(y_true, y_pred) ** 0.5)


def _split_dates(
    dates: pd.Series,
    train_end_date: str | None,
    valid_end_date: str | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    unique_dates = pd.Index(sorted(pd.to_datetime(dates).dropna().unique()))
    if len(unique_dates) < 3:
        raise ValueError("可用交易日太少，无法切分训练/验证/测试集。")

    if train_end_date and valid_end_date:
        train_end = pd.to_datetime(train_end_date)
        valid_end = pd.to_datetime(valid_end_date)
        if not train_end < valid_end:
            raise ValueError("--train-end-date 必须早于 --valid-end-date。")
        return train_end, valid_end

    train_idx = max(0, int(len(unique_dates) * 0.7) - 1)
    valid_idx = max(train_idx + 1, int(len(unique_dates) * 0.85) - 1)
    valid_idx = min(valid_idx, len(unique_dates) - 2)
    train_end = pd.Timestamp(unique_dates[train_idx])
    valid_end = pd.Timestamp(unique_dates[valid_idx])
    if not train_end < valid_end:
        raise ValueError("自动时间切分失败，请手动指定 --train-end-date 和 --valid-end-date。")
    return train_end, valid_end


def _feature_columns(df: pd.DataFrame, target_col: str) -> list[str]:
    excluded = {"date", "code", target_col}
    cols: list[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        if col.startswith("ret_fwd_") or col.startswith("label_up_"):
            continue
        cols.append(col)
    return cols


def _cast_categorical(df: pd.DataFrame, cols: Iterable[str]) -> None:
    for col in cols:
        if col in df.columns:
            df[col] = df[col].astype("category")


def _classification_metrics(y_true: pd.Series, prob: pd.Series, pred: pd.Series) -> dict[str, float]:
    metrics = {
        "auc": float(roc_auc_score(y_true, prob)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "logloss": float(log_loss(y_true, prob)),
    }
    return metrics


def _regression_metrics(y_true: pd.Series, pred: pd.Series) -> dict[str, float]:
    metrics = {
        "rmse": _safe_rmse(y_true, pred),
        "mae": float(mean_absolute_error(y_true, pred)),
        "r2": float(r2_score(y_true, pred)),
    }
    return metrics


def main() -> None:
    args = build_parser().parse_args()
    lgb = _load_lightgbm()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    input_file = args.input_file or _find_latest_feature_file(args.output_dir)
    target_col = _target_column(args.task, args.horizon)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"读取数据: {input_file}")
    print(f"任务类型: {args.task}")
    print(f"目标列: {target_col}")

    df = pd.read_csv(input_file)
    if target_col not in df.columns:
        raise ValueError(f"输入文件缺少目标列: {target_col}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", target_col]).copy()
    if df.empty:
        raise ValueError("过滤空目标后无可用样本。")

    feature_cols = _feature_columns(df, target_col)
    categorical_cols = [col for col in ["pattern", "channel_pattern"] if col in feature_cols]
    _cast_categorical(df, categorical_cols)

    train_end, valid_end = _split_dates(df["date"], args.train_end_date, args.valid_end_date)
    train_df = df[df["date"] <= train_end].copy()
    valid_df = df[(df["date"] > train_end) & (df["date"] <= valid_end)].copy()
    test_df = df[df["date"] > valid_end].copy()
    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError("时间切分后训练/验证/测试集存在空集，请调整日期范围。")

    X_train = train_df[feature_cols].copy()
    X_valid = valid_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    y_train = train_df[target_col]
    y_valid = valid_df[target_col]
    y_test = test_df[target_col]

    common_params = dict(
        num_leaves=args.num_leaves,
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        min_child_samples=args.min_data_in_leaf,
        colsample_bytree=args.feature_fraction,
        subsample=args.bagging_fraction,
        subsample_freq=args.bagging_freq,
        random_state=args.random_state,
        objective="binary" if args.task == "classification" else "regression",
        verbose=-1,
    )

    if args.task == "classification":
        model = lgb.LGBMClassifier(**common_params)
        eval_metric = "binary_logloss"
    else:
        model = lgb.LGBMRegressor(**common_params)
        eval_metric = "l2"

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=eval_metric,
        categorical_feature=categorical_cols or "auto",
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=50),
        ],
    )

    if args.task == "classification":
        valid_prob = model.predict_proba(X_valid)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]
        valid_pred = (valid_prob >= 0.5).astype(int)
        test_pred = (test_prob >= 0.5).astype(int)
        valid_metrics = _classification_metrics(y_valid, valid_prob, valid_pred)
        test_metrics = _classification_metrics(y_test, test_prob, test_pred)
        pred_df = test_df[["date", "code", target_col]].copy()
        pred_df["pred_prob"] = test_prob
        pred_df["pred_label"] = test_pred
    else:
        valid_pred = model.predict(X_valid)
        test_pred = model.predict(X_test)
        valid_metrics = _regression_metrics(y_valid, valid_pred)
        test_metrics = _regression_metrics(y_test, test_pred)
        pred_df = test_df[["date", "code", target_col]].copy()
        pred_df["pred_value"] = test_pred

    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)

    summary_df = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "input_file": str(input_file),
                "task": args.task,
                "target_col": target_col,
                "train_end_date": train_end.date().isoformat(),
                "valid_end_date": valid_end.date().isoformat(),
                "train_rows": len(train_df),
                "valid_rows": len(valid_df),
                "test_rows": len(test_df),
                **{f"valid_{k}": v for k, v in valid_metrics.items()},
                **{f"test_{k}": v for k, v in test_metrics.items()},
            }
        ]
    )

    summary_path = args.output_dir / f"lgbm_summary_{args.task}_{args.horizon}d_{run_id}.csv"
    importance_path = args.output_dir / f"lgbm_importance_{args.task}_{args.horizon}d_{run_id}.csv"
    pred_path = args.output_dir / f"lgbm_test_pred_{args.task}_{args.horizon}d_{run_id}.csv"

    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")
    importance_df.to_csv(importance_path, index=False, encoding="utf-8-sig")
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    print("-" * 90)
    print(f"训练完成: {summary_path}")
    print(f"特征重要性: {importance_path}")
    print(f"测试集预测: {pred_path}")
    print(f"训练集行数: {len(train_df)} | 验证集行数: {len(valid_df)} | 测试集行数: {len(test_df)}")
    print("验证集指标:", valid_metrics)
    print("测试集指标:", test_metrics)


if __name__ == "__main__":
    main()
