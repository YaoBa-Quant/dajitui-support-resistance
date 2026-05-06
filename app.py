from __future__ import annotations

import base64
from datetime import date
from pathlib import Path
from tempfile import NamedTemporaryFile
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
WECHAT_QR = ROOT / "wechat_qr.jpg"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from data_loader import find_data_file, load_stock_data, normalize_stock_code
from indicators import calculate_support_resistance
from tushare_updater import update_stock_data_via_tushare
from visualize import plot_support_resistance


def _default_data_dir() -> Path:
    return ROOT / "data"


def _result_table(normalized_code: str, result: dict) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"指标": "股票代码", "值": normalized_code},
            {"指标": "压力线斜率", "值": f"{result['pressure_slope']:.6f}"},
            {"指标": "支撑线斜率", "值": f"{result['support_slope']:.6f}"},
            {"指标": "持续形态", "值": result["pattern"]},
            {"指标": "通道形态", "值": result["channel_pattern"]},
            {"指标": "向上突破概率", "值": f"{result['breakout_up']:.2%}"},
            {"指标": "向下跌破概率", "值": f"{result['breakout_down']:.2%}"},
            {"指标": "方向强度", "值": f"{result['strength']:.2%}"},
        ]
    )


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .dajitui-hero {
            position: relative;
            margin: 0.55rem 0 1.1rem;
            padding: 1.05rem 1.15rem 1.1rem;
            border-radius: 22px;
            background:
                radial-gradient(circle at top right, rgba(255, 255, 255, 0.28), transparent 28%),
                linear-gradient(135deg, #111827 0%, #1d4ed8 34%, #7c3aed 68%, #f97316 100%);
            color: #ffffff;
            box-shadow: 0 18px 36px rgba(37, 99, 235, 0.16);
            overflow: hidden;
        }
        .dajitui-hero::after {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(120deg, rgba(255, 255, 255, 0.08), transparent 42%, rgba(255, 255, 255, 0.05) 72%, transparent);
            pointer-events: none;
        }
        .dajitui-hero-badge {
            position: relative;
            z-index: 1;
            width: fit-content;
            padding: 0.26rem 0.66rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.16);
            font-size: 0.74rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            margin-bottom: 0.72rem;
        }
        .dajitui-hero-title {
            position: relative;
            z-index: 1;
            font-size: 1.16rem;
            font-weight: 900;
            line-height: 1.4;
            margin-bottom: 0.28rem;
        }
        .dajitui-hero-subtitle {
            position: relative;
            z-index: 1;
            color: rgba(255, 255, 255, 0.88);
            font-size: 0.93rem;
            line-height: 1.65;
        }
        .dashboard-card {
            position: relative;
            display: flex;
            min-height: 118px;
            padding: 0.95rem 0.95rem 0.9rem;
            border-radius: 20px;
            color: #ffffff !important;
            text-decoration: none !important;
            box-shadow: 0 16px 30px rgba(15, 23, 42, 0.12);
            transition: transform 0.18s ease, box-shadow 0.18s ease, filter 0.18s ease;
            overflow: hidden;
        }
        .dashboard-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 22px 34px rgba(15, 23, 42, 0.18);
            filter: saturate(1.04);
        }
        .dashboard-card::after {
            content: "";
            position: absolute;
            inset: 0;
            background:
                radial-gradient(circle at top right, rgba(255, 255, 255, 0.26), transparent 34%),
                linear-gradient(135deg, rgba(255, 255, 255, 0.08), transparent 58%);
            pointer-events: none;
        }
        .dashboard-card-inner {
            position: relative;
            z-index: 1;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
            width: 100%;
            gap: 0.55rem;
        }
        .dashboard-card-tag {
            width: fit-content;
            padding: 0.18rem 0.52rem;
            border-radius: 999px;
            background: rgba(255, 255, 255, 0.16);
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.04em;
        }
        .dashboard-card-title {
            font-size: 0.98rem;
            font-weight: 800;
            line-height: 1.35;
        }
        .dashboard-card-url {
            font-size: 0.77rem;
            line-height: 1.45;
            color: rgba(255, 255, 255, 0.88);
            word-break: break-word;
        }
        .dashboard-card-blue {
            background: linear-gradient(135deg, #2563eb 0%, #0f172a 100%);
        }
        .dashboard-card-teal {
            background: linear-gradient(135deg, #0f766e 0%, #2dd4bf 100%);
        }
        .dashboard-card-purple {
            background: linear-gradient(135deg, #9333ea 0%, #ec4899 100%);
        }
        .dashboard-card-orange {
            background: linear-gradient(135deg, #d97706 0%, #f97316 100%);
        }
        .update-panel {
            margin-top: 0.75rem;
            padding: 1rem 0.95rem 1.1rem;
            border-radius: 18px;
            background: linear-gradient(180deg, #fff6e8 0%, #fdeecf 100%);
            border: 1px solid rgba(228, 177, 73, 0.35);
            box-shadow: 0 10px 22px rgba(137, 99, 20, 0.08);
        }
        .update-panel-title {
            font-size: 1.05rem;
            font-weight: 800;
            color: #513317;
            margin-bottom: 0.7rem;
        }
        .update-date-card {
            background: rgba(255, 255, 255, 0.82);
            border-radius: 14px;
            padding: 0.8rem 0.9rem;
            border: 1px solid rgba(228, 177, 73, 0.28);
            margin-bottom: 0.75rem;
        }
        .update-date-label {
            font-size: 0.82rem;
            color: #8b6a37;
            margin-bottom: 0.15rem;
        }
        .update-date-value {
            font-size: 1.1rem;
            font-weight: 800;
            color: #34210f;
            letter-spacing: 0.02em;
        }
        .update-helper {
            font-size: 0.8rem;
            color: #8b6a37;
            margin: 0.45rem 0 0.6rem;
            line-height: 1.55;
        }
        .feedback-block {
            margin-top: 1rem;
            padding-top: 0.9rem;
            border-top: 1px dashed rgba(144, 110, 42, 0.25);
        }
        .feedback-title {
            font-size: 1rem;
            font-weight: 800;
            color: #26314d;
            margin-bottom: 0.8rem;
        }
        .feedback-card {
            background: #ffffff;
            border-radius: 18px;
            padding: 0.9rem 0.85rem 1rem;
            border: 1px solid rgba(16, 185, 129, 0.16);
            box-shadow: 0 12px 24px rgba(16, 24, 40, 0.08);
            text-align: center;
        }
        .feedback-card img {
            width: 100%;
            max-width: 240px;
            border-radius: 10px;
            display: block;
            margin: 0 auto;
        }
        .feedback-note {
            margin-top: 0.65rem;
            color: #515b6e;
            font-size: 0.92rem;
        }
        .feedback-author {
            margin-top: 0.85rem;
            color: #2f3a4d;
            font-size: 0.95rem;
        }
        .feedback-author code {
            background: #f4f7fb;
            color: #2c6bed;
            padding: 0.08rem 0.35rem;
            border-radius: 6px;
            font-size: 0.88rem;
        }
        .feedback-subtitle {
            margin-top: 0.75rem;
            color: #314158;
            font-size: 0.95rem;
            line-height: 1.6;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_dashboard_links() -> None:
    dashboards = [
        ("全天候量化看板", "https://all-weather.dajitui.vip", "DAJITUI ALPHA", "dashboard-card-blue"),
        ("网格交易看板", "https://grid.dajitui.vip", "DAJITUI GRID", "dashboard-card-teal"),
        ("动量轮动看板", "https://momentum.dajitui.vip", "DAJITUI MOM", "dashboard-card-purple"),
        ("低波长持看板", "https://longhold.dajitui.vip/", "DAJITUI LOWVOL", "dashboard-card-orange"),
    ]
    st.divider()
    st.subheader("大鸡腿策略系列")
    st.caption("支撑压力分析系统与全天候、网格、动量、低波长持看板共同组成大鸡腿系列产品矩阵。")
    cols = st.columns(4, gap="small")
    for col, (title, url, tag, css_class) in zip(cols, dashboards):
        with col:
            st.markdown(
                f"""
                <a class="dashboard-card {css_class}" href="{url}" target="_blank" rel="noopener noreferrer">
                    <div class="dashboard-card-inner">
                        <div class="dashboard-card-tag">{tag}</div>
                        <div class="dashboard-card-title">{title}</div>
                        <div class="dashboard-card-url">{url}</div>
                    </div>
                </a>
                """,
                unsafe_allow_html=True,
            )


def _render_brand_banner() -> None:
    st.markdown(
        """
        <div class="dajitui-hero">
            <div class="dajitui-hero-badge">DAJITUI SERIES</div>
            <div class="dajitui-hero-title">大鸡腿系列 · 技术分析与策略看板矩阵</div>
            <div class="dajitui-hero-subtitle">
                当前系统聚焦支撑位 / 压力位与形态通道分析，下方同步接入全天候量化、网格交易、动量轮动、低波长持四套看板入口。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_feedback_card() -> None:
    if not WECHAT_QR.exists():
        st.caption(f"二维码未找到：{WECHAT_QR.name}")
        return
    encoded = base64.b64encode(WECHAT_QR.read_bytes()).decode("ascii")
    st.markdown(
        f"""
        <div class="feedback-block">
            <div class="feedback-title">🤝 交流与反馈</div>
            <div class="feedback-card">
                <img src="data:image/jpeg;base64,{encoded}" alt="微信二维码" />
                <div class="feedback-note">扫码加好友(备注: 动量)</div>
                <div class="feedback-author">🙋 作者微信: <code>Code_Mvp</code></div>
                <div class="feedback-subtitle">获取最新策略代码 · 探讨量化思路</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _try_load_by_code(stock_code: str) -> tuple[str | None, Path | None, pd.DataFrame | None, str | None]:
    data_dir = _default_data_dir()
    try:
        normalized = normalize_stock_code(stock_code)
        file_path = find_data_file(data_dir, stock_code)
        df = load_stock_data(file_path)
        return normalized, file_path, df, None
    except Exception as exc:
        return None, None, None, str(exc)


def main() -> None:
    st.set_page_config(page_title="支撑位/压力位可视化", page_icon=":chart_with_upwards_trend:", layout="wide")
    _inject_styles()
    st.title("支撑位 / 压力位趋势图")
    st.caption("输入股票代码后，页面直接展示趋势图与关键统计。")
    _render_brand_banner()
    if "flash_message" in st.session_state:
        level, message = st.session_state.pop("flash_message")
        getattr(st, level)(message)

    normalized: str | None = None
    file_path: Path | None = None
    df: pd.DataFrame | None = None
    load_error: str | None = None

    with st.sidebar:
        st.subheader("参数")
        stock_code = st.text_input("股票代码", value="sh688188", help="支持 000001.SZ / 600519.SH / sh600519")
        normalized, file_path, df, load_error = _try_load_by_code(stock_code)
        min_end_date = None
        max_end_date = None
        current_data_date = None
        window = st.slider("回看窗口 (N日)", min_value=30, max_value=300, value=120, step=10)
        if df is not None:
            current_data_date = pd.Timestamp(df["date"].max()).date()
            if len(df) >= window:
                min_end_date = pd.Timestamp(df["date"].iloc[window - 1]).date()
            max_end_date = current_data_date
            if min_end_date is not None:
                st.caption(f"可选截止日: {min_end_date.isoformat()} 至 {max_end_date.isoformat()}")
            else:
                st.caption(f"数据不足 {window} 个交易日，暂不能按截止日分析。")
        elif load_error:
            st.caption(f"数据状态: {load_error}")
        extrema_window = st.slider("局部高低点窗口", min_value=2, max_value=20, value=5, step=1)
        use_end_date = st.checkbox("启用分析截止日", value=False)
        end_date_kwargs: dict[str, object] = {"disabled": not use_end_date}
        if min_end_date is not None and max_end_date is not None:
            end_date_kwargs["value"] = max_end_date
            end_date_kwargs["min_value"] = min_end_date
            end_date_kwargs["max_value"] = max_end_date
        elif max_end_date is not None:
            end_date_kwargs["value"] = max_end_date
            end_date_kwargs["disabled"] = True
        else:
            end_date_kwargs["value"] = date.today()
        end_date = st.date_input("分析截止日", **end_date_kwargs)
        run_clicked = st.button("开始分析", type="primary", use_container_width=True)
        latest_date_text = current_data_date.isoformat() if current_data_date is not None else "暂无数据"
        st.markdown(
            f"""
            <div class="update-panel">
                <div class="update-panel-title">数据更新</div>
                <div class="update-date-card">
                    <div class="update-date-label">目前数据日期</div>
                    <div class="update-date-value">{latest_date_text}</div>
                </div>
                <div class="update-helper">通过 Tushare 拉取最新日线并更新本地 CSV。更新成功后页面会自动刷新。</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        tushare_token = st.text_input(
            "Tushare Token",
            value="",
            type="password",
            help="留空时尝试读取环境变量 TUSHARE_TOKEN。",
        )
        update_clicked = st.button(
            "更新数据",
            use_container_width=True,
            disabled=file_path is None,
            help="通过 Tushare 拉取最新日线并更新本地 CSV。",
        )
        _render_feedback_card()

    if update_clicked:
        try:
            if file_path is None:
                raise ValueError(load_error or "未找到对应股票数据文件。")
            update_info = update_stock_data_via_tushare(file_path, stock_code, token=tushare_token or None)
            if update_info["rows_added"] > 0:
                st.session_state["flash_message"] = (
                    "success",
                    f"更新完成：新增 {update_info['rows_added']} 条，数据日期已更新到 {update_info['new_last_date']}",
                )
            else:
                st.session_state["flash_message"] = (
                    "info",
                    f"当前已是最新数据，最新日期为 {update_info['new_last_date']}",
                )
            st.rerun()
        except Exception as exc:
            st.error(f"数据更新失败：{exc}")
        return

    if not run_clicked:
        st.info("左侧填写参数后点击“开始分析”。")
        _render_dashboard_links()
        return

    try:
        output_dir = ROOT / "outputs"
        if df is None or normalized is None or file_path is None:
            raise ValueError(load_error or "未找到对应股票数据。")

        if use_end_date:
            end_dt = pd.to_datetime(end_date)
            df = df[df["date"] <= end_dt].reset_index(drop=True)
            if df.empty:
                raise ValueError(f"截止到 {end_date} 无可用数据。")

        result = calculate_support_resistance(df, n_window=window, extrema_window=extrema_window)

        with NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            chart_path = Path(tmp.name)
        plot_support_resistance(result["recent_df"], result, normalized, chart_path)

        left, right = st.columns([2, 1], gap="large")
        with left:
            st.image(str(chart_path), caption=f"{normalized} 趋势图", use_container_width=True)
        with right:
            st.dataframe(_result_table(normalized, result), use_container_width=True, hide_index=True)

        st.success(f"分析完成：数据文件 {file_path.name}")
    except Exception as exc:
        st.error(f"分析失败：{exc}")
    _render_dashboard_links()


if __name__ == "__main__":
    main()
