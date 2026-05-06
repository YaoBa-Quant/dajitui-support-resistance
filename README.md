# 支撑位 / 压力位分析与形态研究

基于“局部高低点 + 分位数回归”思路实现的量化研究工具，支持单票分析、全市场特征生成、形态/通道关系统计，以及 Streamlit 可视化展示。

## 功能概览

- 输入任意股票代码，如 `000001.SZ`、`600519.SH`、`sh600519`
- 自动加载并清洗本地 `CSV/JSON` 行情数据
- 计算最近 `N` 日支撑位、压力位、形态与通道特征
- 输出可视化趋势图与关键指标表
- 生成全市场逐日形态特征与未来收益标签
- 评估形态、通道、价格位置与未来 `5/10/20/60` 日收益关系
- 基于 LightGBM 进行基础训练实验
- 提供单元测试验证核心逻辑
- 


## 项目结构

- `app.py`：Streamlit 可视化界面入口
- `src/main.py`：命令行分析入口
- `src/build_feature_labels.py`：生成全市场形态特征与未来收益标签
- `src/analyze_feature_labels.py`：分析形态 / 通道 / 距离与未来收益关系
- `src/train_lgbm.py`：训练 LightGBM 模型
- `src/indicators.py`：核心指标、形态和通道识别逻辑
- `src/data_loader.py`：数据读取与清洗
- `tests/test_indicators.py`：核心测试

## 环境要求

- Python `3.11` 或 `3.12` 更推荐
- Windows / macOS / Linux

说明：

- 项目在较新的 Python 版本上也可能运行，但部分依赖（尤其 `lightgbm`）在新版本下安装兼容性可能较差

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据格式要求

支持 `CSV` 或 `JSON`，至少包含以下字段（中英文都可）：

- 日期 / `date`
- 开盘 / `open`
- 最高 / `high`
- 最低 / `low`
- 收盘 / `close`
- 成交量 / `volume`

示例文件名：

- `sh600519.csv`
- `sz000001.csv`
- `bj920000.csv`

默认读取目录为项目下的 `data/`，你也可以通过 `--data-dir` 自定义路径。

## 快速开始

### 1. 命令行分析

```bash
python src/main.py --code 600519.SH --window 120
```

指定历史分析截止日：

```bash
python src/main.py --code 603993.SH --window 240 --end-date 2026-03-31
```

交互模式：

```bash
python src/main.py
```

### 2. Streamlit 可视化界面

```bash
streamlit run app.py
```

### 3. 过去 5 年形态演化扫描

```bash
python src/main.py --code 603881.SH --window 240 --history-scan --output-dir outputs
```

输出内容包括：

- 最近 5 年逐日形态明细（CSV）
- 连续相同形态区间（CSV）
- 形态变化拐点（CSV）
- 向上突破倾向折线图（PNG）

## 全市场研究流程

### 1. 生成特征与标签

```bash
python src/build_feature_labels.py --window 240
```

默认会在 `outputs/` 下生成：

- `feature_labels_*.csv`
- `feature_labels_quality_*.csv`

### 2. 关系分析

```bash
python src/analyze_feature_labels.py --input-file outputs/feature_labels_xxx.csv
```

默认会输出：

- `relationship_summary_*.csv`
- `relationship_pattern_*.csv`
- `relationship_channel_*.csv`
- `relationship_bins_*.csv`

这些结果可用于研究：

- 形态与未来收益的关系
- 通道形态与未来收益的关系
- 价格在通道中的位置与未来收益的关系
- 价格距离支撑 / 压力边界与未来收益的关系

### 3. 模型训练实验

```bash
python src/train_lgbm.py
```

默认会输出：

- `lgbm_summary_*.csv`
- `lgbm_importance_*.csv`
- `lgbm_test_pred_*.csv`

## 常用参数

以 `src/main.py` 为例：

- `--code`：股票代码
- `--window`：最近 N 个交易日窗口，默认 `120`
- `--extrema-window`：局部高低点识别窗口，默认 `5`
- `--end-date`：分析截止日期，格式 `YYYY-MM-DD`
- `--history-scan`：回溯历史逐日扫描形态变化
- `--data-dir`：行情目录
- `--output-dir`：输出目录

## 算法说明

### 数据加载 `src/data_loader.py`

- 自动识别股票代码格式
- 自动匹配本地文件
- 统一字段名并清洗无效数据
- 去重、排序，并过滤价格异常值

### 指标计算 `src/indicators.py`

核心逻辑包括：

- 价格标准化（窗口首日 = `100`）
- 局部高低点识别（`argrelextrema`）
- 分位数回归拟合支撑线与压力线
- 形态识别：上升三角形、对称三角形、矩形、旗形、喇叭形等
- 通道识别：上升 / 下降 / 横盘 / 收敛 / 发散

当前输出中的 `breakout_up` / `breakout_down` 为启发式方向分数，更适合理解为“向上 / 向下突破倾向”，并不代表真实历史概率。

## 测试

```bash
pytest -q
```

当前测试主要覆盖：

- 局部高低点和支撑 / 压力区间逻辑
- 已知序列下关键指标是否符合预期

## 开源建议

如果你准备将本项目公开到 GitHub，通常建议：

- 不提交 `data/` 全量原始行情数据
- 不提交 `outputs/` 中的大体积研究结果与训练输出
- 不提交本地缓存、日志和个人素材

配套文档：

- `LICENSE`
- `.gitignore`
- `CONTRIBUTING.md`
- `DISCLAIMER.md`

## 免责声明

本项目仅用于技术研究、教学交流与开源学习，不构成任何投资建议，不承诺任何收益。

使用者应自行判断并承担相关风险。详细说明见 [DISCLAIMER.md](DISCLAIMER.md)。
