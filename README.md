# QuantToolKit

[![CI](https://github.com/20070316lbw-netizen/QuantToolKit/actions/workflows/ci.yml/badge.svg)](https://github.com/20070316lbw-netizen/QuantToolKit/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.12-blue?logo=python&logoColor=white)](pyproject.toml)
[![License](https://img.shields.io/github/license/20070316lbw-netizen/QuantToolKit)](LICENSE)
[![Ruff](https://img.shields.io/badge/lint%20%26%20format-Ruff-D7FF64?logo=ruff&logoColor=white)](https://docs.astral.sh/ruff/)
[![Last commit](https://img.shields.io/github/last-commit/20070316lbw-netizen/QuantToolKit)](https://github.com/20070316lbw-netizen/QuantToolKit/commits/main/)

把个人量化研究里经常用到的工具放在一起: 获取行情, 存到 DuckDB, 计算因子, 构造仓位, 再接模型训练和评估。

## 安装与使用

在自己的项目中安装仓库版本:

```bash
uv add "quanttoolkit @ git+https://github.com/20070316lbw-netizen/QuantToolKit.git"
```

先看 [推荐调用顺序](docs/calling-guide.md), 把数据 → 因子 → 分数 → 调仓仓位串起来, 需要时再加模型打分。
各步骤用什么格式, 见 [数据约定](docs/data-structure.md)。

在本仓库运行离线示例: `uv run python scripts/example_price_metrics.py`。
想试模型训练, 评估和保存, 见 [模型说明](docs/model.md),
离线运行: `uv run python scripts/example_model_workflow.py`。

## 模块目录

| 模块 | 内容 |
| --- | --- |
| [db](src/quanttoolkit/db/README.md) | DuckDB 连接, SQL 建表与事务写入 |
| [sources](src/quanttoolkit/sources/USAGE.md) | 标普500成员, Kenneth French 日频因子, Yahoo 行情 |
| [panel](docs/data-structure.md) | 长表转换, 价格补齐, [收益率](docs/returns.md), [波动率](docs/volatility.md) |
| [factors](docs/factors.md) | 动量, 反转, 对数波动率, RSI, 行业中性, 截面标准化与加权合成 |
| [portfolio](docs/sizing.md) | [调仓日历](docs/rebalance.md), 分位仓位, [买入持有基准](docs/benchmark.md) |
| [metrics](docs/metrics.md) | 截面 IC 及汇总, 夏普, 最大回撤等指标 |
| [model](docs/model.md) | [样本组装](docs/dataset.md), [日期切分](docs/split.md), LightGBM 参数与排序训练, 手写回归树, 简化 GBDT, JSON 训练日志 |
| [tracelog](docs/tracelog.md) | 函数耗时, 返回值摘要与调用链 |

首次准备本地数据, 见 [数据源说明](src/quanttoolkit/sources/USAGE.md) 和 [数据库说明](src/quanttoolkit/db/README.md)。

## 许可证

本项目采用 [MIT License](LICENSE)。

## 介绍一下自己

liu

邮箱: 20070316lbw@gmail.com

学校: Hunan University of Information Technology

Agent: Grok, DeepSeek, GPT
