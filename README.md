# QuantToolKit


[![CI](https://github.com/20070316lbw-netizen/QuantToolKit/actions/workflows/ci.yml/badge.svg)](https://github.com/20070316lbw-netizen/QuantToolKit/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-%E2%89%A53.12-blue?logo=python&logoColor=white)](pyproject.toml)
[![License](https://img.shields.io/github/license/20070316lbw-netizen/QuantToolKit)](LICENSE)
[![Ruff](https://img.shields.io/badge/lint%20%26%20format-Ruff-D7FF64?logo=ruff&logoColor=white)](https://docs.astral.sh/ruff/)
[![Last commit](https://img.shields.io/github/last-commit/20070316lbw-netizen/QuantToolKit)](https://github.com/20070316lbw-netizen/QuantToolKit/commits/main/)


仓库版本的安装方式：

```bash
uv add "quanttoolkit @ git+https://github.com/20070316lbw-netizen/QuantToolKit.git"
```

## 模块目录

| 模块 | 内容 | 状态 |
| --- | --- | --- |
| [db](src/quanttoolkit/db/README.md) | DuckDB 连接, `sql` 目录中的建表与事务写入 | 已实现 |
| [sources](src/quanttoolkit/sources/USAGE.md) | 标普500成员, Kenneth French 日频因子, Yahoo 行情下载 | 已实现 |
| [panel](src/quanttoolkit/panel) | 长表转换, 价格补齐, [收益率](docs/returns.md), [波动率](docs/volatility.md) | 已实现 |
| [factors](docs/factors.md) | 动量, 反转, 对数波动率, RSI, 行业中性, 截面 z-score 与加权合成 | 已实现 |
| [portfolio](src/quanttoolkit/portfolio) | [调仓](docs/rebalance.md), [分位仓位](docs/sizing.md), [买入持有](docs/benchmark.md) | 已实现 |
| [metrics](docs/metrics.md) | 截面 IC, 信息比率, 夏普与最大回撤等评定指标 | 已实现 |
| [model](docs/model.md) | [样本](docs/dataset.md), [日期切分](docs/split.md), LightGBM 回归与 lambdarank 参数, JSON 训练日志 | 已实现 |
| [tracelog](docs/tracelog.md) | 函数耗时, 返回值摘要与调用链 | 已实现 |

标普500成员抓取到入库的完整流程见 [db 使用说明](src/quanttoolkit/db/README.md#抓取到入库)

计算接口统一使用 `[date, ticker]` 行情面板, 详见 [数据约定](docs/data-structure.md)。
离线示例: `uv run python scripts/example_price_metrics.py`。


## 许可证

本项目采用 [MIT License](LICENSE)。


## 介绍一下自己
liu 

邮箱: 20070316lbw@gmail.com

学校: Hunan University of Information Technology

Agent: Grok, DeepSeek, GPT
