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
| [transfer](src/quanttoolkit/transfer.py) | 时序长表转为 `[date, ticker]` MultiIndex DataFrame | 已实现 |
| [returns](docs/returns.md) | 历史与未来简单收益率, 单期对数收益率 | 已实现 |
| [volatility](docs/volatility.md) | 历史与未来简单收益波动率 | 已实现 |
| [rebalance](docs/rebalance.md) | 共同交易日历的调仓日期与区间归属 | 已实现 |
| [factors](docs/factors.md) | 动量, 反转, 对数波动率, RSI, 行业中性, 截面 z-score 与加权合成 | 已实现 |
| [split](docs/split.md) | 按交易日切分 train, valid, test, 并空出标签重叠的日期 | 已实现 |
| [sizing](docs/sizing.md) | 调仓日截面的分位多空权重 | 已实现 |
| [benchmark](docs/benchmark.md) | 首日等权买入并持有的组合收益 | 已实现 |
| [metrics](docs/metrics.md) | 截面 IC, 信息比率, 夏普与最大回撤等评定指标 | 已实现 |

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
