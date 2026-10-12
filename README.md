# QuantToolKit


[![CI](https://github.com/20070316lbw-netizen/QuantToolKit/actions/workflows/ci.yml/badge.svg)](https://github.com/20070316lbw-netizen/QuantToolKit/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/20070316lbw-netizen/QuantToolKit/branch/main/graph/badge.svg)](https://codecov.io/gh/20070316lbw-netizen/QuantToolKit)
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
| [db](src/quanttoolkit/db/README.md) | DuckDB 连接, universe 与 prices 建表和事务写入 | 已实现 |
| [sources](src/quanttoolkit/sources/USAGE.md) | 最新标普500成员抓取, SEC 公司名称映射, Yahoo 行情下载 | 已实现 |
| [transfer](src/quanttoolkit/transfer.py) | 时序长表转为 `[date, ticker]` MultiIndex DataFrame | 已实现 |
| [returns](docs/returns.md) | 历史与未来简单收益率, 单期对数收益率 | 已实现 |
| [volatility](docs/volatility.md) | 历史与未来简单收益波动率 | 已实现 |
| [rebalance](docs/rebalance.md) | 共同交易日历的调仓日期与区间归属 | 已实现 |
| [factors](docs/factors.md) | 动量与反转因子基础函数 | 已实现 |

标普500成员抓取到入库的完整流程见 [db 使用说明](src/quanttoolkit/db/README.md#抓取到入库)

计算接口统一使用 `[date, ticker]` 行情面板, 详见 [数据约定](docs/data-structure.md)。
离线示例: `uv run python scripts/example_price_metrics.py`。


## 许可证

本项目采用 [MIT License](LICENSE)。
