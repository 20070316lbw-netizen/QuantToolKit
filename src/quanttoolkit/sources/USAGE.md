# 数据源使用说明

先选数据从哪里来: Yahoo 下载行情, Wikipedia 抓当前成员, French 读取本地因子 ZIP。
拿到后可以直接计算, 也可以按 [数据库说明](../db/README.md) 存起来。三个入口的返回格式不同, 先看下表:

| 数据 | 入口 | 返回格式 | 前提 |
| --- | --- | --- | --- |
| Yahoo 行情 | `get_prices` | `[date, ticker]` MultiIndex DataFrame | 需要网络 |
| 最新标普500成员 | `fetch_sp500_universe` | 普通列 DataFrame | 需要网络 |
| Kenneth French 日频因子 | `read_french_factors` | 以 `date` 为索引的 DataFrame | 已下载官方 ZIP |

仓库脚本在仓库根目录运行; 外部项目直接导入上述接口, 并传入自己的文件或数据库路径。
完整的下游调用见 [推荐调用顺序](../../../docs/calling-guide.md)。

## Kenneth French 日频因子

先下载 FF3 和 FF5 的日频压缩包。下面的命令在仓库根目录执行, 需要网络:

```bash
curl -L -o data/ff3_daily.zip \
  https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_daily_CSV.zip

curl -L -o data/ff5_daily.zip \
  https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip
```

两个压缩包放到 `data/` 后, 从仓库根目录运行:

```bash
uv run python scripts/ingest_french.py
```

脚本把两份日频因子写入 `ff_factors`, `model` 为 `ff3` 或 `ff5`
三因子列为 `mkt_rf`, `smb`, `hml`, `rf`, 五因子再加 `rmw`, `cma`
五因子行才有 `rmw` 和 `cma`, 三因子这两列为空
读取函数将官方百分数除以 100, 返回和入库值都是小数收益率, `0.0009` 表示 0.09%。
三因子和五因子的 `smb` 不是同一条序列, 回归时应按 `model` 分开使用。
`mkt_rf` 已减去无风险利率, `rf` 是日频无风险收益率。

单独读取时:

```python
from quanttoolkit.sources.french import read_french_factors

ff3 = read_french_factors("data/ff3_daily.zip")
ff5 = read_french_factors("data/ff5_daily.zip")
```

返回以 `date` 为索引、按日期升序的 DataFrame, 不带 ticker 层。
读取函数只解析本地压缩包, 下载和入库分别由前面的命令和脚本完成。

## 标普500成员

```python
from quanttoolkit.sources.wikipedia import fetch_sp500_universe

universe = fetch_sp500_universe()
```

返回普通 DataFrame, 列为 `ticker`, `name`, `sub_industry`, 按 ticker 升序排列。
证券代码中的点号替换为连字符, 如 `BRK.B` 转为 `BRK-B`。
它是抓取时的当前名单, 不包含历史进出记录, 不能直接当作历史时点股票池。

## 抓取到入库

从仓库根目录运行:

```bash
uv run python scripts/ingest_universe.py
```

脚本抓取最新成员, 写入仓库 `data/quanttoolkit.db` 的 `universe` 表。
首次运行自动建表, 再次运行完整替换名单, 失败时通过事务保留原数据。
运行结束会显示抓取记录数、入库记录数和数据库路径。

在自己的项目中直接调用, 并显式指定自己的数据库路径。下例需要网络, 会写入数据库:

```python
from pathlib import Path

from quanttoolkit.db.connection import get_duckdb
from quanttoolkit.db.operation import SQL_DIR, operation_sql, read_sql
from quanttoolkit.sources.wikipedia import fetch_sp500_universe

universe = fetch_sp500_universe()
with get_duckdb(path=Path("data/market.db"), read_only=False) as connection:
    written = operation_sql(
        read_sql(SQL_DIR / "universe.sql"),
        read_sql(SQL_DIR / "universe_ddl.sql"),
        connection,
        universe,
    )
```

## Yahoo 行情

`from quanttoolkit.sources import get_prices` 获取已排序的 `[date, ticker]` 行情面板,
无需再次调用 `to_date_ticker_frame`。

| 参数 | 默认值与含义 |
| --- | --- |
| `tickers` | 必填, 单个证券字符串或非空证券列表 |
| `start` | 必填, 包含起始日期 |
| `end` | 默认 `None`, 不包含结束日期, 未指定时由 yfinance 决定 |
| `interval` | 默认 `"1d"`, 原样传给 yfinance |
| `auto_adjust` | 默认 `False`, 是否由 yfinance 调整行情价格 |
| `progress` | 默认 `True`, 显示每批下载进度 |

输出通过 `to_date_ticker_frame` 转换, `date`, `ticker` 作为索引,
数据列为 `open`, `high`, `low`, `close`, `adj_close`, `volume`, 按 `[date, ticker]` 排序。日期去掉时区并保留源数据的当地时间; 数字字段转为数值,
无法转换的值设为缺失, `close` 缺失的记录删除。无数据或所有证券均缺失时返回保留列结构的空表。

`auto_adjust=False` 保留原始 close 和源数据提供的 adj_close; 如果源数据缺少复权价,
adj_close 保持缺失。只有 `auto_adjust=True` 且源数据未返回 Adj Close 时, 才用已复权 close
填充 adj_close。缺少某只证券时跳过该证券, 多只证券的返回数据里没有 ticker 列层级时抛出 `ValueError`。
证券列表应非空, 下载调用抛出的异常继续传播。

### 下载进度

`get_prices` 默认显示 yfinance 下载进度条, 用关键词参数 `progress=False` 可关闭:

```python
from quanttoolkit.sources import get_prices

prices = get_prices(
    tickers=["AAPL", "MSFT"],
    start="2024-01-01",
    end="2024-02-01",
    progress=False,
)
```

证券较多时按每批 40 只下载, 首次下载每批最多 4 个线程, 各批共用一个 Yahoo 会话,
每批之间稍作停顿。进度按该批已处理的 ticker 数量推进, 包含失败任务,
覆盖下载阶段, 不包括清洗和入库。

整段 `close` 缺失的证券最多再下载两次, 重试时逐只下载。
限流也可能使 yfinance 输出 `possibly delisted`, 仅凭这条提示不能认定证券已退市。
多次下载后仍没有 `close` 的证券会被跳过并记录警告。
仅部分日期缺失时, 删除这些缺失行, 不重下整段行情。

### 下载、入库与计算

下面的示例需要网络, 会写入自己项目下的 `data/market.db`。
顺序是下载 → 补齐内部缺行 → 转成长表入库 → 读回面板 → 计算。

日行情入库应使用 `interval="1d"`, 因为数据库的 date 为 SQL DATE, 不保留日内时间。
`fill_price_gaps` 只补每只证券首尾记录之间、输入共同日历中已有的日期,
新增行沿用此前可用价格, volume 为 0。上市前和末条记录之后不补, 已有行中的 NaN 也不修补。
是否补齐要自己决定, 具体规则见 [数据约定](../../../docs/data-structure.md)。

```python
from pathlib import Path

from quanttoolkit.db import SQL_DIR, get_duckdb, operation_sql, read_sql
from quanttoolkit.panel.prices import fill_price_gaps
from quanttoolkit.panel.returns import historical_return
from quanttoolkit.panel.transfer import to_date_ticker_frame
from quanttoolkit.sources import get_prices

prices = fill_price_gaps(
    get_prices("AAPL", start="2024-01-01", end="2024-02-01", progress=False)
)
with get_duckdb(path=Path("data/market.db"), read_only=False) as connection:
    written = operation_sql(
        read_sql(SQL_DIR / "prices.sql"),
        read_sql(SQL_DIR / "prices_ddl.sql"),
        connection,
        prices.reset_index(),
    )
    stored = connection.execute("SELECT * FROM prices ORDER BY date, ticker").df()

prices = to_date_ticker_frame(df=stored)
returns = historical_return(df=prices, n_periods=5)
print(written, returns.tail())
```

[`scripts/ingest_prices.py`](../../../scripts/ingest_prices.py) 读取 `universe` 中的代码, 下载日行情并写入 `prices`。独立计算的离线示例见 [收益率文档](../../../docs/returns.md)。

测试用模拟下载响应校验清洗及下游流程, 不依赖实时 Yahoo 服务。
