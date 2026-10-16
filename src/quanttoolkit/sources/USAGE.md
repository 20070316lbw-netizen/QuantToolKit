## Kenneth French 日频因子

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
官方文件里的数是百分数, 入库前除以 100, `0.0009` 表示 0.09%
三因子和五因子的 `smb` 不是同一条序列, 回归时按 `model` 分开用
`mkt_rf` 已经减去无风险利率, `rf` 是把月度国库券利率摊到每个交易日的简单收益率

单独读取时:

```python
from quanttoolkit.sources.french import read_french_factors

ff3 = read_french_factors("data/ff3_daily.zip")
ff5 = read_french_factors("data/ff5_daily.zip")
```

返回以 `date` 为索引的 DataFrame, 按日期升序, 单位是小数收益率


## 标普500成员

```python
from quanttoolkit.sources.wikipedia import fetch_sp500_universe

universe = fetch_sp500_universe()
```

返回普通 DataFrame, 列为 `ticker`, `name`, `sub_industry`, 按 ticker 升序排列
`ticker` 把页面上的点号换成连字符, 如 `BRK.B` 转为 `BRK-B`

## 抓取到入库

从仓库根目录运行:

```bash
uv run python scripts/ingest_universe.py
```

脚本抓取最新成员, 写入 `data/quanttoolkit.db` 的 `universe` 表
首次运行自动建表, 再次运行完整替换成员名单, 写入过程使用事务, 失败时保留原数据
运行结束会显示抓取记录数, 入库记录数和数据库路径

单独调用入库函数:

```python
from quanttoolkit.db.connection import DB, get_duckdb
from quanttoolkit.db.operation import SQL_DIR, operation_sql, read_sql
from quanttoolkit.sources.wikipedia import fetch_sp500_universe

universe = fetch_sp500_universe()
with get_duckdb(path=DB, read_only=False) as connection:
    written = operation_sql(
        read_sql(SQL_DIR / "universe.sql"),
        read_sql(SQL_DIR / "universe_ddl.sql"),
        connection,
        universe,
    )
```

## Yahoo 行情

`from quanttoolkit.sources import get_prices` 获取 `[date, ticker]` MultiIndex 行情面板。`tickers` 接受单个字符串或证券列表,
`start` 包含起始日期, `end` 不包含结束日期, 默认由 yfinance 决定结束时间。
`interval` 默认 `"1d"`, `auto_adjust` 默认 `False`, 参数透传给 yfinance。

输出通过 `to_date_ticker_frame` 转换, `date`, `ticker` 作为索引,
数据列为 `open`, `high`, `low`, `close`, `adj_close`, `volume`, 按 `[date, ticker]` 排序。日期去掉时区并保留源数据的当地时间; 数字字段转为数值,
无法转换的值设为缺失, `close` 缺失的记录删除。无数据或所有证券均缺失时返回保留列结构的空表。

`auto_adjust=False` 保留原始 close 和源数据提供的 adj_close; 如果源数据缺少复权价,
adj_close 保持缺失。只有 `auto_adjust=True` 且源数据未返回 Adj Close 时, 才用已复权 close
填充 adj_close。缺少某只证券时跳过该证券, 多证券响应缺少证券列层级时抛出 `ValueError`。
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

证券较多时按每批 40 只下载, 每批最多 4 个线程, 各批共用一个 Yahoo 会话, 批与批之间稍作停顿.
进度按该批已处理的 ticker 数量推进, 包含失败任务, 不表示全部抓取成功.
进度条只覆盖下载阶段, 不包括后续清洗和入库.

整段 `close` 都缺失的证券会再下载两次, 第二次起改为逐只下载. Yahoo 限流时, yfinance 常把空结果写成 `possibly delisted`, 重试就是为了把这些仍在交易的证券补回来. 多次下载后仍然没有 `close` 的证券会跳过, 并留下警告. 只缺一段时间的记录(例如上市前)会删掉这些行, 不会因此重下整段行情.

### 下载、入库与计算

下面的示例需要网络, 在仓库根目录执行。日行情入库应使用 `interval="1d"`,
因为数据库的 date 为 SQL DATE, 不保留日内时间。写入前用 `fill_price_gaps` 补齐每只证券首尾之间、其他证券已有成交而它缺失的交易日, 价格沿用前一条记录, volume 写 0。上市前和最后一条行情之后不补, 日历只用数据里已经出现的日期, 不用自然日。下载结果可直接用于价格计算, 写入数据库时用 `.reset_index()` 转为 `prices` 表对应的长表。

```python
from quanttoolkit.db import SQL_DIR, get_duckdb, operation_sql, read_sql
from quanttoolkit.db.connection import DB
from quanttoolkit.prices import fill_price_gaps
from quanttoolkit.returns import historical_return
from quanttoolkit.sources import get_prices
from quanttoolkit.transfer import to_date_ticker_frame

prices = fill_price_gaps(
    get_prices("AAPL", start="2024-01-01", end="2024-02-01", progress=False)
)
with get_duckdb(path=DB, read_only=False) as connection:
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
