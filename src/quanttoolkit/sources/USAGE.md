## SEC 官方 companyfacts.zip 获取方法:

```bash
curl -H "User-Agent: 你的名字 你的邮箱" \
  -o data/companyfacts.zip \
  https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip
```

成员名单入库后, 从仓库根目录运行:

```bash
uv run python scripts/ingest_companyfacts.py
```

脚本读取 `universe` 的 CIK, 将带 `start` 的事实写入 `period_facts`, 其余写入 `instant_facts`
读取单批公司时:

```python
from quanttoolkit.sources.companyfacts import read_company_facts

period, instant = read_company_facts("data/companyfacts.zip", ["0000320193"])
```


## 官方映射表获取方法

```bash
curl -H "User-Agent: 你的名字 you@example.com" \
  -o data/company_tickers.json https://www.sec.gov/files/company_tickers.json
```

先下载映射文件, 再抓取最新标普500名单:

```python
from quanttoolkit.sources.wikipedia import fetch_sp500_universe

universe = fetch_sp500_universe(json_path="data/company_tickers.json")
```

返回普通 DataFrame, 列为 `ticker`, `name`, `sub_industry`, `cik`, `sec_title`, 按 ticker 升序排列
`cik` 为10位字符串, `sec_title` 使用 JSON 的 `cik_str` 对应 `title` 填充, 未匹配时为缺失值
映射文件必须事先下载, 文件不存在时抛出 `FileNotFoundError: 没有映射文件`, 并提示按上述方法获取

## 抓取到入库

先按上面的命令下载 `data/company_tickers.json`, 再从仓库根目录运行:

```bash
uv run python scripts/ingest_universe.py
```

脚本抓取最新成员并映射 SEC 名称, 写入 `data/quanttoolkit.db` 的 `universe` 表
首次运行自动建表, 再次运行完整替换成员名单, 写入过程使用事务, 失败时保留原数据
运行结束会显示抓取记录数, 入库记录数, SEC 名称缺失数和数据库路径

单独调用入库函数:

```python
from quanttoolkit.db.connection import DB, get_duckdb
from quanttoolkit.db.operation import SQL_DIR, operation_sql, read_sql
from quanttoolkit.sources.wikipedia import fetch_sp500_universe

universe = fetch_sp500_universe(json_path="data/company_tickers.json")
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

进度按处理完成的 ticker 数量推进, 包含失败任务, 不表示全部抓取成功
进度条只覆盖下载阶段, 不包括后续清洗和入库

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
