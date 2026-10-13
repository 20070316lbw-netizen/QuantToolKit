# db

提供 DuckDB 连接和事务写入, 支持最新标普500成员名单的 `universe` 表和日行情的 `prices` 表

| 文件 | 接口 | 职责 |
| --- | --- | --- |
| [connection.py](connection.py) | `get_duckdb(*, path, read_only=True)` | 打开数据库, 默认只读 |
| [connection.py](connection.py) | `DB` | 默认数据库路径 `data/quanttoolkit.db` |
| [operation.py](operation.py) | `read_sql(path)` | 读取 `sql` 目录中的语句 |
| [operation.py](operation.py) | `operation_sql(sql, ddl, connection, df=None)` | 在事务中先建表再写入, 返回最后一条语句的计数 |
| [sql](sql) | `*_ddl.sql`, `*.sql` | 建表语句和写入语句 |

## SQL 文件

语句都在 [sql](sql) 目录, 由 `SQL_DIR` 定位

| 文件 | 作用 |
| --- | --- |
| `universe_ddl.sql` | 创建 `universe` |
| `universe.sql` | 清空后按列名写入整份成员名单 |
| `prices_ddl.sql` | 创建 `prices`, 主键为 `(ticker, date)` |
| `prices.sql` | 按 `(ticker, date)` 新增或整行替换日行情 |

写入语句里的 `df` 指向 `operation_sql` 的同名参数
DuckDB 只在执行 `execute` 的函数里查找这个名字, 所以 DataFrame 必须传给 `operation_sql`, 不能只放在调用方的局部变量里

## universe 表

| 字段 | 数据库类型 | 内容 |
| --- | --- | --- |
| `ticker` | VARCHAR | 证券代码, 如 `AAPL`, `BRK-B` |
| `name` | VARCHAR | Wikipedia 成员表中的公司名称 |
| `sub_industry` | VARCHAR | Wikipedia 成员表中的 GICS 子行业 |
| `cik` | VARCHAR | 10位 SEC 公司标识, 保留前导零, 如 `0000320193` |
| `sec_title` | VARCHAR | SEC company_tickers.json 中按 CIK 匹配的 title, 未匹配时为 NULL |

数据源返回普通 DataFrame, 五个字段作为普通列, 按 ticker 升序排列
`universe` 保存最近一次写入的完整成员名单

## 抓取到入库

先按 [sources 获取方法](../sources/USAGE.md#官方映射表获取方法) 下载 `data/company_tickers.json`
映射文件不存在时, 抓取函数会先抛出 `FileNotFoundError: 没有映射文件`

从仓库根目录运行:

```bash
uv run python scripts/ingest_universe.py
```

[流程脚本](../../../scripts/ingest_universe.py) 读取本地映射文件, 抓取最新成员, 写入 `data/quanttoolkit.db`
运行结束显示抓取条数, 写入条数, 表中记录数, SEC 名称缺失数和数据库路径
脚本以自身位置确定仓库目录, 数据文件路径不依赖当前工作目录

## 读取数据库

```python
from quanttoolkit.db.connection import DB, get_duckdb

with get_duckdb(path=DB) as connection:
    members = connection.execute("SELECT * FROM universe ORDER BY ticker").df()

print(members.head())
```

`get_duckdb` 默认 `read_only=True`, 数据库不存在时抛出 `FileNotFoundError`
使用 `read_only=False` 可创建数据库及父目录, 连接由调用方通过 `with` 或 `close()` 关闭
建表由 `operation_sql` 在写入事务里执行

## 单独写入成员名单

下面的示例使用内存数据库, 可以独立执行:

```python
import duckdb
import pandas as pd

from quanttoolkit.db.operation import SQL_DIR, operation_sql, read_sql

universe = pd.DataFrame(
    [
        {
            "ticker": "AAPL",
            "name": "Apple Inc.",
            "sub_industry": "Technology Hardware, Storage & Peripherals",
            "cik": "0000320193",
            "sec_title": "Apple Inc.",
        }
    ]
)

with duckdb.connect(":memory:") as connection:
    written = operation_sql(
        read_sql(SQL_DIR / "universe.sql"),
        read_sql(SQL_DIR / "universe_ddl.sql"),
        connection,
        universe,
    )
    stored = connection.execute("SELECT COUNT(*) FROM universe").fetchone()[0]

print(written, stored)  # 1 1
```

调用时需传入可写连接, 且连接上没有未结束的事务
首次调用按 DDL 建表, 再次调用完整替换名单, 连续写入同一份数据保持相同记录数
空 DataFrame 会清空成员名单, 写入异常会回滚整个事务, 保留原有数据
DataFrame 列按名字写入, 缺失值写为 SQL NULL, 字段不匹配等错误由 DuckDB 抛出
其他数据表保留原样

## 写入日行情

`prices` 以 `(ticker, date)` 为主键, 输入为普通索引的长表 DataFrame, 列为
`date`, `ticker`, `open`, `high`, `low`, `close`, `adj_close`, `volume`, 列顺序不限
每个主键应对应一条输入记录, `date`, `ticker`, `close` 不可缺失
`date` 写为 SQL DATE, 不保留日内时间, 此表适用于日行情
价格和成交量写为 DOUBLE, 沿用输入单位和复权口径, 其余缺失值写为 SQL NULL

相同主键整行替换, 包括将原有值更新为 NULL, 新主键追加, 其他历史行情保留
返回值是最后一条写入语句的计数, 重复写入不会增加表中记录数
空表输入保留已有行情, 建表与写入在同一事务中完成, 失败时整体回滚

下载日行情再写入的完整调用见 [Yahoo 行情流程](../sources/USAGE.md#下载入库与计算)
`uv run python scripts/ingest_prices.py` 读取已入库的 `universe`, 下载日行情并写入 `prices`
它使用 `connection.DB` 定位数据库, 不依赖当前工作目录

下面的示例使用内存数据库, 可以独立执行:

```python
import duckdb
import pandas as pd

from quanttoolkit.db.operation import SQL_DIR, operation_sql, read_sql

prices = pd.DataFrame(
    [
        {
            "date": "2024-01-02",
            "ticker": "AAPL",
            "open": 185.0,
            "high": 188.0,
            "low": 184.0,
            "close": 187.0,
            "adj_close": 186.0,
            "volume": 1000.0,
        }
    ]
)

with duckdb.connect(":memory:") as connection:
    written = operation_sql(
        read_sql(SQL_DIR / "prices.sql"),
        read_sql(SQL_DIR / "prices_ddl.sql"),
        connection,
        prices,
    )
    prices["close"] = 188.0
    updated = operation_sql(
        read_sql(SQL_DIR / "prices.sql"),
        read_sql(SQL_DIR / "prices_ddl.sql"),
        connection,
        prices,
    )
    rows = connection.execute("SELECT ticker, close FROM prices").fetchall()

print(written, updated, rows)  # 1 1 [('AAPL', 188.0)]
```

## 本地检查

```bash
uv run pytest -q tests/test_operation.py tests/test_wikipedia.py tests/test_price_pipeline.py
uv run python -m doctest src/quanttoolkit/db/operation.py
uv run ruff check src/quanttoolkit/db/operation.py scripts/ingest_universe.py tests/test_operation.py
uv run ruff format --check src/quanttoolkit/db/operation.py scripts/ingest_universe.py tests/test_operation.py
```

测试覆盖文件持久化, 完整替换名单, 行情更新与追加, 重复写入, SQL NULL,
空输入, 主键及非空约束, 日内时间转换, 只读写入拒绝和事务失败后的回滚
GitHub Actions 生成 XML, HTML 和运行摘要中的覆盖率报告, 上传报告供下载并发送给 Codecov
首页 Coverage 徽章显示 Codecov 收到的 main 分支覆盖率
