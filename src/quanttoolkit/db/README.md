# 数据库连接与写入

先打开 DuckDB 连接, 再读取数据, 或把建表和写入放进同一次事务执行。
数据从哪里来, 见 [数据源说明](../sources/USAGE.md); 读出来后怎么计算, 见 [推荐调用顺序](../../../docs/calling-guide.md)。

| 文件 | 接口 | 做什么 |
| --- | --- | --- |
| [connection.py](connection.py) | `get_duckdb(*, path, read_only=True)` | 打开数据库, 默认只读 |
| [connection.py](connection.py) | `DB` | 按本包所在位置推导的 `data/quanttoolkit.db` |
| [operation.py](operation.py) | `read_sql(sql)` | 读取 SQL 文件文本 |
| [operation.py](operation.py) | `operation_sql(sql, ddl, connection, df=None)` | 在事务中先建表再写入, 返回最后一条语句的计数 |
| [sql](sql) | `*_ddl.sql`, `*.sql` | 建表语句和写入语句 |

`DB` 供本仓库脚本使用, 不会自动指向依赖项目的数据目录。
外部项目调用 `get_duckdb` 时应显式传入自己的路径。

| 数据对象 | 关键字段 | 再次写入时的行为 |
| --- | --- | --- |
| `universe` | `ticker`, `name`, `sub_industry` | 清空后替换完整名单 |
| `prices` | 主键 `(ticker, date)` | 同键整行替换, 新键追加 |
| `ff_factors` | 主键 `(model, date)` | 清空后替换本次全部因子 |
| `factor_panel` | `date`, `ticker` + 固定因子列 | 视图, 随底层表查询结果变化 |

## SQL 文件

语句都在 [sql](sql) 目录, 由 `SQL_DIR` 定位。

| 文件 | 作用 |
| --- | --- |
| `universe_ddl.sql` | 创建 `universe` |
| `universe.sql` | 清空后按列名写入整份成员名单 |
| `prices_ddl.sql` | 创建 `prices`, 主键为 `(ticker, date)` |
| `prices.sql` | 按 `(ticker, date)` 新增或整行替换日行情 |
| `ff_factors_ddl.sql` | 创建 `ff_factors`, 主键为 `(model, date)` |
| `ff_factors.sql` | 清空后按列名写入 FF3 与 FF5 日频因子 |
| `factor_panel.sql` | 创建或替换日频因子视图 |

写入语句中的 `df` 由 `operation_sql` 的同名参数提供,
调用时把 DataFrame 传进去, 函数先建表再写入, 成功就提交, 失败就回滚。

## universe 表

| 字段 | 数据库类型 | 内容 |
| --- | --- | --- |
| `ticker` | VARCHAR | 证券代码, 如 `AAPL`, `BRK-B` |
| `name` | VARCHAR | Wikipedia 成员表中的公司名称 |
| `sub_industry` | VARCHAR | Wikipedia 成员表中的 GICS 子行业 |

数据源返回普通 DataFrame, 三个字段作为普通列, 按 ticker 升序排列。
`universe` 保存最近一次写入的完整成员名单, 不保留成员变更历史。

## 抓取到入库

从仓库根目录运行:

```bash
uv run python scripts/ingest_universe.py
```

[流程脚本](../../../scripts/ingest_universe.py) 抓取最新成员, 写入 `data/quanttoolkit.db`
运行结束显示抓取条数, 写入条数, 表中记录数和数据库路径
脚本以自身位置确定仓库目录, 数据文件路径不依赖当前工作目录

## 读取数据库

下面假定自己的 `data/market.db` 已有 `universe` 表, 只读查询不会创建数据库:

```python
from pathlib import Path

from quanttoolkit.db.connection import get_duckdb

with get_duckdb(path=Path("data/market.db")) as connection:
    members = connection.execute("SELECT * FROM universe ORDER BY ticker").df()

print(members.head())
```

`get_duckdb` 默认 `read_only=True`, 数据库不存在时抛出 `FileNotFoundError`
使用 `read_only=False` 可创建数据库及父目录, 连接由调用方通过 `with` 或 `close()` 关闭
建表由 `operation_sql` 在写入事务里执行。
行情查询得到的普通长表还需用 `to_date_ticker_frame(df=raw)` 转为计算面板,
静态成员名单保留普通列即可。

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

assert written == stored == 1
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
`uv run python scripts/ingest_prices.py` 读取已入库的 `universe`, 下载日行情, 补齐每只证券首尾之间的缺失交易日后写入 `prices`
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

assert written == updated == 1
assert rows == [("AAPL", 188.0)]
print(written, updated, rows)  # 1 1 [('AAPL', 188.0)]
```

## 写入 French 因子

`ff_factors` 保存 Kenneth French 的 FF3 与 FF5 日频因子收益
主键是 `(model, date)`, `model` 取 `ff3` 或 `ff5`
列为 `mkt_rf`, `smb`, `hml`, `rmw`, `cma`, `rf`, 单位是小数收益率
`rmw` 和 `cma` 只在 `ff5` 有值
每次写入先清空全表, 再写入这次传入的全部行

先按 [sources 获取方法](../sources/USAGE.md#kenneth-french-日频因子) 放好两个压缩包, 从仓库根目录运行:

```bash
uv run python scripts/ingest_french.py
```

## 因子视图

`factor_panel` 将 FF5 与固定的一组日频价格、成交量因子放在同一视图中,
定义见 [factor_panel.sql](sql/factor_panel.sql)。SQL 查询结果仍是普通长表,
经过 `to_date_ticker_frame` 才得到 `[date, ticker]` MultiIndex。

视图从 `prices` 的第 64 个不同交易日起保留记录, 并要求当天有 FF5 数据。
从这一天开始, 是因为 `mom_63` 至少需要 64 条价格。不过各只证券的历史不一定一样长:
较晚上市或历史不足的证券仍可出现, 对应因子列为空。

| 列 | 口径 |
| --- | --- |
| `mkt_rf`, `smb`, `hml`, `rmw`, `cma`, `rf` | 当天的 FF5, 小数收益率 |
| `mom_21`, `mom_63` | 与 `base_momfactor` 相同, 用 `adj_close` 的累计对数收益 |
| `logvol_21` | 与 `base_logvolfactor` 相同, 样本标准差 `ddof=1` |
| `overnight` | `ln(open / 前收盘)`, 当天 `volume` 为 0 时为空 |
| `intraday` | `ln(close / open)`, 当天 `volume` 为 0 时为空 |
| `dvol` | `ln(当天成交量 / 前一交易日成交量)`, 两天里有一天 `volume` 为 0 时为空 |
| `rel_volume_21` | 当天成交量除以含当天的最近 21 条记录成交量均值, 窗口必须完整且 volume 非空、非零 |

隔夜因子使用 `open` 与前条 `close`, 日内因子使用 `close/open`, 二者使用行情的 OHLC 口径。
动量和对数波动率使用 `adj_close`, 与 Python 基础因子保持一致。
补齐价格仍参与价格窗口, `volume=0` 会转为缺失, 不进入完整的成交量窗口。
FF5 在同一天对所有证券相同, 这几列是市场时间序列, 不提供当天的证券截面差异。

已有 `prices` 和 `ff_factors` 时, 在可写连接上执行 `factor_panel.sql` 即可创建视图。
查询视图时会按底层表重新得到结果, 不用把这些计算值再存一份。

## 本地检查

```bash
uv run pytest -q tests/test_operation.py tests/test_wikipedia.py tests/test_price_pipeline.py tests/test_french.py tests/test_factor_panel.py
uv run python -m doctest src/quanttoolkit/db/operation.py
uv run ruff check src/quanttoolkit/db/operation.py scripts/ingest_universe.py tests/test_operation.py
uv run ruff format --check src/quanttoolkit/db/operation.py scripts/ingest_universe.py tests/test_operation.py
```

测试覆盖文件持久化, 完整替换名单, 行情更新与追加, 重复写入, SQL NULL,
空输入, 主键及非空约束, 日内时间转换, 只读写入拒绝和事务失败后的回滚
GitHub Actions 生成 XML, HTML 和运行摘要中的覆盖率报告, 并上传报告供下载
