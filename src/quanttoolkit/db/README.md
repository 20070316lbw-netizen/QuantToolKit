# db

提供 DuckDB 连接, 建表语句和数据写入接口, 当前支持最新标普500成员名单的 `universe` 表

| 文件 | 接口 | 职责 |
| --- | --- | --- |
| [connection.py](connection.py) | `get_duckdb(*, path, read_only=True)` | 打开数据库, 默认只读 |
| [ddl.py](ddl.py) | `UNIVERSE_DDL` | 定义 universe 表结构 |
| [ingestion.py](ingestion.py) | `ingestion_universe(*, df, connection)` | 在事务中完整更新成员名单, 返回写入条数 |

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
from pathlib import Path

from quanttoolkit.db.connection import get_duckdb

with get_duckdb(path=Path("data/quanttoolkit.db")) as connection:
    members = connection.execute("SELECT * FROM universe ORDER BY ticker").df()

print(members.head())
```

`get_duckdb` 默认 `read_only=True`, 数据库不存在时抛出 `FileNotFoundError`
使用 `read_only=False` 可创建数据库及父目录, 连接由调用方通过 `with` 或 `close()` 关闭
建表由入库函数负责

## 单独写入成员名单

下面的示例使用内存数据库, 可以独立执行:

```python
import duckdb
import pandas as pd

from quanttoolkit.db.ingestion import ingestion_universe

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
    written = ingestion_universe(df=universe, connection=connection)
    stored = connection.execute("SELECT COUNT(*) FROM universe").fetchone()[0]

print(written, stored)  # 1 1
```

调用时需传入可写连接, 且连接上没有未结束的事务
首次调用按 DDL 建表, 再次调用完整替换名单, 连续写入同一份数据保持相同记录数
空 DataFrame 会清空成员名单, 写入异常会回滚整个事务, 保留原有数据
DataFrame 列按名字写入, 缺失值写为 SQL NULL, 字段不匹配等错误由 DuckDB 抛出
其他数据表保留原样

## 本地检查

```bash
uv run pytest -q --cov=quanttoolkit --cov-branch --cov-report=term-missing
uv run python -m doctest src/quanttoolkit/db/ingestion.py
uv run ruff check src/quanttoolkit/db/ingestion.py scripts/ingest_universe.py tests/test_ingestion.py
uv run ruff format --check src/quanttoolkit/db/ingestion.py scripts/ingest_universe.py tests/test_ingestion.py
```

测试覆盖文件持久化, 完整替换名单, 重复写入, SQL NULL 和写入失败后的事务回滚
GitHub Actions 生成 XML, HTML 和运行摘要中的覆盖率报告, 上传报告供下载并发送给 Codecov
首页 Coverage 徽章显示 Codecov 收到的 main 分支覆盖率
