"""DuckDB 数据写入接口"""

import duckdb
import pandas as pd

from quanttoolkit.db.ddl import PRICES_DDL, UNIVERSE_DDL


def ingestion_universe(
    *, df: pd.DataFrame, connection: duckdb.DuckDBPyConnection
) -> int:
    """在事务中用最新成员名单替换 universe 表

    Args:
        df: 普通索引 DataFrame, 列为 ticker, name, sub_industry, cik, sec_title
            cik 为10位字符串, sec_title 可以缺失, 缺失值写为 SQL NULL
        connection: 可写 DuckDB 连接, 调用前不应有未结束的事务
            本函数负责事务, 连接由调用方关闭

    Returns:
        int: 成功写入的记录数
        表不存在时按 UNIVERSE_DDL 建表, 每次调用完整替换 universe 表
        空 DataFrame 会清空成员名单, 写入失败时回滚

    Raises:
        duckdb.Error: 数据库只读, 字段不匹配或事务执行失败

    Example:
        >>> import duckdb
        >>> import pandas as pd
        >>> df = pd.DataFrame([{
        ...     "ticker": "AAPL", "name": "Apple Inc.",
        ...     "sub_industry": "Technology Hardware, Storage & Peripherals",
        ...     "cik": "0000320193", "sec_title": "Apple Inc."
        ... }])
        >>> with duckdb.connect(":memory:") as connection:
        ...     ingestion_universe(df=df, connection=connection)
        1
    """
    connection.execute("BEGIN TRANSACTION")
    try:
        connection.execute(UNIVERSE_DDL)
        connection.execute("DELETE FROM universe")
        connection.execute("INSERT INTO universe BY NAME SELECT * FROM df")
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    return len(df)


def ingestion_prices(*, df: pd.DataFrame, connection: duckdb.DuckDBPyConnection) -> int:
    """在事务中按 (ticker, date) 新增或替换日行情

    Args:
        df: 普通索引 DataFrame, 列为 date, ticker, open, high, low, close,
            adj_close, volume, 列顺序不限, 每个 (ticker, date) 对应一条记录
            date 写为 SQL DATE, 不保留日内时间, ticker 和 close 不可缺失
            价格及成交量沿用输入单位和复权口径, 其余缺失值写为 SQL NULL
        connection: 可写 DuckDB 连接, 调用前不应有未结束的事务
            本函数负责事务, 连接由调用方关闭

    Returns:
        int: 成功处理的输入记录数, 包含新增和替换的记录
        表不存在时按 PRICES_DDL 建表, 相同主键整行替换, 其他历史记录保留
        空 DataFrame 保留已有行情, 写入失败时回滚整个事务

    Raises:
        duckdb.Error: 数据库只读, 字段不匹配, 约束违反或事务执行失败

    Example:
        >>> import duckdb
        >>> import pandas as pd
        >>> df = pd.DataFrame([{
        ...     "date": "2024-01-02", "ticker": "AAPL",
        ...     "open": 185.0, "high": 188.0, "low": 184.0,
        ...     "close": 187.0, "adj_close": 186.0, "volume": 1000.0
        ... }])
        >>> with duckdb.connect(":memory:") as connection:
        ...     ingestion_prices(df=df, connection=connection)
        ...     df["close"] = 188.0
        ...     ingestion_prices(df=df, connection=connection)
        ...     connection.execute("SELECT ticker, close FROM prices").fetchall()
        1
        1
        [('AAPL', 188.0)]
    """
    connection.execute("BEGIN TRANSACTION")
    try:
        connection.execute(PRICES_DDL)
        connection.execute("INSERT OR REPLACE INTO prices BY NAME SELECT * FROM df")
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    return len(df)
