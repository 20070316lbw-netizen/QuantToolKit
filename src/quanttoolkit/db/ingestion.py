"""DuckDB 数据写入接口"""

import duckdb
import pandas as pd

from quanttoolkit.db.ddl import UNIVERSE_DDL


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
