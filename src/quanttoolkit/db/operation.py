"""在事务中执行写入 SQL

建表语句和写入语句放在同一次事务里
SQL 文件位于 SQL_DIR
语句里的 df 使用本函数的同名参数
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

SQL_DIR = Path(__file__).resolve().parent / "sql"


def read_sql(sql: Path) -> str:
    """读取 sql 文件的文本

    Args:
        sql: *.sql 文件路径

    Returns:
        str: 文件内容
    """
    return Path(sql).read_text(encoding="utf-8")


def operation_sql(
    sql: str,
    ddl: str,
    connection: duckdb.DuckDBPyConnection,
    df: pd.DataFrame | None = None,
) -> int:
    """在事务中执行建表语句和写入语句

    Args:
        sql: 写入 SQL, 多条语句用分号分隔
            语句中的 df 指向同名参数, 不引用数据表时该参数为 None
        ddl: 建表 SQL, 表不存在时由此建表
        connection: 可写 DuckDB 连接, 调用前不应有未结束的事务
            本函数负责事务, 连接由调用方关闭
        df: 供 sql 引用的 DataFrame, 默认 None

    Returns:
        int: 最后一条语句返回的计数
        INSERT 时为写入行数
        写入失败时回滚, 已提交的数据保留

    Raises:
        duckdb.Error: 数据库只读, 字段不匹配或事务执行失败

    Example:
        >>> import duckdb
        >>> import pandas as pd
        >>> df = pd.DataFrame([{
        ...     "ticker": "AAPL", "name": "Apple Inc.",
        ...     "sub_industry": "Technology Hardware",
        ...     "cik": "0000320193", "sec_title": "Apple Inc."
        ... }])
        >>> with duckdb.connect(":memory:") as connection:
        ...     operation_sql(
        ...         read_sql(SQL_DIR / "universe.sql"),
        ...         read_sql(SQL_DIR / "universe_ddl.sql"),
        ...         connection,
        ...         df,
        ...     )
        1
    """
    connection.execute("BEGIN TRANSACTION")
    try:
        connection.execute(ddl)
        written = connection.execute(sql).fetchone()[0]
        connection.execute("COMMIT")
    except Exception:
        connection.execute("ROLLBACK")
        raise
    return int(written)
