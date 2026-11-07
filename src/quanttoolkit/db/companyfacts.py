"""把同一批 SEC 期间事实和时点事实写入一个事务"""

import duckdb
import pandas as pd

from quanttoolkit.db.operation import SQL_DIR, read_sql


def write_company_facts(
    connection: duckdb.DuckDBPyConnection,
    period: pd.DataFrame,
    instant: pd.DataFrame,
) -> tuple[int, int]:
    """在同一事务写入期间和时点事实, 同键重导入时覆盖

    Args:
        connection: 可写 DuckDB 连接, 调用前无需开启事务
        period: read_company_facts 返回的期间事实 DataFrame, 普通列
        instant: read_company_facts 返回的时点事实 DataFrame, 普通列

    Returns:
        tuple[int, int]: 本次期间写入行数和时点写入行数
        空 DataFrame 写入行数为 0

    Raises:
        duckdb.Error: 建表或写入失败, 整批回滚

    Example:
        >>> import duckdb
        >>> from quanttoolkit.sources.companyfacts import read_company_facts
        >>> period, instant = read_company_facts(  # doctest: +SKIP
        ...     "data/companyfacts.zip", ["0000320193"]
        ... )
        >>> with duckdb.connect() as connection:  # doctest: +SKIP
        ...     write_company_facts(connection, period, instant) == (len(period), len(instant))
        True
    """
    connection.begin()
    try:
        counts = []
        for name, df in (("period_facts", period), ("instant_facts", instant)):
            connection.execute(read_sql(SQL_DIR / f"{name}_ddl.sql"))
            counts.append(
                connection.execute(read_sql(SQL_DIR / f"{name}.sql")).fetchone()[0]
            )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    return counts[0], counts[1]
