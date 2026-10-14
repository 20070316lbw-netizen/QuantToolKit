"""按 universe 的 CIK 读取 companyfacts.zip, 写入 period_facts 与 instant_facts"""

from pathlib import Path

from quanttoolkit.db.connection import DB, get_duckdb
from quanttoolkit.db.operation import SQL_DIR, operation_sql, read_sql
from quanttoolkit.sources.companyfacts import read_company_facts

ZIP_PATH = Path(__file__).resolve().parents[1] / "data" / "companyfacts.zip"
COMPANY_BATCH = 25


def main() -> None:
    with get_duckdb(path=DB, read_only=True) as connection:
        ciks = (
            connection.execute("SELECT cik FROM universe ORDER BY cik")
            .df()["cik"]
            .tolist()
        )

    period_sql = read_sql(SQL_DIR / "period_facts.sql")
    period_ddl = read_sql(SQL_DIR / "period_facts_ddl.sql")
    instant_sql = read_sql(SQL_DIR / "instant_facts.sql")
    instant_ddl = read_sql(SQL_DIR / "instant_facts_ddl.sql")

    period_written = 0
    instant_written = 0
    with get_duckdb(path=DB, read_only=False) as connection:
        for offset in range(0, len(ciks), COMPANY_BATCH):
            batch = ciks[offset : offset + COMPANY_BATCH]
            period, instant = read_company_facts(ZIP_PATH, batch)
            period_written += operation_sql(period_sql, period_ddl, connection, period)
            instant_written += operation_sql(
                instant_sql, instant_ddl, connection, instant
            )
            print(f"已处理 {min(offset + COMPANY_BATCH, len(ciks))} / {len(ciks)} 家")

    with get_duckdb(path=DB, read_only=True) as connection:
        period_count, instant_count = connection.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM period_facts),
                (SELECT COUNT(*) FROM instant_facts)
            """
        ).fetchone()

    print(f"写入期间事实 {period_written} 条, 时点事实 {instant_written} 条")
    print(f"库内期间事实 {period_count} 条, 时点事实 {instant_count} 条")
    print(f"数据库: {DB}")


if __name__ == "__main__":
    main()
