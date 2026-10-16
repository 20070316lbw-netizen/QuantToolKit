"""抓取最新标普500成员并写入 DuckDB"""

from pathlib import Path

from quanttoolkit.db.connection import get_duckdb
from quanttoolkit.db.operation import SQL_DIR, operation_sql, read_sql
from quanttoolkit.sources.wikipedia import fetch_sp500_universe

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "quanttoolkit.db"


def main() -> None:
    universe = fetch_sp500_universe()
    with get_duckdb(path=DB_PATH, read_only=False) as connection:
        written = operation_sql(
            read_sql(SQL_DIR / "universe.sql"),
            read_sql(SQL_DIR / "universe_ddl.sql"),
            connection,
            universe,
        )
        stored = connection.execute("SELECT COUNT(*) FROM universe").fetchone()[0]

    print(f"抓取 {len(universe)} 条成员记录, 写入 {written} 条, 入库后共 {stored} 条")
    print(f"数据库: {DB_PATH}")


if __name__ == "__main__":
    main()
