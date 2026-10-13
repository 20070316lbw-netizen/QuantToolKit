"""使用本地 SEC 映射文件抓取最新标普500成员并写入 DuckDB"""

from pathlib import Path

from quanttoolkit.db.connection import get_duckdb
from quanttoolkit.db.operation import SQL_DIR, operation_sql, read_sql
from quanttoolkit.sources.wikipedia import fetch_sp500_universe

PROJECT_ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = PROJECT_ROOT / "data" / "company_tickers.json"
DB_PATH = PROJECT_ROOT / "data" / "quanttoolkit.db"


def main() -> None:
    universe = fetch_sp500_universe(json_path=JSON_PATH)
    with get_duckdb(path=DB_PATH, read_only=False) as connection:
        written = operation_sql(
            read_sql(SQL_DIR / "universe.sql"),
            read_sql(SQL_DIR / "universe_ddl.sql"),
            connection,
            universe,
        )
        stored, missing_titles = connection.execute(
            "SELECT COUNT(*), COUNT(*) FILTER (WHERE sec_title IS NULL) FROM universe"
        ).fetchone()

    print(f"抓取 {len(universe)} 条成员记录, 写入 {written} 条, 入库后共 {stored} 条")
    print(f"未匹配 SEC 名称: {missing_titles} 条")
    print(f"数据库: {DB_PATH}")


if __name__ == "__main__":
    main()
