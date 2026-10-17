"""按 universe 下载日行情并写入 prices"""

from __future__ import annotations

from quanttoolkit.db.connection import DB, get_duckdb
from quanttoolkit.db.operation import SQL_DIR, operation_sql, read_sql
from quanttoolkit.panel.prices import fill_price_gaps
from quanttoolkit.sources import get_prices


def main() -> None:
    with get_duckdb(path=DB, read_only=True) as con:
        universe = con.execute("SELECT ticker FROM universe").df()["ticker"]

    with get_duckdb(path=DB, read_only=False) as connection:
        prices = fill_price_gaps(
            get_prices(
                tickers=universe,
                start="2016-01-01",
                end="2026-10-10",
                auto_adjust=True,
                progress=True,
            )
        )
        written = operation_sql(
            read_sql(SQL_DIR / "prices.sql"),
            read_sql(SQL_DIR / "prices_ddl.sql"),
            connection,
            prices.reset_index(),
        )

    with get_duckdb(path=DB, read_only=True) as con:
        summary = con.execute(
            """
            SELECT COUNT(*),
                   COUNT(DISTINCT ticker),
                   MIN(date),
                   MAX(date)
            FROM prices
            """
        ).fetchone()

    print(f"写入 {written} 条")
    print(f"库内 {summary[0]} 条, {summary[1]} 个 ticker, {summary[2]} ~ {summary[3]}")


if __name__ == "__main__":
    main()
