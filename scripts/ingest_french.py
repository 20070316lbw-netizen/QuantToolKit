"""先读取 data 下的 FF3 和 FF5 日频 ZIP, 再一起写入 ff_factors"""

from pathlib import Path

import pandas as pd

from quanttoolkit.db.connection import DB, get_duckdb
from quanttoolkit.db.operation import SQL_DIR, operation_sql, read_sql
from quanttoolkit.sources.french import read_french_factors

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "ff3": ROOT / "data" / "ff3_daily.zip",
    "ff5": ROOT / "data" / "ff5_daily.zip",
}
COLUMNS = ["date", "model", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"]


def main() -> None:
    frames = []
    for model, path in FILES.items():
        factors = read_french_factors(path).reset_index()
        factors.insert(1, "model", model)
        frames.append(factors)
    factors = pd.concat(frames, ignore_index=True).reindex(columns=COLUMNS)

    with get_duckdb(path=DB, read_only=False) as connection:
        written = operation_sql(
            read_sql(SQL_DIR / "ff_factors.sql"),
            read_sql(SQL_DIR / "ff_factors_ddl.sql"),
            connection,
            factors,
        )
        summary = connection.execute(
            """
            SELECT model, COUNT(*), MIN(date), MAX(date)
            FROM ff_factors
            GROUP BY model
            ORDER BY model
            """
        ).fetchall()

    print(f"写入 {written} 条")
    for model, count, start, end in summary:
        print(f"{model}: {count} 条, {start} ~ {end}")
    print(f"数据库: {DB}")


if __name__ == "__main__":
    main()
