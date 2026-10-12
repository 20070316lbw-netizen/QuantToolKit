"""只读展示本地 prices 表, 转换为 [date, ticker] 行情面板"""

from pathlib import Path

from quanttoolkit.db import get_duckdb
from quanttoolkit.transfer import to_date_ticker_frame

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "quanttoolkit.db"


def main(*, path: Path = DB) -> None:
    with get_duckdb(path=path, read_only=True) as connection:
        df = connection.execute("SELECT * FROM prices").df()
    print(to_date_ticker_frame(df=df))


if __name__ == "__main__":
    main()
