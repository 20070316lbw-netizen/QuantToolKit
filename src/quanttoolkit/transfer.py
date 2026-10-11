"""读取边界：将长表转换为标准 (date, ticker) MultiIndex DataFrame"""

from __future__ import annotations

import pandas as pd


def to_data_ticker_frame(*, df: pd.DataFrame) -> pd.DataFrame:
    """将读取的数据转换为 MultIndex DataFrame ['date', 'ticker']"""

    df['date'] = pd.to_datetime(df['date'], errors="raise")

    return df.set_index(['date', 'ticker']).sort_index()



if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "date": ["2024-01-03", "2024-01-02", "2024-01-03", "2024-01-02", "2024-01-04"],
            "ticker": ["AAPL", "MSFT", "MSFT", "AAPL", "AAPL"],
            "open": [185.0, 370.0, 371.5, 184.5, 186.0],
            "close": [186.5, 372.0, 373.5, 185.8, 187.2],
            "volume": [100000, 200000, 220000, 120000, 150000]
        })

    df = to_data_ticker_frame(df = df)

    print(df)

    