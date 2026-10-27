"""把含 date 和 ticker 列的长表转成计算使用的 [date, ticker] 面板"""

from __future__ import annotations

import pandas as pd


def to_date_ticker_frame(*, df: pd.DataFrame) -> pd.DataFrame:
    """把 date 和 ticker 列设为索引, 得到 [date, ticker] 面板

    Args:
        df: 普通索引 DataFrame, 必需列为 date 和 ticker, 无默认值
            date 可转换为日期, ticker 为证券代码, 其余列沿用输入单位
            本函数会将输入 df 的 date 列转为日期类型

    Returns:
        按 [date, ticker] 升序排序的 MultiIndex DataFrame
        保留其余数据列及缺失值, 不补齐日期或删除记录
        空输入返回保留列结构的空 DataFrame

    Raises:
        KeyError: date 或 ticker 列不存在
        ValueError: date 无法转换为日期

    Example:
        >>> raw = pd.DataFrame({
        ...     "date": ["2024-01-03", "2024-01-02"],
        ...     "ticker": ["A", "A"], "close": [110.0, 100.0],
        ... })
        >>> prices = to_date_ticker_frame(df=raw)
        >>> list(prices.index.names)
        ['date', 'ticker']
        >>> prices["close"].tolist()
        [100.0, 110.0]
    """

    df["date"] = pd.to_datetime(df["date"], errors="raise")

    return df.set_index(["date", "ticker"]).sort_index()
