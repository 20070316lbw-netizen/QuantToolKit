"""写入 prices 前, 补齐行情面板里的缺失交易日"""

from __future__ import annotations

import pandas as pd


def fill_price_gaps(prices: pd.DataFrame) -> pd.DataFrame:
    """按已有交易日补齐每只证券首尾之间的缺失行情

    Args:
        prices: [date, ticker] MultiIndex DataFrame, 不修改输入
            列沿用行情面板, 通常含 open, high, low, close, adj_close, volume
            日历取输入里已经出现的日期, 不使用自然日

    Returns:
        按 [date, ticker] 升序排序的 DataFrame, 列与输入相同
        每只证券只补 [最早日期, 最晚日期] 内, 日历上有而该证券没有的日期
        补上的价格列沿用该证券前一条记录, volume 为 0
        已有记录保持原值, 上市前和最晚日期之后不补
        空输入返回保留列结构的空 DataFrame

    Raises:
        KeyError: 索引缺少 date 或 ticker 层

    Example:
        >>> index = pd.MultiIndex.from_tuples(
        ...     [
        ...         (pd.Timestamp("2024-01-02"), "A"),
        ...         (pd.Timestamp("2024-01-03"), "A"),
        ...         (pd.Timestamp("2024-01-02"), "B"),
        ...         (pd.Timestamp("2024-01-04"), "B"),
        ...     ],
        ...     names=["date", "ticker"],
        ... )
        >>> prices = pd.DataFrame(
        ...     {
        ...         "close": [10.0, 11.0, 20.0, 22.0],
        ...         "volume": [100.0, 110.0, 200.0, 220.0],
        ...     },
        ...     index=index,
        ... )
        >>> filled = fill_price_gaps(prices)
        >>> filled.xs("A", level="ticker").index.strftime("%Y-%m-%d").tolist()
        ['2024-01-02', '2024-01-03']
        >>> filled.xs("B", level="ticker")["close"].tolist()
        [20.0, 20.0, 22.0]
        >>> filled.xs("B", level="ticker")["volume"].tolist()
        [200.0, 0.0, 220.0]
    """
    frame = prices.sort_index()
    if frame.empty:
        return frame.copy()

    observed = frame.index.to_frame(index=False)
    calendar = pd.DataFrame({"date": observed["date"].unique()})
    span = observed.groupby("ticker", sort=False)["date"].agg(start="min", end="max")
    grid = span.reset_index().merge(calendar, how="cross")
    grid = grid.loc[(grid["date"] >= grid["start"]) & (grid["date"] <= grid["end"])]
    full = pd.MultiIndex.from_frame(
        grid[["date", "ticker"]], names=["date", "ticker"]
    ).sort_values()

    expanded = frame.reindex(full)
    added = ~expanded.index.isin(frame.index)
    price_cols = [column for column in expanded.columns if column != "volume"]
    if price_cols and bool(added.any()):
        carried = expanded.groupby(level="ticker", sort=False)[price_cols].ffill()
        expanded.loc[added, price_cols] = carried.loc[added, price_cols]
    if "volume" in expanded.columns and bool(added.any()):
        expanded.loc[added, "volume"] = 0.0
    return expanded
