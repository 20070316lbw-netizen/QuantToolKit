"""首日等金额买入, 之后保持份额不变, 用作买入持有基准"""

from __future__ import annotations

import pandas as pd


def equal_weight_hold(
    *,
    prices: pd.DataFrame,
    price_col: str = "adj_close",
) -> pd.Series:
    """在首个交易日等权买入有价格的证券, 之后持仓不变

    持仓不变是指首日买入的份额一直留到最后, 不按后续涨跌再平衡
    只使用面板排序后第一个日期上价格非缺失的证券
    之后才出现的证券不买入
    组合净值是这些证券 price / 首日价格 的等权平均
    不修改输入, 也不填补价格

    Args:
        prices: [date, ticker] MultiIndex DataFrame, 含 price_col 列
            键非空且唯一, 非缺失价格为正数
        price_col: 价格列名, 默认 adj_close

    Returns:
        按日期升序的 Series, name 为 equal_weight_hold
        索引是排序后的日期, 值为组合单期简单收益率
        首日没有前一日净值, 为 NaN
        已持有证券在某日价格缺失时, 当天和下一期收益为 NaN
        空输入返回空 Series

    Raises:
        KeyError: price_col 指定的列或 ticker 索引层不存在

    Example:
        >>> index = pd.MultiIndex.from_product(
        ...     [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["A", "B"]],
        ...     names=["date", "ticker"],
        ... )
        >>> prices = pd.DataFrame({"adj_close": [1.0, 1.0, 2.0, 4.0]}, index=index)
        >>> held = equal_weight_hold(prices=prices)
        >>> held.name
        'equal_weight_hold'
        >>> held.isna().iloc[0].item()
        True
        >>> round(held.iloc[1].item(), 6)
        2.0
    """
    level = prices[price_col].sort_index()
    wide = level.unstack(level="ticker")
    if wide.empty:
        return pd.Series(dtype=float, name="equal_weight_hold")

    start = wide.iloc[0]
    held = wide.loc[:, start.notna()]
    gross = held.div(start.dropna(), axis=1)
    returns = gross.mean(axis=1, skipna=False).pct_change()
    returns.name = "equal_weight_hold"
    return returns
