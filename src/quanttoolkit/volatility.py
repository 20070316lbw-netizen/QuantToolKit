"""基于 (date, ticker) MultiIndex DataFrame 的波动率计算"""

from __future__ import annotations

import pandas as pd

from quanttoolkit._prices import validate_period
from quanttoolkit.returns import historical_return


def _rolling_vol(returns: pd.Series, window: int) -> pd.Series:
    if returns.empty:
        return returns
    return returns.groupby(level="ticker", sort=False).transform(
        lambda series: series.rolling(window, min_periods=window).std(ddof=1)
    )


def history_vol(
    *,
    df: pd.DataFrame,
    window: int = 21,
    price_col: str = "close",
) -> pd.Series:
    """计算历史日简单收益率的滚动波动率。

    Args:
        df: [date, ticker] MultiIndex DataFrame, 含 price_col 列, 键非空且唯一。
        window: 窗口内收益率数量, 默认 21, 至少为 2 的整数, 不接受布尔值。
            期数按每只股票自己的交易记录计数。
        price_col: 价格列名, 默认 close, 非缺失价格须为正数。

    Returns:
        排序后的 [date, ticker] Series, name 为 volatility_{window}, 小数波动率。
        最近 window 个日简单收益率的样本标准差, ddof=1, 包含当天收益率。
        不年化、不填补价格；需 window+1 条有效价格记录才能首次计算。
        窗口不足或窗口内收益率缺失时为 NaN, 空输入返回空 Series。

    Raises:
        TypeError: df 非 DataFrame, 或 window 非整数。
        ValueError: 索引、价格不符合约定, 或 window 小于 2。

    Example:
        >>> index = pd.MultiIndex.from_product(
        ...     [pd.date_range("2024-01-01", periods=3), ["A"]],
        ...     names=["date", "ticker"],
        ... )
        >>> prices = pd.DataFrame({"close": [1., 2., 4.]}, index=index)
        >>> float(history_vol(df=prices, window=2).iloc[-1])
        0.0
    """
    window = validate_period(window, "window", 2)
    returns = historical_return(df=df, n_periods=1, price_col=price_col)
    return _rolling_vol(returns, window).rename(f"volatility_{window}")


def forward_volatility(
    *,
    df: pd.DataFrame,
    window: int = 21,
    gap: int = 1,
    price_col: str = "close",
) -> pd.Series:
    """计算未来持有区间内的波动率标签。

    Args:
        df: [date, ticker] MultiIndex DataFrame, 含 price_col 列, 键非空且唯一。
        window: 持有区间的单期收益率数量, 默认 21, 至少为 2 的整数。
        gap: 基准日到区间起点的记录数, 默认 1, 非负整数。
            期数按每只股票自己的交易记录计数, 均不接受布尔值。
        price_col: 价格列名, 默认 close, 非缺失价格须为正数。

    Returns:
        排序后的 [date, ticker] Series, name 为 forward_volatility_{window}_gap_{gap}。
        使用 price[t+gap] 至 price[t+gap+window] 的 window+1 个价格,
        计算 window 个日简单收益率的样本标准差, ddof=1, 小数波动率。
        不年化、不填补价格；未来记录不足或窗口收益率缺失时为 NaN。
        空输入返回空 Series。使用未来数据, 仅用于事后生成标签。

    Raises:
        TypeError: df 非 DataFrame, 或 window/gap 非整数。
        ValueError: 索引、价格不符合约定, window 小于 2 或 gap 为负数。

    Example:
        >>> index = pd.MultiIndex.from_product(
        ...     [pd.date_range("2024-01-01", periods=3), ["A"]],
        ...     names=["date", "ticker"],
        ... )
        >>> prices = pd.DataFrame({"close": [1., 2., 4.]}, index=index)
        >>> float(forward_volatility(df=prices, window=2, gap=0).iloc[0])
        0.0
    """
    window = validate_period(window, "window", 2)
    gap = validate_period(gap, "gap", 0)
    returns = historical_return(df=df, n_periods=1, price_col=price_col)
    volatility = _rolling_vol(returns, window)
    return (
        volatility.groupby(level="ticker", sort=False)
        .shift(-(gap + window))
        .rename(f"forward_volatility_{window}_gap_{gap}")
    )
