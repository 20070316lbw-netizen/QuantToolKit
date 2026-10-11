"""基于 [date, ticker] 行情的逐证券收益率计算。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quanttoolkit._prices import sorted_prices, validate_period


def future_returns(
    *,
    df: pd.DataFrame,
    n_periods: int = 5,
    gap: int = 1,
    price_col: str = "adj_close",
) -> pd.Series:
    """生成未来持有期的简单收益率标签。

    Args:
        df: [date, ticker] MultiIndex DataFrame, 含 price_col 列, 键非空且唯一。
        n_periods: 持有期数, 默认 5, 正整数, 按每只证券的交易记录计数。
        gap: 当天到入场的记录数, 默认 1, 非负整数;均不接受布尔值。
        price_col: 价格列名, 默认 adj_close, 非缺失价格须为正数。

    Returns:
        排序后的 [date, ticker] Series, name 为 forward_return_{n_periods}_gap_{gap}。
        值为 price[t+gap+n_periods] / price[t+gap] - 1, 单位为小数收益率。
        未来记录不足或端点缺失时为 NaN, 不填补价格;空输入返回空 Series。
        使用未来数据, 仅用于事后生成标签。

    Raises:
        TypeError: df 非 DataFrame, 或期数参数非整数。
        ValueError: 索引、价格不符合约定, n_periods 非正数或 gap 为负数。

    Example:
        >>> index = pd.MultiIndex.from_product(
        ...     [pd.date_range("2024-01-01", periods=3), ["A"]],
        ...     names=["date", "ticker"],
        ... )
        >>> prices = pd.DataFrame({"adj_close": [1., 2., 4.]}, index=index)
        >>> future_returns(df=prices, n_periods=1, gap=0).iloc[:2].tolist()
        [1.0, 1.0]
    """
    n_periods = validate_period(n_periods, "n_periods", 1)
    gap = validate_period(gap, "gap", 0)
    prices = sorted_prices(df, price_col)
    grouped = prices.groupby(level="ticker", sort=False)
    entry = grouped.shift(-gap)
    exit_price = grouped.shift(-(gap + n_periods))
    return (exit_price / entry - 1).rename(f"forward_return_{n_periods}_gap_{gap}")


def historical_return(
    *,
    df: pd.DataFrame,
    n_periods: int = 5,
    price_col: str = "adj_close",
) -> pd.Series:
    """计算截至基准日期的历史简单收益率。

    Args:
        df: [date, ticker] MultiIndex DataFrame, 含 price_col 列, 键非空且唯一。
        n_periods: 回看期数, 默认 5, 正整数, 不接受布尔值。
            按每只证券自己的交易记录计数, 不补齐交易日。
        price_col: 价格列名, 默认 adj_close, 非缺失价格须为正数。

    Returns:
        排序后的 [date, ticker] Series, name 为 historical_return_{n_periods}。
        值为 price[t] / price[t-n_periods] - 1, 单位为小数收益率。
        包含当天价格;历史记录不足或端点缺失时为 NaN, 不填补价格。
        空输入返回空 Series, 可用于历史特征。

    Raises:
        TypeError: df 非 DataFrame, 或 n_periods 非整数。
        ValueError: 索引、价格不符合约定, 或 n_periods 非正数。

    Example:
        >>> index = pd.MultiIndex.from_product(
        ...     [pd.date_range("2024-01-01", periods=3), ["A"]],
        ...     names=["date", "ticker"],
        ... )
        >>> prices = pd.DataFrame({"adj_close": [1., 2., 4.]}, index=index)
        >>> historical_return(df=prices, n_periods=1).iloc[1:].tolist()
        [1.0, 1.0]
    """
    n_periods = validate_period(n_periods, "n_periods", 1)
    prices = sorted_prices(df, price_col)
    previous = prices.groupby(level="ticker", sort=False).shift(n_periods)
    return (prices / previous - 1).rename(f"historical_return_{n_periods}")


def log_returns(
    *,
    df: pd.DataFrame,
    price_col: str = "adj_close",
) -> pd.Series:
    """计算按 ticker 隔离的单期对数收益率。

    Args:
        df: [date, ticker] MultiIndex DataFrame, 含 price_col 列, 键非空且唯一。
        price_col: 价格列名, 默认 adj_close, 非缺失价格须为正数。

    Returns:
        排序后的 [date, ticker] Series, name 为 log_return, 单位为对数收益率。
        值为 log(price[t]) - log(price[t-1]);每只股票首行或相邻价格缺失时
        为 NaN, 不填补交易记录或价格。空输入返回空 Series。

    Raises:
        TypeError: df 非 DataFrame。
        ValueError: 索引、价格或所选列不符合约定。

    Example:
        >>> index = pd.MultiIndex.from_product(
        ...     [pd.date_range("2024-01-01", periods=2), ["A"]],
        ...     names=["date", "ticker"],
        ... )
        >>> prices = pd.DataFrame({"adj_close": [1., np.e]}, index=index)
        >>> log_returns(df=prices).iloc[1].round(6)
        np.float64(1.0)
    """
    prices = sorted_prices(df, price_col)
    return (
        np.log(prices).groupby(level="ticker", sort=False).diff().rename("log_return")
    )
