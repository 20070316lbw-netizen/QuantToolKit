"""在每只证券内计算历史收益, 未来收益标签和单期对数收益"""

from __future__ import annotations

import numpy as np
import pandas as pd


def future_returns(
    *,
    df: pd.DataFrame,
    n_periods: int = 5,
    gap: int = 1,
    price_col: str = "adj_close",
) -> pd.Series:
    """生成未来持有期的简单收益率标签

    Args:
        df: [date, ticker] MultiIndex DataFrame, 含 price_col 列, 键非空且唯一
        n_periods: 持有期数, 默认 5, 正整数, 按每只证券的交易记录计数
        gap: 当天到入场的记录数, 默认 1, 非负整数
        price_col: 价格列名, 默认 adj_close, 非缺失价格须为正数

    Returns:
        排序后的 [date, ticker] Series, name 为 forward_return_{n_periods}_gap_{gap}
        值为 price[t+gap+n_periods] / price[t+gap] - 1, 单位为小数收益率
        未来记录不足或端点缺失时为 NaN, 不填补价格; 空输入返回空 Series
        使用未来数据, 是监督模型的标签, 例如 LightGBM 要预测的持有期收益
        基准日还不知道这个值, 不能当作当天的特征
        与 split_dates 合用时, embargo 取 gap + n_periods
        因为标签用到的最后价格落在 t + gap + n_periods

    Raises:
        KeyError: price_col 指定的列或 ticker 索引层不存在

    Example:
        >>> index = pd.MultiIndex.from_product(
        ...     [pd.date_range("2024-01-01", periods=3), ["A"]],
        ...     names=["date", "ticker"],
        ... )
        >>> prices = pd.DataFrame({"adj_close": [1., 2., 4.]}, index=index)
        >>> future_returns(df=prices, n_periods=1, gap=0).iloc[:2].tolist()
        [1.0, 1.0]
    """
    prices = df[price_col].sort_index()
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
    """计算截至基准日期的历史简单收益率

    Args:
        df: [date, ticker] MultiIndex DataFrame, 含 price_col 列, 键非空且唯一
        n_periods: 回看期数, 默认 5, 正整数
            按每只证券自己的交易记录计数, 不补齐交易日
        price_col: 价格列名, 默认 adj_close, 非缺失价格须为正数

    Returns:
        排序后的 [date, ticker] Series, name 为 historical_return_{n_periods}
        值为 price[t] / price[t-n_periods] - 1, 单位为小数收益率
        包含当天价格; 历史记录不足或端点缺失时为 NaN, 不填补价格
        空输入返回空 Series, 可用于历史特征

    Raises:
        KeyError: price_col 指定的列或 ticker 索引层不存在

    Example:
        >>> index = pd.MultiIndex.from_product(
        ...     [pd.date_range("2024-01-01", periods=3), ["A"]],
        ...     names=["date", "ticker"],
        ... )
        >>> prices = pd.DataFrame({"adj_close": [1., 2., 4.]}, index=index)
        >>> historical_return(df=prices, n_periods=1).iloc[1:].tolist()
        [1.0, 1.0]
    """
    prices = df[price_col].sort_index()
    previous = prices.groupby(level="ticker", sort=False).shift(n_periods)
    return (prices / previous - 1).rename(f"historical_return_{n_periods}")


def log_returns(
    *,
    df: pd.DataFrame,
    price_col: str = "adj_close",
) -> pd.Series:
    """在每只证券内计算相邻两条价格的对数收益率

    Args:
        df: [date, ticker] MultiIndex DataFrame, 含 price_col 列, 键非空且唯一
        price_col: 价格列名, 默认 adj_close, 非缺失价格须为正数

    Returns:
        排序后的 [date, ticker] Series, name 为 log_return, 单位为对数收益率
        值为 log(price[t]) - log(price[t-1]); 每只股票首行或相邻价格缺失时
        为 NaN, 不填补交易记录或价格
        空输入返回空 Series

    Raises:
        KeyError: price_col 指定的列或 ticker 索引层不存在

    Example:
        >>> index = pd.MultiIndex.from_product(
        ...     [pd.date_range("2024-01-01", periods=2), ["A"]],
        ...     names=["date", "ticker"],
        ... )
        >>> prices = pd.DataFrame({"adj_close": [1., np.e]}, index=index)
        >>> log_returns(df=prices).iloc[1].round(6)
        np.float64(1.0)
    """
    prices = df[price_col].sort_index()
    return (
        np.log(prices).groupby(level="ticker", sort=False).diff().rename("log_return")
    )
