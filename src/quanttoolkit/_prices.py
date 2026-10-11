"""收益率和波动率共用的价格输入校验。"""

from numbers import Integral, Real

import numpy as np
import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_complex_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)


def validate_period(value: int, name: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} 必须为整数, 不接受布尔值")
    if value < minimum:
        raise ValueError(f"{name} 必须至少为 {minimum}")
    return int(value)


def sorted_prices(df: pd.DataFrame, price_col: str) -> pd.Series:
    """保留缺失价格, 返回排序后的 float64 Series, 不修改输入。"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df 必须为 DataFrame")
    index = df.index
    if (
        not isinstance(index, pd.MultiIndex)
        or index.nlevels != 2
        or index.names != ["date", "ticker"]
    ):
        raise ValueError("df 必须使用两层 [date, ticker] MultiIndex")
    dates = index.get_level_values("date")
    tickers = index.get_level_values("ticker")
    if not is_datetime64_any_dtype(dates.dtype):
        raise ValueError("date 层必须为 DatetimeIndex")
    if (
        dates.hasnans
        or tickers.hasnans
        or any(isinstance(ticker, str) and not ticker.strip() for ticker in tickers)
    ):
        raise ValueError("date 和 ticker 键不能为空")
    if not index.is_unique:
        raise ValueError("[date, ticker] 键必须唯一")
    if not isinstance(price_col, str) or price_col not in df.columns:
        raise ValueError("price_col 必须指定存在的价格列")
    if df.columns.tolist().count(price_col) != 1:
        raise ValueError("所选价格列名必须唯一")
    prices = df[price_col]
    if is_bool_dtype(prices.dtype) or is_complex_dtype(prices.dtype):
        raise ValueError("非缺失价格必须为实数")
    # 普通浮点价格与 pd.NA 混合时, pandas 可能保留 object dtype。
    if not is_numeric_dtype(prices.dtype) and any(
        not isinstance(value, Real) or isinstance(value, (bool, np.bool_))
        for value in prices.dropna()
    ):
        raise ValueError("非缺失价格必须为实数")
    prices = pd.Series(
        prices.to_numpy(dtype="float64", na_value=np.nan), index=index, name=price_col
    ).sort_index()
    valid = prices.dropna()
    if not np.isfinite(valid).all() or (valid <= 0).any():
        raise ValueError("非缺失价格必须为有限正数")
    return prices
