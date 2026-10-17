"""动量因子生成函数的基础函数"""

from __future__ import annotations

import pandas as pd

from quanttoolkit.panel.returns import log_returns


def base_momfactor(
    prices: pd.DataFrame,
    window: int,
    *,
    skip: int = 0,
    price_col: str = "adj_close",
) -> pd.Series:
    """计算滚动窗口动量, 按 skip 期整体后移

    Args:
        prices: [date, ticker] MultiIndex DataFrame, 含 price_col 列
            价格按同一复权口径提供, 非缺失价格为正数, 不修改输入
        window: 无默认值, 正整数, 窗口内单期对数收益率数量
            按每只证券自己的交易记录计数, 不补齐交易日
        skip: 默认 0, 非负整数, 剔除最近 skip 期收益
            整体往后 shift, skip 不包含在 window 内
        price_col: 价格列名, 默认 adj_close

    Returns:
        排序后的 [date, ticker] pd.Series, 单位为累计对数收益率
        skip 为 0 时 name 为 mom_{window}, 否则为 mom_{window}_{skip}
        在记录 t 上计算 log(price[t-skip] / price[t-skip-window])
        窗口内价格缺失或记录不足时为 NaN, 不填补价格
        首次计算需要 window+skip+1 条有效价格记录

    Raises:
        KeyError: price_col 指定的列或 ticker 索引层不存在
        ValueError: pandas 不接受指定的滚动窗口

    Example:
        >>> import numpy as np
        >>> index = pd.MultiIndex.from_product(
        ...     [pd.date_range("2024-01-01", periods=4), ["A"]],
        ...     names=["date", "ticker"],
        ... )
        >>> prices = pd.DataFrame({"adj_close": np.exp([0., 1., 2., 3.])}, index=index)
        >>> factor = base_momfactor(prices, window=2, skip=1)
        >>> factor.name
        'mom_2_1'
        >>> float(factor.iloc[-1].round(6))
        2.0
    """

    ret = log_returns(df=prices, price_col=price_col)

    rolled = ret.groupby(level="ticker").transform(lambda x: x.rolling(window).sum())

    if skip:
        rolled = rolled.groupby(level="ticker").shift(skip)
    rolled.name = f"mom_{window}_{skip}" if skip else f"mom_{window}"

    return rolled
