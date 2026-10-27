"""计算过去一段时间里对数收益的波动"""

from __future__ import annotations

import pandas as pd

from quanttoolkit.panel.returns import log_returns


def base_logvolfactor(
    prices: pd.DataFrame,
    window: int,
    *,
    price_col: str = "adj_close",
) -> pd.Series:
    """计算滚动窗口对数波动率, 取单期对数收益率的样本标准差

    Args:
        prices: [date, ticker] MultiIndex DataFrame, 含 price_col 列
            价格按同一复权口径提供, 非缺失价格为正数, 不修改输入
        window: 无默认值, 正整数, 窗口内单期对数收益率数量
            按每只证券自己的交易记录计数, 不补齐交易日
            样本标准差 ddof=1, 至少 2 条对数收益才能得到数值
        price_col: 价格列名, 默认 adj_close

    Returns:
        排序后的 [date, ticker] pd.Series, name 为 logvol_{window}
        单位为对数收益率的样本标准差, 不年化
        在记录 t 上计算 t-window+1 到 t 这 window 条对数收益的标准差
        窗口内对数收益缺失或记录不足时为 NaN, 不填补价格
        首次计算需要 window+1 条有效价格记录
        空输入返回命名的空 Series

    Raises:
        KeyError: price_col 指定的列或 ticker 索引层不存在
        ValueError: pandas 不接受指定的滚动窗口

    Example:
        >>> index = pd.MultiIndex.from_product(
        ...     [pd.date_range("2024-01-01", periods=3), ["A"]],
        ...     names=["date", "ticker"],
        ... )
        >>> prices = pd.DataFrame({"adj_close": [1., 2., 4.]}, index=index)
        >>> factor = base_logvolfactor(prices, window=2)
        >>> factor.name
        'logvol_2'
        >>> float(factor.iloc[-1])
        0.0
    """
    ret = log_returns(df=prices, price_col=price_col)
    if ret.empty:
        return ret.rename(f"logvol_{window}")

    rolled = ret.groupby(level="ticker").transform(
        lambda series: series.rolling(window).std(ddof=1)
    )
    rolled.name = f"logvol_{window}"
    return rolled
