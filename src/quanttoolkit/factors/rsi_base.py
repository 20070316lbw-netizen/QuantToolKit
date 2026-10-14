"""RSI 因子的基础函数"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi_value(avg_up: float, avg_down: float) -> float:
    total = avg_up + avg_down
    if total == 0:
        return np.nan
    return 100.0 * avg_up / total


def _wilder_rsi(prices: np.ndarray, window: int) -> np.ndarray:
    # 前 window 个连续变动用简单平均做初值, 之后用 Wilder 平滑
    out = np.full(prices.shape, np.nan)
    avg_up = np.nan
    avg_down = np.nan
    pending = 0
    up_sum = 0.0
    down_sum = 0.0

    for i in range(1, len(prices)):
        delta = prices[i] - prices[i - 1]
        if np.isnan(delta) or np.isnan(avg_up):
            if np.isnan(delta):
                pending = 0
                up_sum = 0.0
                down_sum = 0.0
                avg_up = np.nan
                avg_down = np.nan
                continue
            up = delta if delta > 0 else 0.0
            down = -delta if delta < 0 else 0.0
            pending += 1
            up_sum += up
            down_sum += down
            if pending < window:
                continue
            avg_up = up_sum / window
            avg_down = down_sum / window
            out[i] = _rsi_value(avg_up, avg_down)
            pending = 0
            up_sum = 0.0
            down_sum = 0.0
            continue

        up = delta if delta > 0 else 0.0
        down = -delta if delta < 0 else 0.0
        avg_up = (avg_up * (window - 1) + up) / window
        avg_down = (avg_down * (window - 1) + down) / window
        out[i] = _rsi_value(avg_up, avg_down)
    return out


def base_rsifactor(
    prices: pd.DataFrame,
    window: int = 14,
    *,
    price_col: str = "adj_close",
) -> pd.Series:
    """计算 Wilder RSI

    Args:
        prices: [date, ticker] MultiIndex DataFrame, 含 price_col 列
            价格按同一复权口径提供, 非缺失价格为正数, 不修改输入
        window: 默认 14, 正整数, 平滑所用的价格变动个数
            按每只证券自己的交易记录计数, 不补齐交易日
        price_col: 价格列名, 默认 adj_close

    Returns:
        排序后的 [date, ticker] pd.Series, name 为 rsi_{window}
        取值在 0 到 100 之间, 无单位
        前 window 个连续变动取简单平均, 作为平均上涨和平均下跌的初值
        之后 avg = (上一期 avg * (window - 1) + 当期涨跌) / window
        RSI = 100 * 平均上涨 / (平均上涨 + 平均下跌)
        只有上涨时为 100, 只有下跌时为 0, 两者都为 0 时为 NaN
        价格缺失会打断连续变动, 之后重新累计 window 个变动
        不填补价格, 不跳过成交量为 0 的记录
        首次计算需要 window+1 条有效价格记录
        空输入返回命名的空 Series

    Raises:
        KeyError: price_col 指定的列或 ticker 索引层不存在

    Example:
        >>> index = pd.MultiIndex.from_product(
        ...     [pd.date_range("2024-01-01", periods=4), ["A"]],
        ...     names=["date", "ticker"],
        ... )
        >>> prices = pd.DataFrame({"adj_close": [1., 2., 1., 2.]}, index=index)
        >>> factor = base_rsifactor(prices, window=2)
        >>> factor.name
        'rsi_2'
        >>> bool(factor.iloc[:2].isna().all())
        True
        >>> float(factor.iloc[2])
        50.0
        >>> float(factor.iloc[3])
        75.0
    """
    close = prices[price_col].sort_index()
    if close.empty:
        return close.rename(f"rsi_{window}")

    result = close.groupby(level="ticker", sort=False).transform(
        lambda series: _wilder_rsi(series.to_numpy(dtype=float), window)
    )
    result.name = f"rsi_{window}"
    return result
