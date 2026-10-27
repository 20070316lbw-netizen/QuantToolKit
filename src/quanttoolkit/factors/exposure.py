"""用滚动 FF5 回归计算股票的 CMA 暴露

先把股票简单收益减去 rf, 再对同期五个因子加截距做回归
这里只取 CMA 的系数, 供两个 smoke 脚本共用
"""

import numpy as np
import pandas as pd

FF5 = ["mkt_rf", "smb", "hml", "rmw", "cma"]


def _rolling_cma(excess: np.ndarray, factors: np.ndarray, window: int) -> np.ndarray:
    rows, width = factors.shape
    design = np.column_stack([np.ones(rows), factors])
    rank = width + 1
    finite = np.isfinite(excess) & np.isfinite(factors).all(axis=1)
    design = np.where(finite[:, None], design, 0.0)
    target = np.where(finite, excess, 0.0)
    moment = np.cumsum(design[:, :, None] * design[:, None, :], axis=0)
    cross = np.cumsum(design * target[:, None], axis=0)
    count = np.cumsum(finite.astype(np.int32), axis=0)
    moment[window:] -= moment[:-window]
    cross[window:] -= cross[:-window]
    count[window:] -= count[:-window]

    ready = count == window
    eye = np.eye(rank)
    moment[~ready] = eye
    cross[~ready] = 0.0
    moment[ready] = moment[ready] + eye * 1e-12
    coef = np.linalg.solve(moment, cross[..., None])[..., 0]
    cma = coef[:, -1]
    cma[~ready] = np.nan
    return cma


def cma_exposure(prices: pd.DataFrame, ff: pd.DataFrame, window: int) -> pd.Series:
    """计算含截距 FF5 滚动回归中的 CMA 暴露

    Args:
        prices: 已排序的 [date, ticker] DataFrame, 含 adj_close 和 volume
        ff: 按 date 索引的日频 DataFrame, 含 rf 和 FF5 列, 收益单位为小数
        window: 无默认值, 正整数, 按每只证券自己的记录计数

    Returns:
        排序后的 [date, ticker] Series, name 为 b_cma, 无量纲
        以简单收益减 rf 为因变量, 同期 FF5 加截距为自变量
        窗口不足, 收益或 FF5 缺失, 或窗口内存在 volume <= 0 时为 NaN
        回归矩阵对角线加 1e-12, 减轻求解时的数值不稳定, 不修改输入

    Raises:
        KeyError: 缺少必需列或索引层
        ValueError: 输入数组无法参与回归计算

    Example:
        >>> dates = pd.date_range("2024-01-01", periods=2)
        >>> index = pd.MultiIndex.from_product([dates, ["A"]], names=["date", "ticker"])
        >>> prices = pd.DataFrame({"adj_close": [1., 2.], "volume": [1., 1.]}, index=index)
        >>> ff = pd.DataFrame(0.0, index=dates, columns=[*FF5, "rf"])
        >>> cma_exposure(prices, ff, window=2).isna().tolist()
        [True, True]
    """
    simple = prices["adj_close"].groupby(level="ticker").pct_change()
    excess = simple - ff["rf"].reindex(simple.index, level="date")
    excess = excess.where(prices["volume"] > 0)
    factors = ff[FF5].reindex(excess.index, level="date")
    pieces = []
    for index in excess.groupby(level="ticker").groups.values():
        ordered = excess.loc[index].sort_index()
        values = _rolling_cma(
            ordered.to_numpy(dtype=float),
            factors.loc[ordered.index, FF5].to_numpy(dtype=float),
            window,
        )
        pieces.append(pd.Series(values, index=ordered.index, name="b_cma"))
    return pd.concat(pieces).sort_index()
