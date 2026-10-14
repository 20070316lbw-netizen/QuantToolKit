"""因子截面相关与收益序列的基本评定指标"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _cross_sectional_corr(
    factor: pd.Series,
    forward_return: pd.Series,
    *,
    method: str,
) -> pd.Series:
    paired = pd.concat(
        [factor.rename("_factor"), forward_return.rename("_forward")],
        axis=1,
        join="inner",
    )
    if paired.empty:
        return pd.Series(dtype=float, index=paired.index.get_level_values("date")[:0])

    corr = paired.groupby(level="date", sort=True).corr(method=method)
    result = corr.xs("_factor", level=-1)["_forward"]
    result.index.name = "date"
    return result


def _finite_values(returns: pd.Series) -> np.ndarray | None:
    values = returns.sort_index().to_numpy(dtype=float)
    if values.size == 0 or np.isnan(values).any():
        return None
    return values


def _mean_over_std(mean: float, std: float) -> float:
    if np.isnan(mean) or np.isnan(std):
        return np.nan
    if std == 0:
        return np.inf if mean != 0 else np.nan
    return float(mean / std)


def ic(*, factor: pd.Series, forward_return: pd.Series) -> pd.Series:
    """计算每个日期截面上因子与未来收益的 Pearson 相关

    Args:
        factor: [date, ticker] Series, 因子值, 不修改输入
        forward_return: [date, ticker] Series, 与因子按索引对齐的未来简单收益率
            通常来自 future_returns, 含有未来数据, 只用于事后评定

    Returns:
        按 date 升序的 Series, name 为 ic
        每个日期只使用两边都存在的证券, 成对去掉缺失值后计算 Pearson 相关
        有效证券少于 2 只, 或任一变量没有波动时, 该日期为 NaN
        两边都没有成对记录的日期不出现; 空输入返回空 Series

    Raises:
        KeyError: 索引没有 date 层

    Example:
        >>> index = pd.MultiIndex.from_product(
        ...     [pd.to_datetime(["2024-01-02"]), ["A", "B", "C"]],
        ...     names=["date", "ticker"],
        ... )
        >>> factor = pd.Series([1.0, 2.0, 3.0], index=index)
        >>> forward = pd.Series([0.1, 0.2, 0.3], index=index)
        >>> float(ic(factor=factor, forward_return=forward).iloc[0])
        1.0
    """
    result = _cross_sectional_corr(factor, forward_return, method="pearson")
    result.name = "ic"
    return result


def rank_ic(*, factor: pd.Series, forward_return: pd.Series) -> pd.Series:
    """计算每个日期截面上因子与未来收益的 Spearman 相关

    Args:
        factor: [date, ticker] Series, 因子值, 不修改输入
        forward_return: [date, ticker] Series, 与因子按索引对齐的未来简单收益率
            通常来自 future_returns, 含有未来数据, 只用于事后评定

    Returns:
        按 date 升序的 Series, name 为 rank_ic
        在有效证券上先取平均秩, 再计算秩的 Pearson 相关
        有效证券少于 2 只, 或秩没有波动时, 该日期为 NaN
        两边都没有成对记录的日期不出现; 空输入返回空 Series

    Raises:
        KeyError: 索引没有 date 层

    Example:
        >>> index = pd.MultiIndex.from_product(
        ...     [pd.to_datetime(["2024-01-03"]), ["A", "B", "C"]],
        ...     names=["date", "ticker"],
        ... )
        >>> factor = pd.Series([1.0, 2.0, 4.0], index=index)
        >>> forward = pd.Series([0.0, 0.1, 0.1], index=index)
        >>> float(rank_ic(factor=factor, forward_return=forward).iloc[0].round(6))
        0.866025
    """
    result = _cross_sectional_corr(factor, forward_return, method="spearman")
    result.name = "rank_ic"
    return result


def ic_summary(*, ic: pd.Series) -> pd.Series:
    """汇总信息系数序列的均值, 样本标准差和信息比率

    Args:
        ic: 按日期索引的 ic 或 rank_ic
            缺失日期表示该截面无法计算, 汇总时忽略

    Returns:
        name 为 ic_summary 的 Series, 索引为 ic_mean, ic_std, ic_ir
        ic_std 使用 ddof=1, ic_ir 为 ic_mean / ic_std, 都不年化
        有效日期少于 2 个时 ic_std 与 ic_ir 为 NaN
        标准差为 0 且均值非 0 时 ic_ir 为 inf
        均值和标准差都为 0 时 ic_ir 为 NaN

    Raises:
        本函数不额外校验输入

    Example:
        >>> [round(v, 6) for v in ic_summary(ic=pd.Series([1.0, -1.0]))]
        [0.0, 1.414214, 0.0]
    """
    mean = ic.mean()
    std = ic.std(ddof=1)
    return pd.Series(
        {
            "ic_mean": mean,
            "ic_std": std,
            "ic_ir": _mean_over_std(float(mean), float(std)),
        },
        name="ic_summary",
    )


def total_return(*, returns: pd.Series) -> float:
    """计算简单收益率序列的区间复利收益

    Args:
        returns: 简单收益率 Series, 按索引升序复利, 不修改输入
            不是对数收益率, 不要求索引类型, 但索引必须能排序

    Returns:
        prod(1 + r) - 1, 单位为小数收益率
        任一收益缺失或输入为空时为 NaN, 不删去缺失收益后缩短区间

    Raises:
        ValueError: returns 含有无法转换为浮点数的值
        TypeError: 索引无法排序

    Example:
        >>> total_return(returns=pd.Series([0.5, -0.5]))
        -0.25
    """
    values = _finite_values(returns)
    if values is None:
        return np.nan
    return float(np.prod(1 + values) - 1)


def sharpe(*, returns: pd.Series) -> float:
    """计算简单收益率序列的样本夏普比率

    Args:
        returns: 简单收益率 Series, 按索引升序后取均值和标准差, 不修改输入
            无风险利率按 0, 不年化

    Returns:
        mean(r) / std(r, ddof=1)
        有效收益少于 2 个, 或任一收益缺失, 或输入为空时为 NaN
        标准差为 0 且均值非 0 时为 inf, 均值和标准差都为 0 时为 NaN
        不删去缺失收益后缩短样本

    Raises:
        ValueError: returns 含有无法转换为浮点数的值
        TypeError: 索引无法排序

    Example:
        >>> round(sharpe(returns=pd.Series([0.1, 0.3])), 6)
        1.414214
    """
    values = _finite_values(returns)
    if values is None:
        return np.nan
    return _mean_over_std(float(values.mean()), float(values.std(ddof=1)))


def max_drawdown(*, returns: pd.Series) -> float:
    """计算简单收益率复利净值的最大回撤

    Args:
        returns: 简单收益率 Series, 按索引升序复利, 不修改输入
            净值从 1 起算, 第一笔收益之前的峰值为 1

    Returns:
        各时点净值 / 历史峰值 - 1 的最小值, 单位为小数
        没有回撤时为 0, 回撤 20% 时为 -0.2
        任一收益缺失或输入为空时为 NaN, 不删去缺失收益后缩短路径

    Raises:
        ValueError: returns 含有无法转换为浮点数的值
        TypeError: 索引无法排序

    Example:
        >>> max_drawdown(returns=pd.Series([0.5, -0.5]))
        -0.5
    """
    values = _finite_values(returns)
    if values is None:
        return np.nan
    nav = np.cumprod(1 + values)
    peak = np.maximum.accumulate(np.concatenate(([1.0], nav)))
    return float(np.min(nav / peak[1:] - 1))


def win_rate(*, returns: pd.Series) -> float:
    """计算简单收益率大于 0 的比例

    Args:
        returns: 简单收益率 Series, 不修改输入
            按索引排序后再计数, 排序不改变比例

    Returns:
        收益严格大于 0 的个数除以样本个数
        0 不算获胜
        任一收益缺失或输入为空时为 NaN, 不删去缺失收益后缩短样本

    Raises:
        ValueError: returns 含有无法转换为浮点数的值
        TypeError: 索引无法排序

    Example:
        >>> win_rate(returns=pd.Series([0.1, -0.2, 0.0, 0.3]))
        0.5
    """
    values = _finite_values(returns)
    if values is None:
        return np.nan
    return float(np.mean(values > 0))
