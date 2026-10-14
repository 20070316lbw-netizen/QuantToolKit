"""按分数构造多空分位权重"""

from __future__ import annotations

import pandas as pd


def simple_quantile_long_short(
    *,
    score: pd.Series,
    n_quantiles: int,
    date_level: str = "date",
) -> pd.Series:
    """把每个日期的分数分成 n_quantiles 组, 最高组等权做多, 最低组等权做空

    中间组权重为 0
    多头权重和为 1, 空头权重和为 -1
    分组使用 pd.qcut, 重复边界会被丢掉, 所以极端组可能比 1/n_quantiles 更宽
    缺失分数先剔除, 不记为 0
    不修改输入

    Args:
        score: [date, ticker] MultiIndex Series
            每个日期是一个截面分数, 通常只放入调仓日
        n_quantiles: 分组数, 至少为 2
            5 表示五分位, 最高和最低大约各占 20%
        date_level: 日期索引层名称, 默认 date

    Returns:
        名为 weight 的 Series, 索引是剔除缺失值后的 [date, ticker]
        多头为 1/多头股票数, 空头为 -1/空头股票数, 中间组为 0
        某一天有效股票少于 n_quantiles, 或实际分组少于 2 组时, 该天不出现
        空输入返回空 Series

    Raises:
        ValueError: n_quantiles 小于 2
        KeyError: 索引没有 date_level 指定的层

    Example:
        >>> index = pd.MultiIndex.from_product(
        ...     [pd.to_datetime(["2024-01-01"]), ["A", "B", "C", "D", "E", "F"]],
        ...     names=["date", "ticker"],
        ... )
        >>> score = pd.Series([5.0, 3.0, 1.0, -1.0, -3.0, -5.0], index=index)
        >>> weights = simple_quantile_long_short(score=score, n_quantiles=3)
        >>> weights.name
        'weight'
        >>> weights.tolist()
        [0.5, 0.5, 0.0, 0.0, -0.5, -0.5]
    """

    def _weight(scores: pd.Series) -> pd.Series:
        if len(scores) < n_quantiles:
            return pd.Series(dtype=float)

        bucket = pd.qcut(scores, n_quantiles, labels=False, duplicates="drop")
        if bucket.nunique(dropna=True) < 2:
            return pd.Series(dtype=float)

        weights = pd.Series(0.0, index=scores.index)
        long_mask = bucket == bucket.max()
        short_mask = bucket == bucket.min()
        weights[long_mask] = 1.0 / int(long_mask.sum())
        weights[short_mask] = -1.0 / int(short_mask.sum())
        return weights

    if n_quantiles < 2:
        raise ValueError("n_quantiles 至少为 2, 才可以分出多头和空头")

    pieces = []
    for _, scores in score.dropna().groupby(level=date_level, sort=True):
        weights = _weight(scores)
        if not weights.empty:
            pieces.append(weights)
    if not pieces:
        return pd.Series(dtype=float, index=score.iloc[:0].index, name="weight")

    weighted = pd.concat(pieces)
    weighted.name = "weight"
    return weighted.sort_index()
