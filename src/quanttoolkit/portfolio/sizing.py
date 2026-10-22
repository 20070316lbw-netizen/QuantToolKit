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


def vol_quantile_long_short(
    *,
    score: pd.Series,
    vol: pd.Series,
    n_vol_groups: int,
    n_quantiles: int,
    date_level: str = "date",
) -> pd.Series:
    """先按波动率分层, 再在每一层内按分数做多空

    每个日期先用 pd.qcut 把波动率分成 n_vol_groups 层
    每一层再按分数分成 n_quantiles 组, 最高组做多, 最低组做空
    各层选出的多头合并后等权, 权重和为 1
    空头合并后等权, 权重和为 -1
    中间组不出现在结果里
    重复的分位边界会被丢掉, 所以层数或极端组可能比请求的更少
    波动率切不开时, 整天当作一层
    缺失的分数或波动率先剔除, 只保留两边都有的股票
    不修改输入

    Args:
        score: [date, ticker] MultiIndex Series
            每个日期是一个截面分数, 通常只放入调仓日
        vol: 与 score 按索引对齐的波动率
            可以是 history_vol 的结果, 也可以是别的风险指标
            和 score 取交集后再计算
        n_vol_groups: 波动率分层数
        n_quantiles: 每一层内的分数分组数, 至少为 2
        date_level: 日期索引层名称, 默认 date

    Returns:
        名为 weight 的 Series
        索引是当天被选入多头或空头的 [date, ticker]
        多头为 1/多头股票数, 空头为 -1/空头股票数
        某一层有效股票少于 n_quantiles, 或该层实际分组少于 2 组时, 跳过这一层
        当天没有多头或没有空头时, 这一天不出现
        空输入返回空 Series

    Raises:
        ValueError: n_quantiles 小于 2
        KeyError: 索引没有 date_level 指定的层

    Example:
        >>> tickers = ["A", "B", "C", "D", "E", "F", "G", "H"]
        >>> index = pd.MultiIndex.from_product(
        ...     [pd.to_datetime(["2024-01-01"]), tickers],
        ...     names=["date", "ticker"],
        ... )
        >>> score = pd.Series([9, 7, 5, 3, 20, 16, 12, 4], index=index)
        >>> vol = pd.Series([1, 1, 1, 1, 9, 9, 9, 9], index=index)
        >>> weights = vol_quantile_long_short(
        ...     score=score, vol=vol, n_vol_groups=2, n_quantiles=2
        ... )
        >>> weights.name
        'weight'
        >>> sorted(weights[weights > 0].index.get_level_values("ticker"))
        ['A', 'B', 'E', 'F']
        >>> sorted(weights[weights < 0].index.get_level_values("ticker"))
        ['C', 'D', 'G', 'H']
        >>> [round(value, 2) for value in weights.tolist()]
        [0.25, 0.25, -0.25, -0.25, 0.25, 0.25, -0.25, -0.25]
    """

    def _weight(day: pd.DataFrame) -> pd.Series:
        if len(day) < n_vol_groups:
            return pd.Series(dtype=float)
        vol_bucket = pd.qcut(day["vol"], n_vol_groups, labels=False, duplicates="drop")
        if vol_bucket.nunique(dropna=True) < 1:
            vol_bucket = pd.Series(0, index=day.index)
        long_parts = []
        short_parts = []
        for _, layer in day.groupby(vol_bucket, sort=False):
            if len(layer) < n_quantiles:
                continue
            score_bucket = pd.qcut(
                layer["score"], n_quantiles, labels=False, duplicates="drop"
            )
            if score_bucket.nunique(dropna=True) < 2:
                continue
            scores = score_bucket.to_numpy()
            long_parts.append(layer.index[scores == score_bucket.max()])
            short_parts.append(layer.index[scores == score_bucket.min()])
        if not long_parts or not short_parts:
            return pd.Series(dtype=float)
        long_index = long_parts[0].append(long_parts[1:])
        short_index = short_parts[0].append(short_parts[1:])
        weights = pd.concat(
            [
                pd.Series(1.0 / len(long_index), index=long_index),
                pd.Series(-1.0 / len(short_index), index=short_index),
            ]
        )
        return weights

    if n_quantiles < 2:
        raise ValueError("n_quantiles 至少为 2, 才可以分出多头和空头")

    frame = pd.concat(
        [score.rename("score"), vol.rename("vol")],
        axis=1,
        join="inner",
    ).dropna()
    pieces = []
    for _, day in frame.groupby(level=date_level, sort=True):
        weights = _weight(day)
        if not weights.empty:
            pieces.append(weights)
    if not pieces:
        return pd.Series(dtype=float, index=frame.iloc[:0].index, name="weight")

    weighted = pd.concat(pieces)
    weighted.name = "weight"
    return weighted.sort_index()
