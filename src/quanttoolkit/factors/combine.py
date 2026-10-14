"""把已经标准化的因子列加权合成一个分数"""

from __future__ import annotations

import pandas as pd


def combine_scores(
    z: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.Series:
    """按行对标准化因子做加权平均

    某一行缺失的因子不参与该行
    分子是权重乘因子后求和
    分母是参与因子的权重绝对值之和
    因此权重不必事先加总为 1
    不修改输入

    Args:
        z: [date, ticker] MultiIndex DataFrame
            每一列是一个已经标准化的因子, 通常来自 zscore_by_date
        weights: 列名到权重的映射, 默认 None
            None 时每一列权重为 1
            面板中存在但 weights 未写出的列, 权重为 0
            权重可以是负数, 用来反转因子方向

    Returns:
        名为 score 的 Series, 索引与 z 相同
        某一行没有可参与的因子, 或参与权重绝对值之和为 0 时, 为 NaN

    Raises:
        KeyError: weights 中的列名不在 z 里

    Example:
        >>> index = pd.MultiIndex.from_product(
        ...     [pd.to_datetime(["2024-01-01"]), ["A", "B", "C"]],
        ...     names=["date", "ticker"],
        ... )
        >>> z = pd.DataFrame(
        ...     {"f1": [-1.0, 0.0, 1.0], "f2": [-0.71, float("nan"), 0.71]},
        ...     index=index,
        ... )
        >>> score = combine_scores(z)
        >>> score.name
        'score'
        >>> [round(float(value), 3) for value in score.tolist()]
        [-0.855, 0.0, 0.855]
    """
    if weights is None:
        weight = pd.Series(1.0, index=z.columns)
    else:
        missing = set(weights) - set(z.columns)
        if missing:
            raise KeyError(f"weights 含有因子面板中没有的列: {sorted(missing)}")
        weight = pd.Series(weights).reindex(z.columns).fillna(0.0)

    present = z.notna()
    numerator = z.fillna(0.0).mul(weight, axis=1).sum(axis=1)
    denominator = present.mul(weight.abs(), axis=1).sum(axis=1)
    score = (numerator / denominator).where(denominator > 0)
    score.name = "score"
    return score
