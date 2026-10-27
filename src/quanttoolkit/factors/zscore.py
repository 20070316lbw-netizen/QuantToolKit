"""在同一天的股票之间计算 z-score, 统一各因子的尺度"""

from __future__ import annotations

import pandas as pd


def zscore_by_date(
    factors: pd.DataFrame,
    *,
    date_level: str = "date",
) -> pd.DataFrame:
    """把每一列因子变成当天截面上的 z-score

    在每个日期上, 对每一列分别计算
    z = (原始值 - 当天均值) / 当天样本标准差
    均值和标准差忽略缺失值, 标准差使用 ddof=1
    不修改输入

    Args:
        factors: [date, ticker] MultiIndex DataFrame
            每一列是一个因子, 列名沿用输入
        date_level: 日期索引层名称, 默认 date

    Returns:
        与 factors 同索引, 同列名的 DataFrame
        数值是无量纲 z-score
        当天该列有效值少于 2 个, 或标准差为 0 时, 这些位置为 NaN
        空输入返回同结构的空 DataFrame

    Raises:
        KeyError: 索引没有 date_level 指定的层

    Example:
        >>> index = pd.MultiIndex.from_product(
        ...     [pd.to_datetime(["2024-01-01"]), ["A", "B", "C"]],
        ...     names=["date", "ticker"],
        ... )
        >>> factors = pd.DataFrame({"f1": [1.0, 2.0, 3.0]}, index=index)
        >>> zscore_by_date(factors)["f1"].tolist()
        [-1.0, 0.0, 1.0]
    """
    grouped = factors.groupby(level=date_level)
    return (factors - grouped.transform("mean")) / grouped.transform("std")
