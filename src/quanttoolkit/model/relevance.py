"""把每天的未来收益分成整数档, 交给 LambdaRank 学习股票排序"""

from __future__ import annotations

import numpy as np
import pandas as pd


def relevance_by_date(label: pd.Series, *, n_grades: int = 31) -> pd.Series:
    """在每个日期内把收益分成整数档, 收益越高档位越高

    档位从 0 开始, 最高是 n_grades - 1
    默认 31 档, 对应 LightGBM 默认 label_gain 能接受的 0 到 30
    同一天里的并列收益按当前行顺序拆开, 不取平均
    缺失收益保持缺失, 不参与当天的分档
    某一天没有有效收益时, 这一天全部为缺失
    不修改输入

    Args:
        label: [date, ticker] Series, 通常是 future_returns 的结果
            值越大表示越相关
        n_grades: 档数, 默认 31, 正整数
            使用 LambdaRankParams 且不改 label_gain 时不要超过 31

    Returns:
        排序后的 [date, ticker] Series, name 为 relevance_{n_grades}
        有效位置是 0 到 n_grades - 1 的整数, 缺失位置保持缺失
        空输入返回空 Series

    Raises:
        KeyError: 索引没有 date 层

    Example:
        >>> index = pd.MultiIndex.from_product(
        ...     [pd.to_datetime(["2024-01-02"]), ["A", "B", "C", "D"]],
        ...     names=["date", "ticker"],
        ... )
        >>> label = pd.Series([0.1, 0.4, 0.2, 0.3], index=index)
        >>> relevance_by_date(label, n_grades=4).tolist()
        [0, 3, 1, 2]
    """
    order = label.groupby(level="date").rank(method="first", ascending=True)
    count = label.groupby(level="date").transform("count")
    scaled = (order - 1) * n_grades / count.where(count > 0)
    relevance = np.floor(scaled).astype("Int64")
    relevance.name = f"relevance_{n_grades}"
    return relevance.sort_index()
