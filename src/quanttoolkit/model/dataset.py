"""把因子面板和未来收益标签对齐成监督学习样本"""

from __future__ import annotations

import pandas as pd


def assemble_samples(*, features: pd.DataFrame, label: pd.Series) -> pd.DataFrame:
    """按 [date, ticker] 内连接特征与标签, 丢掉不完整的行

    特征列在前, 标签列在最后, 列名使用 label.name
    标签通常来自 future_returns, 用的是未来价格
    只保留每个特征列和标签都有值的行
    只出现在一侧的索引不进入结果
    结果可以直接按 split_dates 的 train, valid, test 取行
    除标签列以外的列交给 LightGBM 作为特征
    不修改输入, 也不填补缺失值

    Args:
        features: [date, ticker] MultiIndex DataFrame
            每一列是基准日已经知道的特征
            索引名须与 label 相同, 键唯一
        label: 与 features 使用同类索引的 Series
            name 是标签列名, 通常为 forward_return_{n_periods}_gap_{gap}
            不能与特征列同名, 索引名和唯一性规则与 features 相同

    Returns:
        排序后的 [date, ticker] DataFrame
        列顺序为全部特征列, 最后一列是标签
        没有完整样本时, 返回保留列名和索引名的空 DataFrame

    Raises:
        ValueError: label 没有 name, 两侧索引名不一致, 或 [date, ticker] 不唯一
        KeyError: label.name 已经是特征列名

    Example:
        >>> index = pd.MultiIndex.from_product(
        ...     [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["A", "B"]],
        ...     names=["date", "ticker"],
        ... )
        >>> features = pd.DataFrame({"mom_2": [0.1, 0.2, float("nan"), 0.4]}, index=index)
        >>> label = pd.Series(
        ...     [0.01, float("nan"), 0.03, 0.04],
        ...     index=index,
        ...     name="forward_return_1_gap_1",
        ... )
        >>> samples = assemble_samples(features=features, label=label)
        >>> samples.columns.tolist()
        ['mom_2', 'forward_return_1_gap_1']
        >>> samples["mom_2"].tolist()
        [0.1, 0.4]
    """
    if label.name is None:
        raise ValueError("label 必须带 name 作为标签列名")
    if label.name in features.columns:
        raise KeyError(f"标签列名已经在特征里: {label.name}")
    if features.index.names != label.index.names:
        raise ValueError(f"索引名不一致: {features.index.names} vs {label.index.names}")
    if not (features.index.is_unique and label.index.is_unique):
        raise ValueError("[date, ticker] 索引必须唯一")
    aligned = features.join(label, how="inner")
    aligned = aligned.loc[aligned.notna().all(axis=1)]
    return aligned.sort_index()
