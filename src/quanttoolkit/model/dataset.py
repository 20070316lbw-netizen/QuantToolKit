"""把基准日已知的特征和之后发生的结果拼成训练样本

assemble_samples 把特征 X 和标签 y 按 [date, ticker] 对齐
X 可以是历史动量或波动率, y 可以是未来持有期收益
两边都有值的行才留下, 特征列在前, 标签列在最后

label_end_dates 用来看每条标签要等到哪一天才知道
基准记录为 t 时, future_returns 要用到 t + gap + n_periods 的价格
结束日期从完整行情索引生成, 再交给 split_samples 检查标签有没有跨段
"""

from __future__ import annotations

import pandas as pd


def label_end_dates(
    index: pd.MultiIndex, *, n_periods: int = 5, gap: int = 1
) -> pd.Series:
    """找出每条 future_returns 标签要等到哪一天才知道

    Args:
        index: 生成标签时完整行情的 [date, ticker] 索引, 键唯一
            不能使用删去缺失样本后的索引
        n_periods: 持有期记录数, 默认 5, 正整数, 必须与标签相同
        gap: 入场间隔记录数, 默认 1, 非负整数, 必须与标签相同

    Returns:
        排序后的 [date, ticker] Series, name 为 label_end
        值为同一证券 t + gap + n_periods 记录的日期, 保留时区
        未来记录不足时为 NaT, 空输入返回空 Series
        只按记录定位, 不检查该日期的价格是否缺失

    Raises:
        KeyError: 索引没有 date 或 ticker 层

    Example:
        >>> index = pd.MultiIndex.from_product(
        ...     [pd.to_datetime(["2024-01-02", "2024-01-05", "2024-01-08"]), ["A"]],
        ...     names=["date", "ticker"],
        ... )
        >>> ends = label_end_dates(index, n_periods=1, gap=1)
        >>> ends.iloc[0] == pd.Timestamp("2024-01-08")
        True
        >>> ends.isna().tolist()
        [False, True, True]
    """
    ordered = index.sort_values()
    dates = pd.Series(ordered.get_level_values("date"), index=ordered)
    return dates.groupby(level="ticker").shift(-(gap + n_periods)).rename("label_end")


def assemble_samples(*, features: pd.DataFrame, label: pd.Series) -> pd.DataFrame:
    """把特征和标签按 [date, ticker] 拼在一起, 只留下完整样本

    特征列在前, 标签列在最后, 列名使用 label.name
    标签通常来自 future_returns, 用的是未来价格
    只保留每个特征列和标签都有值的行
    只出现在一侧的索引不进入结果
    结果可以直接按 split_dates 的 train, valid, test 取行
    训练时把标签列单独取出, 其余列才交给模型作为特征
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
