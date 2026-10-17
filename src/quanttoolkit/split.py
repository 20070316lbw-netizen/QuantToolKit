"""按共同交易日历把日期分成 train, valid, test, 并空出标签重叠的交易日"""

from __future__ import annotations

import pandas as pd


def split_dates(
    dates: pd.DatetimeIndex,
    *,
    train_end: str | pd.Timestamp,
    valid_end: str | pd.Timestamp,
    embargo: int,
) -> pd.Series:
    """按日期边界切分, 并丢掉上一段末尾会看到下一段价格的交易日

    比较的是排序去重后的共同日历, 不按自然日, 也不按单只证券自己的记录
    date <= train_end 先记为 train
    train_end < date <= valid_end 记为 valid
    date > valid_end 记为 test
    然后丢掉 train 和 valid 各自末尾 embargo 个交易日, 这些日期改为缺失
    valid 和 test 的第一天保持不变, test 的末尾不丢
    被丢掉的日期落在最终的上一段最后一天和下一段第一天之间, 两段都不用
    embargo 取未来收益标签的 gap + n_periods
    这样训练日标签用到的最后价格仍早于 valid 的第一天, valid 相对 test 同样成立
    不修改输入, 也不补齐交易日

    整个函数流程划分为: 
        对共同交易日历排序并去重
                |
        对齐端点时区：
        train_stamp = _align_timestamp(train_end, calendar.tz)
        valid_stamp = _align_timestamp(valid_end, calendar.tz)
                |
        初始化一个全部缺失的, index=calendar, dtype="object", name="split" 的 Series
                |
        按边界打标签：
        labels[calendar <= train_stamp] = "train"
        labels[(calendar > train_stamp) & (calendar <= valid_stamp)] = "valid"
        labels[calendar > valid_stamp] = "test"
                |
        最后分别丢掉 train 和 valid 的末尾
           


    Args:
        dates: 共同交易日历, 不要求排序或去重, 不含 NaT
        train_end: 训练段原来的最后一天, 含当天
            字符串或 Timestamp
            日历带时区而端点没有时区时, 按日历时区理解
        valid_end: 验证段原来的最后一天, 含当天
            应不早于 train_end, 类型规则与 train_end 相同
        embargo: train 和 valid 末尾各自丢掉的交易日数, 非负整数
            0 表示按边界相邻切分, 不丢掉日期
            段内天数少于 embargo 时, 这一段全部变为缺失

    Returns:
        名为 split 的 Series
        索引是排序去重后的 dates, 保留索引名, 时区和精度
        值为 train, valid 或 test, 被丢掉的日期为缺失
        空日历返回空 Series

    Raises:
        TypeError: 端点带时区而日历没有时区, 无法比较

    Example:
        >>> dates = pd.bdate_range("2024-01-01", periods=8, name="date")
        >>> result = split_dates(
        ...     dates, train_end="2024-01-04", valid_end="2024-01-09", embargo=1
        ... )
        >>> result.name
        'split'
        >>> result.fillna("embargo").tolist()
        ['train', 'train', 'train', 'embargo', 'valid', 'valid', 'embargo', 'test']
    """
    calendar = dates.sort_values().unique()
    train_stamp = _align_timestamp(train_end, calendar.tz)
    valid_stamp = _align_timestamp(valid_end, calendar.tz)
    labels = pd.Series(pd.NA, index=calendar, dtype="object", name="split")
    labels[calendar <= train_stamp] = "train"
    labels[(calendar > train_stamp) & (calendar <= valid_stamp)] = "valid"
    labels[calendar > valid_stamp] = "test"
    _drop_tail(labels, "train", embargo)
    _drop_tail(labels, "valid", embargo)
    return labels


def _align_timestamp(value: str | pd.Timestamp, tz) -> pd.Timestamp:
    """
    如果日历没有时区：直接返回端点。此时如果端点带时区，后续比较会抛出 TypeError
    如果日历有时区，而端点没有时区：用 tz_localize(tz)，按日历时区理解端点。
    如果日历有时区，端点也有时区：用 tz_convert(tz) 转成日历时区。
    """
    stamp = pd.Timestamp(value)
    if tz is None or stamp.tzinfo is not None:
        if tz is not None and stamp.tzinfo is not None:
            return stamp.tz_convert(tz)
        return stamp
    return stamp.tz_localize(tz)


def _drop_tail(labels: pd.Series, role: str, embargo: int) -> None:
    """
    1. 如果 embargo <= 0, 不处理
    2. 找出当前标签为 role 的所有日期。
    3. 把这些日期的最后 embargo 个改成 pd.NA
    """
    if embargo <= 0:
        return
    chosen = labels.index[labels == role]
    labels.loc[chosen[-embargo:]] = pd.NA
