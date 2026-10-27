"""先按共同交易日历分出 train, valid, test, 再处理标签跨段

split_dates 在训练和验证段末尾留出 embargo 个交易日
split_samples 再检查每条标签实际结束的日期, 去掉仍然跨段的样本
"""

from __future__ import annotations

import pandas as pd


def split_samples(
    *,
    samples: pd.DataFrame,
    dates: pd.DatetimeIndex,
    label_end: pd.Series,
    train_end: str | pd.Timestamp,
    valid_end: str | pd.Timestamp,
    embargo: int,
) -> dict[str, pd.DataFrame]:
    """按共同日历切分样本, 再排除实际标签结束日期跨段的行

    Args:
        samples: assemble_samples 返回的 [date, ticker] DataFrame
        dates: 完整行情的共同交易日历, 不能从 samples 反推
        label_end: label_end_dates 在完整行情索引上生成的 Series
            使用与标签相同的 gap 和 n_periods
        train_end: 训练段边界, 含当天, 类型和时区规则同 split_dates
        valid_end: 验证段边界, 含当天, 不早于 train_end
        embargo: 非负交易日数, 一般取 gap + n_periods

    Returns:
        键为 train, valid, test 的 dict, 值为排序后的样本表
        train 的标签结束日期不超过 train_end, valid 不超过 valid_end
        日历以外, 隔离期或标签结束日期缺失的行不进入任一段
        test 不额外限制标签结束日期, 各段允许为空, 不修改输入

    Raises:
        ValueError: train_end 晚于 valid_end, 或 embargo 为负数
        TypeError: 无时区日历与有时区端点不能比较

    Example:
        >>> from quanttoolkit.model.dataset import label_end_dates
        >>> dates = pd.bdate_range("2024-01-01", periods=8, name="date")
        >>> index = pd.MultiIndex.from_product([dates, ["A"]], names=["date", "ticker"])
        >>> samples = pd.DataFrame({"x": 1.0, "y": 0.1}, index=index)
        >>> parts = split_samples(
        ...     samples=samples, dates=dates,
        ...     label_end=label_end_dates(index, n_periods=1, gap=0),
        ...     train_end=dates[3], valid_end=dates[5], embargo=1,
        ... )
        >>> [len(parts[role]) for role in ("train", "valid", "test")]
        [3, 1, 1]
    """
    roles = split_dates(
        dates, train_end=train_end, valid_end=valid_end, embargo=embargo
    )
    ordered = samples.sort_index()
    row_roles = roles.reindex(ordered.index.get_level_values("date"))
    ends = label_end.reindex(ordered.index)
    limits = {
        "train": _align_timestamp(train_end, roles.index.tz),
        "valid": _align_timestamp(valid_end, roles.index.tz),
    }
    parts = {}
    for role in ("train", "valid", "test"):
        keep = row_roles.eq(role).to_numpy() & ends.notna().to_numpy()
        if role in limits:
            keep &= ends.le(limits[role]).to_numpy()
        parts[role] = ordered.loc[keep].copy()
    return parts


def split_dates(
    dates: pd.DatetimeIndex,
    *,
    train_end: str | pd.Timestamp,
    valid_end: str | pd.Timestamp,
    embargo: int,
) -> pd.Series:
    """按日期边界切分, 并丢掉上一段末尾会看到下一段价格的交易日

    先对共同日历排序去重, 再把边界对齐到日历时区
    date <= train_end 记为 train
    train_end < date <= valid_end 记为 valid, 其余日期记为 test
    最后把 train 和 valid 各自末尾 embargo 个交易日改成缺失, 两段都不用
    valid 和 test 的第一天不动, test 末尾也不丢

    未来收益标签通常取 embargo = gap + n_periods
    证券记录与共同日历对应时, 这样标签最后看到的价格就不会进入下一段
    证券缺行时还要检查实际标签结束日期, 可以接着用 split_samples
    不修改输入, 也不补齐交易日

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
        ValueError: train_end 晚于 valid_end, 或 embargo 为负数
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
    if train_stamp > valid_stamp or embargo < 0:
        raise ValueError("train_end 不能晚于 valid_end, embargo 不能为负数")
    labels = pd.Series(pd.NA, index=calendar, dtype="object", name="split")
    labels[calendar <= train_stamp] = "train"
    labels[(calendar > train_stamp) & (calendar <= valid_stamp)] = "valid"
    labels[calendar > valid_stamp] = "test"
    _drop_tail(labels, "train", embargo)
    _drop_tail(labels, "valid", embargo)
    return labels


def _align_timestamp(value: str | pd.Timestamp, tz) -> pd.Timestamp:
    """把日期边界对齐到共同日历的时区

    日历没有时区时直接返回端点, 若端点带时区, 后续比较会抛出 TypeError
    日历有时区而端点没有时区时, 用 tz_localize(tz) 按日历时区理解端点
    两边都有时区时, 用 tz_convert(tz) 把端点转成日历时区
    """
    stamp = pd.Timestamp(value)
    if tz is None or stamp.tzinfo is not None:
        if tz is not None and stamp.tzinfo is not None:
            return stamp.tz_convert(tz)
        return stamp
    return stamp.tz_localize(tz)


def _drop_tail(labels: pd.Series, role: str, embargo: int) -> None:
    """先找出属于 role 的日期, 再把末尾 embargo 个改成 pd.NA

    embargo <= 0 时保留原样
    """
    if embargo <= 0:
        return
    chosen = labels.index[labels == role]
    labels.loc[chosen[-embargo:]] = pd.NA
