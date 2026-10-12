"""按共同交易日历选择调仓日, 并映射每个日期所属的调仓区间"""

from __future__ import annotations

import pandas as pd


def rebalance_dates(dates: pd.DatetimeIndex, freq: int) -> pd.DatetimeIndex:
    """从最早日期开始, 每隔 freq 条共同交易日期选择一个调仓日

    Args:
        dates: 完整共同日历的 DatetimeIndex, 不要求排序或去重, 不含 NaT
            从行情的 date 层提取, 不按单只证券自己的记录选择调仓日
        freq: 无默认值, 正整数
            单位为排序去重后的日历记录数, 不是自然日数

    Returns:
        升序 DatetimeIndex, 保留索引名、时区和日期精度
        选择第 0、freq、2*freq 条日期, 不补齐缺失交易日
        空输入返回空索引; freq 超过日历长度时只返回首日

    Raises:
        ValueError: 切片步长 freq 为 0
        TypeError: 切片步长 freq 非整数

    Example:
        >>> dates = pd.bdate_range("2024-01-01", periods=7)
        >>> rebalance_dates(dates, freq=3).strftime("%Y-%m-%d").tolist()
        ['2024-01-01', '2024-01-04', '2024-01-09']
    """
    return dates.sort_values().unique()[::freq]


def rebalance_dates_belong(dates: pd.DatetimeIndex, freq: int) -> pd.Series:
    """将每个交易日期映射到最近已经发生的调仓日

    Args:
        dates: 完整共同日历的 DatetimeIndex, 不要求排序或去重, 不含 NaT
            应从标准 [date, ticker] 行情的 date 层提取
        freq: 无默认值, 正整数
            按共同日历的排序去重记录数计数, 不是自然日数

    Returns:
        DatetimeIndex Series, name 为 rebalance_date, 值为调仓日期
        索引为排序去重后的 dates, 值和索引均保留日期精度与时区
        区间为 [本次调仓日, 下次调仓日), 调仓日自己开启新区间
        最后一个区间覆盖剩余日期; 不补齐交易日, 无缺失输出
        空日历返回保留日期类型的空 Series

    Raises:
        ValueError: 非空日历的切片步长 freq 为 0
        TypeError: 非空日历的切片步长 freq 非整数

    Example:
        >>> dates = pd.bdate_range("2024-01-01", periods=4)
        >>> result = rebalance_dates_belong(dates, freq=3)
        >>> result.dt.strftime("%Y-%m-%d").tolist()
        ['2024-01-01', '2024-01-01', '2024-01-01', '2024-01-04']
    """
    calendar = dates.sort_values().unique()
    if calendar.empty:
        return pd.Series(index=calendar, dtype=calendar.dtype, name="rebalance_date")

    selected = calendar[::freq]
    # 右侧插入点让调仓日自身归入新区间, 非调仓日沿用最近已发生的起点
    positions = selected.searchsorted(calendar, side="right") - 1
    return pd.Series(selected[positions], index=calendar, name="rebalance_date")
