"""在每个日期内减去同行业均值, 去掉因子里的行业平均差异"""

from __future__ import annotations

import pandas as pd


def neutralize_by_industry(
    factors: pd.DataFrame,
    industry: pd.Series,
    *,
    date_level: str = "date",
    ticker_level: str = "ticker",
) -> pd.DataFrame:
    """在每个日期内, 按行业减去该行业的截面均值

    每一列单独中性化
    行业内有效值取均值时忽略缺失值
    没有行业标签的证券不进入任何行业, 结果为 NaN
    行业内只有一个有效值时, 该值减去自身, 结果为 0
    中性化之后可交给已有的 zscore_by_date 和 combine_scores
    不修改输入

    Args:
        factors: [date, ticker] MultiIndex DataFrame
            每一列是一个因子, 列名沿用输入
        industry: 索引为 ticker 的 Series, 值是行业标签
            一只证券一行, 按 factors 的 ticker 层对齐
        date_level: 日期索引层名称, 默认 date
        ticker_level: 证券索引层名称, 默认 ticker

    Returns:
        与 factors 同索引, 同列名的 DataFrame
        数值是原因子减去当天同行业均值
        行业标签缺失, 或该行业当天该列没有有效值时, 为 NaN
        空输入返回同结构的空 DataFrame

    Raises:
        KeyError: 索引没有 date_level 或 ticker_level 指定的层

    Example:
        >>> index = pd.MultiIndex.from_product(
        ...     [pd.to_datetime(["2024-01-02"]), ["A", "B", "C"]],
        ...     names=["date", "ticker"],
        ... )
        >>> factors = pd.DataFrame({"f1": [1.0, 3.0, 10.0]}, index=index)
        >>> industry = pd.Series(["Tech", "Tech", "Bank"], index=["A", "B", "C"])
        >>> neutralize_by_industry(factors, industry)["f1"].tolist()
        [-1.0, 1.0, 0.0]
    """
    tickers = factors.index.get_level_values(ticker_level)
    labels = pd.Series(industry.reindex(tickers).to_numpy(), index=factors.index)
    means = factors.groupby(
        [date_level, labels],
        dropna=True,
        sort=False,
    ).transform("mean")
    return factors - means
