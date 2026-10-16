"""用申报日面板和行情计算基本面因子"""

from __future__ import annotations

import pandas as pd

from quanttoolkit.fundamentals import align_filings

__all__ = ["asset_growth", "book_to_market", "earnings_yield"]


def _cap(aligned: pd.DataFrame) -> pd.Series:
    cap = aligned["close"] * aligned["shares"]
    return cap.where(cap > 0)


def book_to_market(prices: pd.DataFrame, filings: pd.DataFrame) -> pd.Series:
    """计算账面市值比, 权益除以当天市值

    市值是 close 乘以股数, 不用 adj_close
    权益和股数来自各自不晚于当天的最后一次披露, 两次披露之间沿用
    不修改输入

    Args:
        prices: [date, ticker] MultiIndex DataFrame, 含 close 列
        filings: 申报日面板, 列 ticker, filed, equity, shares
            通常来自 filing_panel

    Returns:
        排序后的 [date, ticker] Series, name 为 book_to_market
        值为 equity / (close * shares), 无量纲
        市值不是正数, 或当天之前没有权益或股数时为 NaN

    Raises:
        KeyError: close, equity, shares, filed 或行情索引层不存在

    Example:
        >>> index = pd.MultiIndex.from_product(
        ...     [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["A"]],
        ...     names=["date", "ticker"],
        ... )
        >>> prices = pd.DataFrame({"close": [5.0, 10.0]}, index=index)
        >>> filings = pd.DataFrame({
        ...     "ticker": ["A"],
        ...     "filed": pd.to_datetime(["2024-01-02"]),
        ...     "equity": [50.0],
        ...     "shares": [2.0],
        ... })
        >>> book_to_market(prices, filings).tolist()
        [5.0, 2.5]
    """

    aligned = align_filings(prices, filings)
    factor = (aligned["equity"] / _cap(aligned)).rename("book_to_market")
    return factor


def earnings_yield(prices: pd.DataFrame, filings: pd.DataFrame) -> pd.Series:
    """计算盈利收益率, 滚动四季净利润除以当天市值

    市值是 close 乘以股数, 不用 adj_close
    净利润和股数来自各自不晚于当天的最后一次披露
    不修改输入

    Args:
        prices: [date, ticker] MultiIndex DataFrame, 含 close 列
        filings: 申报日面板, 列 ticker, filed, net_income_ttm, shares

    Returns:
        排序后的 [date, ticker] Series, name 为 earnings_yield
        值为 net_income_ttm / (close * shares), 无量纲
        市值不是正数, 或当天之前没有净利润或股数时为 NaN

    Raises:
        KeyError: close, net_income_ttm, shares, filed 或行情索引层不存在

    Example:
        >>> index = pd.MultiIndex.from_product(
        ...     [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["A"]],
        ...     names=["date", "ticker"],
        ... )
        >>> prices = pd.DataFrame({"close": [5.0, 10.0]}, index=index)
        >>> filings = pd.DataFrame({
        ...     "ticker": ["A"],
        ...     "filed": pd.to_datetime(["2024-01-02"]),
        ...     "net_income_ttm": [8.0],
        ...     "shares": [2.0],
        ... })
        >>> earnings_yield(prices, filings).tolist()
        [0.8, 0.4]
    """

    aligned = align_filings(prices, filings)
    factor = (aligned["net_income_ttm"] / _cap(aligned)).rename("earnings_yield")
    return factor


def asset_growth(prices: pd.DataFrame, filings: pd.DataFrame) -> pd.Series:
    """把申报日上的资产同比增长对齐到每个交易日

    增长率在 filing_panel 里按披露日算好
    当前资产除以大约一年前的资产再减 1
    新的资产披露如果算不出增长率, 当天起为缺失, 不沿用上一份
    不修改输入

    Args:
        prices: [date, ticker] MultiIndex DataFrame, 只用它的日期和证券
        filings: 申报日面板, 列 ticker, filed, asset_growth

    Returns:
        排序后的 [date, ticker] Series, name 为 asset_growth
        值为小数增长率
        当天之前没有可计算的资产增长时为 NaN

    Raises:
        KeyError: asset_growth, filed 或行情索引层不存在

    Example:
        >>> index = pd.MultiIndex.from_product(
        ...     [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["A"]],
        ...     names=["date", "ticker"],
        ... )
        >>> prices = pd.DataFrame({"close": [5.0, 10.0]}, index=index)
        >>> filings = pd.DataFrame({
        ...     "ticker": ["A"],
        ...     "filed": pd.to_datetime(["2024-01-02"]),
        ...     "asset_growth": [0.5],
        ... })
        >>> asset_growth(prices, filings).tolist()
        [0.5, 0.5]
    """

    return align_filings(prices, filings)["asset_growth"].rename("asset_growth")
