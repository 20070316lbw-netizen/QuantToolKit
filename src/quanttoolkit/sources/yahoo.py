"""Yahoo 数据抓取"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime

import pandas as pd
from loguru import logger

from quanttoolkit.transfer import to_date_ticker_frame

DateLike = str | date | datetime

_COLUMN_MAP = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}

_OUTPUT_COLUMNS = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
]
_NUMERIC_COLUMNS = ["open", "high", "low", "close", "adj_close", "volume"]


def get_prices(
    tickers: str | Sequence[str],
    start: DateLike,
    end: DateLike | None = None,
    *,
    interval: str = "1d",
    auto_adjust: bool = False,
    progress: bool = True,
) -> pd.DataFrame:
    """抓取一批股票的历史行情, 返回标准 [date, ticker] 行情面板

    Args:
        tickers: 单个 ticker 或 ticker 列表, 如 "AAPL" 或 ["AAPL", "MSFT"]
        start: 起始日期(含), 如 "2020-01-01"
        end: 结束日期(不含); 默认到今天(由 yfinance 决定)
        interval: yfinance 支持的周期, 如 "1d"/"1wk"/"1mo"
        auto_adjust: 透传给 yfinance; True 时 yfinance 直接返回复权价并可能
            丢弃 Adj Close 列, False(默认)保留原始 close 与 adj_close 两列
        progress: 是否显示 yfinance 下载进度条, 默认 True, 仅限关键词传入
            进度按处理完成的 ticker 数量推进, 包含失败任务, 不包括清洗和入库

    Returns:
        [date, ticker] MultiIndex DataFrame, 按索引升序排序
        列为 open, high, low, close, adj_close, volume, 单位沿用 Yahoo 行情
        close 缺失的记录删除, 其他缺失字段保留缺失值
        找不到数据时返回保留索引和列结构的空 DataFrame

    Raises:
        ValueError: 多证券响应缺少 ticker 列层级

    Example:
        下载示例需联网
        >>> prices = get_prices("AAPL", start="2024-01-01", end="2024-02-01", progress=False)
        >>> list(prices.index.names)
        ['date', 'ticker']
    """
    ticker_list = [tickers] if isinstance(tickers, str) else list(tickers)

    logger.info(
        f"Fetching prices for {len(ticker_list)} ticker(s) "
        f"from {start} to {end or 'today'} (interval={interval})"
    )

    import yfinance as yf

    raw = yf.download(
        ticker_list,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        group_by="ticker",
        progress=progress,
        threads=True,
    )

    if raw is None or raw.empty:
        logger.warning("yfinance 未返回任何数据")
        return _empty_frame()

    frames = []
    if isinstance(raw.columns, pd.MultiIndex):
        # 多 ticker: 列是 (ticker, field) 两层
        available = set(raw.columns.get_level_values(0))
        for ticker in ticker_list:
            if ticker not in available:
                logger.warning(f"{ticker}: yfinance 无数据, 跳过")
                continue
            frames.append(_clean_single(raw[ticker].copy(), ticker, auto_adjust))  # type: ignore
    else:
        # 单 ticker: 列是单层
        if len(ticker_list) != 1:
            raise ValueError("多证券行情缺少 ticker 列层级, 无法确认证券身份")
        frames.append(_clean_single(raw.copy(), ticker_list[0], auto_adjust))

    if not frames:
        return _empty_frame()

    result = pd.concat(frames, ignore_index=True)
    return to_date_ticker_frame(df=result)


def _clean_single(
    df: pd.DataFrame, ticker: str, auto_adjust: bool = False
) -> pd.DataFrame:
    """把 yfinance 单 ticker 的宽表整理成标准长表 schema"""
    df = df.rename(columns=_COLUMN_MAP)
    df = df.reset_index().rename(columns={"Date": "date", "Datetime": "date"})
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df["ticker"] = ticker

    # 只有数据源明确已复权时, close 才能当作 adj_close; 缺少复权价时保留空值
    if "adj_close" not in df.columns and auto_adjust:
        df["adj_close"] = df.get("close")

    for col in _OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[_OUTPUT_COLUMNS]

    df[_NUMERIC_COLUMNS] = df[_NUMERIC_COLUMNS].apply(pd.to_numeric, errors="coerce")

    before = len(df)
    df = df.dropna(subset=["close"])
    dropped = before - len(df)
    if dropped:
        logger.debug(f"{ticker}: 丢弃 {dropped} 行 close 缺失的记录")

    return df


def _empty_frame() -> pd.DataFrame:
    return to_date_ticker_frame(df=pd.DataFrame(columns=_OUTPUT_COLUMNS))
