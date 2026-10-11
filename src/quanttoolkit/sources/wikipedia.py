"""抓取 Wikipedia 上最新的标普500成员名单"""

from io import StringIO
from urllib.request import Request, urlopen

import pandas as pd

from quanttoolkit.transfer import to_date_ticker_frame as _to_date_ticker_frame

__all__ = ["fetch_sp500_universe"]


def fetch_sp500_universe() -> pd.DataFrame:
    """抓取最新成员名单, 返回以抓取日期和证券代码为索引的快照

    Args:
        无参数, 固定访问 Wikipedia 当前成员页面, 请求超时为15秒

    Returns:
        pd.DataFrame: ['date', 'ticker'] MultiIndex, 按索引升序排列
        date 为运行机器本地的抓取日期, 不代表历史成员资格生效日期
        ticker 将点号替换为连字符, 如 BRK.B 转为 BRK-B
        列为 name, sector, sub_industry, date_added, cik
        date_added 为加入日期, 无法解析时为 NaT; cik 为保留前导零的字符串
        其他列保留页面文本, 空单元格为空字符串, 行数取决于当前成员表

    Raises:
        urllib.error.URLError: 页面请求失败
        TimeoutError: 请求超过15秒
        ValueError: 页面没有当前成员表或 HTML 解析失败
        KeyError: 当前成员表缺少使用的列

    Example:
        >>> from quanttoolkit.sources.wikipedia import fetch_sp500_universe
        >>> universe = fetch_sp500_universe()  # 需要网络连接
        >>> universe.index.names
        ['date', 'ticker']
        >>> universe.columns.tolist()
        ['name', 'sector', 'sub_industry', 'date_added', 'cik']
    """
    request = Request(
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        headers={"User-Agent": "QuantToolsKit/0.1 (S&P 500 research)"},
    )
    with urlopen(request, timeout=15) as response:
        html = response.read().decode("utf-8")

    universe = pd.read_html(
        StringIO(html),
        attrs={"id": "constituents"},
        flavor="lxml",
        keep_default_na=False,
        converters={"CIK": str},
    )[0].rename(
        columns={
            "Symbol": "ticker",
            "Security": "name",
            "GICS Sector": "sector",
            "GICS Sub-Industry": "sub_industry",
            "Date added": "date_added",
            "CIK": "cik",
        }
    )
    universe = universe[
        ["ticker", "name", "sector", "sub_industry", "date_added", "cik"]
    ].copy()
    universe["ticker"] = universe["ticker"].str.replace(".", "-", regex=False)
    universe["date_added"] = pd.to_datetime(universe["date_added"], errors="coerce")
    universe["date"] = pd.Timestamp.now().normalize()
    return _to_date_ticker_frame(df=universe)
