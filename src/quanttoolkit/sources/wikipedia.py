"""从 Wikipedia 抓取当前标普500成员, 整理成 universe 需要的三列"""

from io import StringIO
from urllib.request import Request, urlopen

import pandas as pd

__all__ = ["fetch_sp500_universe"]


def fetch_sp500_universe() -> pd.DataFrame:
    """抓取最新成员名单, 返回符合 universe 表 DDL 的数据

    请求最多等待 15 秒

    Returns:
        pd.DataFrame: RangeIndex, 按 ticker 升序排列
        列为 ticker, name, sub_industry
        ticker 将点号替换为连字符, 如 BRK.B 转为 BRK-B
        字段为页面文本, 空单元格为空字符串, 行数取决于当前成员表

    Raises:
        urllib.error.URLError: 页面请求失败
        TimeoutError: 请求超过 15 秒
        ValueError: 页面没有当前成员表或 HTML 解析失败
        KeyError: 当前成员表缺少使用的列

    Example:
        >>> from quanttoolkit.sources.wikipedia import fetch_sp500_universe
        >>> universe = fetch_sp500_universe()
        >>> universe.columns.tolist()
        ['ticker', 'name', 'sub_industry']
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
    )[0].rename(
        columns={
            "Symbol": "ticker",
            "Security": "name",
            "GICS Sub-Industry": "sub_industry",
        }
    )
    universe = universe[["ticker", "name", "sub_industry"]].copy()
    universe["ticker"] = universe["ticker"].str.replace(".", "-", regex=False)
    return universe.sort_values("ticker").reset_index(drop=True)
