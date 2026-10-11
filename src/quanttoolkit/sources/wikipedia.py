"""抓取 Wikipedia 上最新的标普500成员名单"""

import json
from io import StringIO
from pathlib import Path
from urllib.request import Request, urlopen

import pandas as pd

__all__ = ["fetch_sp500_universe"]


def fetch_sp500_universe(json_path: str | Path) -> pd.DataFrame:
    """抓取最新成员名单, 返回符合 universe 表 DDL 的数据

    Args:
        json_path: SEC company_tickers.json 的本地路径, 必填, 支持 str 或 Path
            必须先按 sources/USAGE.md 的官方映射表获取方法下载文件
            Wikipedia 请求超时为15秒

    Returns:
        pd.DataFrame: RangeIndex, 按 ticker 升序排列
        列为 ticker, name, sub_industry, cik, sec_title
        ticker 将点号替换为连字符, 如 BRK.B 转为 BRK-B
        cik 为10位字符串, sec_title 按 CIK 映射 JSON 中的 title, 未匹配时为缺失值
        其他字段为页面文本, 空单元格为空字符串, 行数取决于当前成员表

    Raises:
        FileNotFoundError: 没有映射文件, 需先按 sources/USAGE.md 下载
        json.JSONDecodeError: 映射文件不是有效 JSON
        urllib.error.URLError: 页面请求失败
        TimeoutError: 请求超过15秒
        ValueError: 页面没有当前成员表或 HTML 解析失败
        KeyError: 当前成员表缺少使用的列

    Example:
        >>> from quanttoolkit.sources.wikipedia import fetch_sp500_universe
        >>> universe = fetch_sp500_universe("data/company_tickers.json")
        >>> universe.columns.tolist()
        ['ticker', 'name', 'sub_industry', 'cik', 'sec_title']
    """
    json_path = Path(json_path)
    if not json_path.is_file():
        raise FileNotFoundError(
            f"没有映射文件: {json_path}, "
            "请先按 src/quanttoolkit/sources/USAGE.md 的官方映射表获取方法下载"
        )
    with json_path.open(encoding="utf-8") as file:
        sec_titles = {
            str(row["cik_str"]).zfill(10): row["title"]
            for row in json.load(file).values()
        }

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
            "GICS Sub-Industry": "sub_industry",
            "CIK": "cik",
        }
    )
    universe = universe[["ticker", "name", "sub_industry", "cik"]].copy()
    universe["ticker"] = universe["ticker"].str.replace(".", "-", regex=False)
    universe["cik"] = universe["cik"].str.zfill(10)
    universe["sec_title"] = universe["cik"].map(sec_titles)
    return universe.sort_values("ticker").reset_index(drop=True)
