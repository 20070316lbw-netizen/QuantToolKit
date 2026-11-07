"""下载 SEC Company Facts 归档, 并将指定公司的事实拆成两张表"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen
from zipfile import ZipFile

import pandas as pd

__all__ = ["download_company_facts", "read_company_facts"]

_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"


def download_company_facts(destination: str | Path, *, user_agent: str) -> Path:
    """下载 SEC 归档, 检查 ZIP 后再替换目标文件

    Args:
        destination: 本地 ZIP 路径, 父目录须存在
        user_agent: 包含调用者名称和联系邮箱的 SEC User-Agent

    Returns:
        Path: 下载完成的本地归档路径

    Raises:
        OSError: 请求失败或文件无法写入
        ValueError: User-Agent 或归档为空
        BadZipFile: 下载内容无法作为 ZIP 打开

    Example:
        >>> archive = download_company_facts(  # doctest: +SKIP
        ...     "data/companyfacts.zip", user_agent="Your Name you@example.com"
        ... )
        >>> archive.name  # doctest: +SKIP
        'companyfacts.zip'
    """
    if not user_agent.strip():
        raise ValueError("SEC User-Agent 须包含调用者名称和联系邮箱")
    destination = Path(destination)
    request = Request(_ARCHIVE_URL, headers={"User-Agent": user_agent})
    with TemporaryDirectory(dir=destination.parent) as directory:
        temporary = Path(directory) / "companyfacts.zip"
        with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            copyfileobj(response, output, length=1024 * 1024)
        with ZipFile(temporary) as archive:
            if not archive.namelist():
                raise ValueError("SEC Company Facts 归档为空")
        temporary.replace(destination)
    return destination


PERIOD_COLUMNS = [
    "cik",
    "taxonomy",
    "concept",
    "unit",
    "period_start",
    "period_end",
    "val",
    "accn",
    "form",
    "filed",
    "fy",
    "fp",
    "frame",
]

INSTANT_COLUMNS = [
    "cik",
    "taxonomy",
    "concept",
    "unit",
    "period_end",
    "val",
    "accn",
    "form",
    "filed",
    "fy",
    "fp",
    "frame",
]


def _columns(names: list[str]) -> dict[str, list]:
    return {name: [] for name in names}


def _append(
    bucket: dict[str, list],
    cik: str,
    taxonomy: str,
    concept: str,
    unit: str,
    fact: dict,
) -> None:
    bucket["cik"].append(cik)
    bucket["taxonomy"].append(taxonomy)
    bucket["concept"].append(concept)
    bucket["unit"].append(unit)
    if "period_start" in bucket:
        bucket["period_start"].append(fact["start"])
    bucket["period_end"].append(fact["end"])
    bucket["val"].append(fact["val"])
    bucket["accn"].append(fact["accn"])
    bucket["form"].append(fact["form"])
    bucket["filed"].append(fact["filed"])
    bucket["fy"].append(fact.get("fy"))
    bucket["fp"].append(fact.get("fp"))
    bucket["frame"].append(fact.get("frame"))


def _as_table(data: dict[str, list], date_columns: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {column: pd.Series(values, dtype=object) for column, values in data.items()}
    )
    for column in date_columns:
        frame[column] = pd.to_datetime(frame[column])
    frame["val"] = pd.to_numeric(frame["val"]).astype("float64")
    frame["fy"] = pd.to_numeric(frame["fy"]).astype("Int64")
    return frame


def read_company_facts(
    zip_path: str | Path, ciks: Iterable[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """读取指定 CIK 的公司事实, 按期间数和时点数拆开

    Args:
        zip_path: companyfacts.zip 的本地路径, 支持 str 或 Path
        ciks: 10位 CIK, 对应压缩包内的 CIK##########.json

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: 依次为 period_facts, instant_facts 的列
        带 start 的事实进入第一张表, 其余进入第二张表
        日期列为 datetime64, fy 为可空整数
        缺失的 frame, fy, fp 为空值
        没有事实时返回带列名的空表

    Raises:
        FileNotFoundError: zip 不存在
        KeyError: 某个 CIK 在压缩包中没有对应文件
        json.JSONDecodeError: 公司文件不是有效 JSON

    Example:
        >>> from quanttoolkit.sources.companyfacts import read_company_facts
        >>> period, instant = read_company_facts(
        ...     "data/companyfacts.zip", ["0000320193"]
        ... )
        >>> period.columns[-1]
        'frame'
        >>> "period_start" in instant.columns
        False
    """
    period = _columns(PERIOD_COLUMNS)
    instant = _columns(INSTANT_COLUMNS)
    with ZipFile(zip_path) as archive:
        for cik in dict.fromkeys(ciks):
            payload = json.loads(archive.read(f"CIK{cik}.json"))
            for taxonomy, concepts in (payload.get("facts") or {}).items():
                for concept, node in concepts.items():
                    for unit, facts in (node.get("units") or {}).items():
                        for fact in facts:
                            bucket = period if "start" in fact else instant
                            _append(bucket, cik, taxonomy, concept, unit, fact)
    return (
        _as_table(period, ("period_start", "period_end", "filed")),
        _as_table(instant, ("period_end", "filed")),
    )
