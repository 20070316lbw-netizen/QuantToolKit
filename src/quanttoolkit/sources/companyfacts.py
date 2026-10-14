"""从 companyfacts.zip 读取 SEC 公司事实"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

__all__ = ["read_company_facts"]

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
        for cik in ciks:
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
