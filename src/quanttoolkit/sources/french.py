"""读取 Kenneth French 日频因子压缩包"""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

__all__ = ["read_french_factors"]

_COLUMN_MAP = {
    "Mkt-RF": "mkt_rf",
    "SMB": "smb",
    "HML": "hml",
    "RMW": "rmw",
    "CMA": "cma",
    "RF": "rf",
}


def read_french_factors(path: str | Path) -> pd.DataFrame:
    """读取官方日频因子 zip, 返回小数收益率

    压缩包内是一份 CSV
    表头之前的说明不读, 数据表后的空行和版权声明不读
    官方数值是百分数, 这里除以 100
    三因子和五因子的 SMB 口径不同, 两份文件各自读取

    Args:
        path: 日频因子 zip 的本地路径, 支持 str 或 Path
            三因子文件列为 Mkt-RF, SMB, HML, RF
            五因子文件再加 RMW, CMA

    Returns:
        以 date 为索引的 DataFrame, 按日期升序
        列为 mkt_rf, smb, hml, rf, 五因子另有 rmw, cma
        0.0009 表示 0.09%
        不修改压缩包

    Raises:
        FileNotFoundError: zip 不存在
        ValueError: 压缩包里没有 Mkt-RF 表头

    Example:
        >>> from quanttoolkit.sources.french import read_french_factors
        >>> factors = read_french_factors("data/ff3_daily.zip")
        >>> factors.columns.tolist()
        ['mkt_rf', 'smb', 'hml', 'rf']
        >>> round(float(factors.loc["1926-07-01", "mkt_rf"]), 4)
        0.0009
    """

    with ZipFile(path) as archive:
        text = archive.read(archive.namelist()[0]).decode("utf-8")

    lines = text.splitlines()
    start = next(
        (index for index, line in enumerate(lines) if line.startswith(",Mkt-RF")),
        None,
    )
    if start is None:
        raise ValueError("压缩包里没有因子表")

    rows = [lines[start]]
    for line in lines[start + 1 :]:
        if not line.strip():
            break
        rows.append(line)

    factors = pd.read_csv(StringIO("\n".join(rows)), index_col=0)
    factors.index = pd.to_datetime(factors.index.astype(str), format="%Y%m%d")
    factors.index.name = "date"
    factors = factors.rename(columns=_COLUMN_MAP)
    columns = [name for name in _COLUMN_MAP.values() if name in factors.columns]
    return factors.loc[:, columns].div(100).sort_index()
