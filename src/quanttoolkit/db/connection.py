"""DuckDB 连接入口, 默认只读, 连接的关闭由调用方负责

读取接口使用本函数; 写入层显式开启事务
导入模块不会连接数据库
"""

from __future__ import annotations

from pathlib import Path

import duckdb
from loguru import logger

ROOT = Path(__file__).resolve().parent.parent.parent.parent
DB = ROOT / "data" / "quanttoolkit.db"


def get_duckdb(*, path: Path, read_only: bool = True) -> duckdb.DuckDBPyConnection:
    """
    连接 DuckDB 数据库

    Args:
        path: 数据库文件路径 (Path 或可转换为 Path 的字符串)
        read_only: 是否以只读模式打开 (默认True防止回测时意外修改数据)

    Returns:
        duckdb.DuckDBPyConnection: 数据库连接对象, 可用 with 管理生命周期

    Raises:
        FileNotFoundError: 只读模式下数据库不存在
        duckdb.Error: 打开失败、锁冲突或文件格式错误

    Notes:
        read_only=False 自动创建父目录和新数据库文件; 数据表由写入方法初始化
        只读连接不创建目录或文件
    """
    path = Path(path)
    if read_only and not path.exists():
        logger.error(f"数据库文件不存在: {path}")
        raise FileNotFoundError(f"Database file not found at {path}")
    if not read_only:
        path.parent.mkdir(parents=True, exist_ok=True)

    logger.debug(f"正在连接 DuckDB: {path} (read_only = {read_only})")
    return duckdb.connect(str(path), read_only=read_only)
