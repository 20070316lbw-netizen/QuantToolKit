"""把名单, 行情和 French 因子准备好, 再建出可以直接查询的因子视图

在自己的项目里调用 main(), 不用再把下载和入库步骤一块一块拼起来
默认从当前工作目录建立 logs 和 data, 不往安装包所在目录写数据
"""

from __future__ import annotations

from pathlib import Path
from urllib.request import Request, urlopen

import duckdb
import pandas as pd
from loguru import logger

from quanttoolkit.db.connection import get_duckdb
from quanttoolkit.db.operation import SQL_DIR, operation_sql, read_sql
from quanttoolkit.panel.prices import fill_price_gaps
from quanttoolkit.sources import get_prices
from quanttoolkit.sources.french import read_french_factors
from quanttoolkit.sources.wikipedia import fetch_sp500_universe
from quanttoolkit.tracelog import configure_log, print_call_chain, trace

__all__ = ["main"]

_FRENCH_URLS = {
    "ff3": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_Factors_daily_CSV.zip",
    "ff5": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
}


def main(
    *,
    logger_path: str | Path | None = None,
    data_path: str | Path | None = None,
) -> Path:
    """准备当前标普500的数据, 返回可以直接打开的 DuckDB 文件路径

    先获取名单, 再下载 FF3/FF5 和日行情, 写入后建立 factor_panel 视图
    行情从 2016-01-01 开始, 结束范围由 Yahoo 决定, 使用复权 OHLC
    补齐每只证券首尾记录之间的内部缺行, 新增行的成交量为 0
    名单是运行时的当前成员, 不是历史时点股票池

    Args:
        logger_path: 日志文件夹, 默认 None, 使用当前工作目录的 logs
            支持 str 或 Path, 自动创建, trace.log 记录函数追踪与调用链
            data_pipeline.log 记录整个流程及数据源的 loguru 日志
            后续 tracelog 也沿用这里的 trace.log, 日志按追加写入
        data_path: 数据文件夹, 默认 None, 使用当前工作目录的 data
            支持 str 或 Path, 自动创建, 数据库名固定为 quanttoolkit.db
            同目录保存 ff3_daily.zip 和 ff5_daily.zip, 每次重新下载

    Returns:
        Path: 数据库绝对路径, 含 universe, prices, ff_factors 和 factor_panel
        再次运行替换名单与 French 因子, 行情按 (ticker, date) 更新或追加
        各表分别事务写入, 后续步骤失败时保留已完成步骤的数据
        部分证券抓取失败时记录警告并使用其余行情, 全部为空时抛出异常
        FF5 日期以官方发布为准, 视图只保留有 FF5 且经过预热期的日期
        各证券历史不足时对应滚动因子为空, 不填充这些因子缺失值

    Raises:
        OSError: 创建目录, 下载文件或写日志失败
        ValueError: 名单, French 因子或行情为空, 或数据源解析失败
        duckdb.Error: 数据库打开, 写入或建立视图失败
        Exception: 其他数据源异常记录后原样抛出, 不继续后续步骤

    Example:
        需要网络, 首次运行会下载数百只股票的历史行情
        >>> from quanttoolkit.data_pipeline import main
        >>> database = main()  # doctest: +SKIP
        >>> database.name  # doctest: +SKIP
        'quanttoolkit.db'
    """
    root = Path.cwd()
    log_dir = (root / "logs" if logger_path is None else Path(logger_path)).expanduser()
    data_dir = (root / "data" if data_path is None else Path(data_path)).expanduser()
    log_dir = log_dir.resolve()
    data_dir = data_dir.resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    configure_log(log_dir / "trace.log")
    sink = logger.add(
        log_dir / "data_pipeline.log",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {message}",
        diagnose=False,
        encoding="utf-8",
    )
    try:
        logger.info("日志文件夹已准备好: {}", log_dir)
        if logger_path is None or data_path is None:
            defaults = []
            if logger_path is None:
                defaults.append("logs/")
            if data_path is None:
                defaults.append("data/")
            logger.info(
                "未指定 {} 的地址, 自动在当前工作目录 {} 下创建",
                ", ".join(defaults),
                root,
            )
        data_dir.mkdir(parents=True, exist_ok=True)
        database = data_dir / "quanttoolkit.db"
        logger.info("数据文件夹已准备好: {}, 数据库: {}", data_dir, database)
        _prepare_data(database, data_dir)
        logger.info("数据准备完成, 数据库: {}", database)
        return database
    except Exception:
        logger.exception("数据准备失败, 已停止后续步骤")
        raise
    finally:
        try:
            print_call_chain()
        finally:
            logger.remove(sink)


@trace
def _write_table(
    connection: duckdb.DuckDBPyConnection, table: str, frame: pd.DataFrame
) -> int:
    written = operation_sql(
        read_sql(SQL_DIR / f"{table}.sql"),
        read_sql(SQL_DIR / f"{table}_ddl.sql"),
        connection,
        frame,
    )
    logger.info("{} 写入 {} 条记录", table, written)
    return written


@trace
def _fetch_french(data_dir: Path) -> pd.DataFrame:
    frames = []
    for model, url in _FRENCH_URLS.items():
        path = data_dir / f"{model}_daily.zip"
        logger.info("下载 {} 日频因子: {}", model, path)
        request = Request(url, headers={"User-Agent": "QuantToolKit/0.1"})
        with urlopen(request, timeout=30) as response:
            path.write_bytes(response.read())
        factors = read_french_factors(path).reset_index()
        if factors.empty:
            raise ValueError(f"{model} 日频因子为空")
        factors.insert(1, "model", model)
        frames.append(factors)
    return pd.concat(frames, ignore_index=True).reindex(
        columns=["date", "model", "mkt_rf", "smb", "hml", "rmw", "cma", "rf"]
    )


@trace
def _prepare_data(database: Path, data_dir: Path) -> None:
    logger.info("[1/4] 获取当前标普500成员名单")
    universe = fetch_sp500_universe()
    if universe.empty:
        raise ValueError("标普500成员名单为空")
    with get_duckdb(path=database, read_only=False) as connection:
        _write_table(connection, "universe", universe)

        logger.info("[2/4] 获取 FF3 和 FF5 日频因子")
        _write_table(connection, "ff_factors", _fetch_french(data_dir))

        logger.info("[3/4] 获取 2016-01-01 起的日行情, 再补齐内部缺行")
        prices = get_prices(
            tickers=universe["ticker"].tolist(),
            start="2016-01-01",
            end=None,
            auto_adjust=True,
            progress=True,
        )
        if prices.empty:
            raise ValueError("日行情为空")
        missing = sorted(
            set(universe["ticker"]) - set(prices.index.get_level_values("ticker"))
        )
        if missing:
            logger.warning(
                "以下 {} 个成员没有本次行情: {}", len(missing), ", ".join(missing)
            )
        filled = fill_price_gaps(prices)
        logger.info(
            "内部缺行补齐 {} 条, 新增行的 volume 为 0", len(filled) - len(prices)
        )
        _write_table(connection, "prices", filled.reset_index())

        logger.info("[4/4] 建立 factor_panel 视图")
        connection.execute(read_sql(SQL_DIR / "factor_panel.sql"))
        for table in ("universe", "prices", "ff_factors"):
            count = connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            logger.info("库内 {} 共 {} 条", table, count)
        logger.info("factor_panel 视图已建立, 查询时按底层数据计算")
