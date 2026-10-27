from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import duckdb
import numpy as np
import pandas as pd
import pytest
from loguru import logger

from quanttoolkit import data_pipeline, tracelog


@pytest.fixture(autouse=True)
def isolated_trace(monkeypatch, tmp_path):
    tracelog._reset()
    monkeypatch.setattr(tracelog, "_LOG_PATH", Path("logs/trace.log"))
    monkeypatch.chdir(tmp_path)
    yield
    tracelog._reset()


@pytest.fixture
def sources(monkeypatch):
    dates = pd.bdate_range("2024-01-02", periods=70)
    universe = pd.DataFrame(
        {"ticker": ["A", "B"], "name": ["Alpha", "Beta"], "sub_industry": ["X", "Y"]}
    )
    prices = pd.DataFrame(
        {"date": np.repeat(dates, 2), "ticker": ["A", "B"] * len(dates)}
    )
    prices["close"] = np.exp(np.repeat(np.arange(70), 2) * 0.01) * 100
    for column in ("open", "high", "low", "adj_close"):
        prices[column] = prices["close"]
    prices["volume"] = 1000.0
    prices = prices.set_index(["date", "ticker"]).drop((dates[10], "B"))
    calls = []

    def fetch_prices(**kwargs):
        calls.append(kwargs)
        logger.info("模拟数据源的下载进度")
        return prices.copy()

    def download(request, *, timeout):
        assert timeout == 30
        five = "5_Factors" in request.full_url
        columns = ",Mkt-RF,SMB,HML,RMW,CMA,RF" if five else ",Mkt-RF,SMB,HML,RF"
        values = ",1,2,3,4,5,0.1" if five else ",1,2,3,0.1"
        content = "header\n" + columns + "\n"
        content += "\n".join(day.strftime("%Y%m%d") + values for day in dates)
        archive = BytesIO()
        with ZipFile(archive, "w") as zipped:
            zipped.writestr("factors.csv", content + "\n\ncopyright\n")
        return BytesIO(archive.getvalue())

    monkeypatch.setattr(data_pipeline, "fetch_sp500_universe", lambda: universe.copy())
    monkeypatch.setattr(data_pipeline, "get_prices", fetch_prices)
    monkeypatch.setattr(data_pipeline, "urlopen", download)
    return dates, universe, prices, calls


@pytest.mark.parametrize(
    ("log_dir", "data_dir"),
    [
        (None, None),
        ("custom/logs", "custom/data"),
        ("custom/logs", None),
        (None, "custom/data"),
    ],
)
def test_main_prepares_database_and_logs(tmp_path, sources, log_dir, data_dir):
    dates, _, _, calls = sources
    database = data_pipeline.main(logger_path=log_dir, data_path=data_dir)
    expected_data = tmp_path / (data_dir or "data")
    expected_logs = tmp_path / (log_dir or "logs")
    assert database == expected_data / "quanttoolkit.db"
    assert (expected_data / "ff3_daily.zip").is_file()
    assert (expected_data / "ff5_daily.zip").is_file()
    assert calls == [
        {
            "tickers": ["A", "B"],
            "start": "2016-01-01",
            "end": None,
            "auto_adjust": True,
            "progress": True,
        }
    ]
    with duckdb.connect(str(database), read_only=True) as connection:
        assert connection.execute("SELECT COUNT(*) FROM universe").fetchone()[0] == 2
        assert connection.execute("SELECT COUNT(*) FROM prices").fetchone()[0] == 140
        assert (
            connection.execute("SELECT COUNT(*) FROM ff_factors").fetchone()[0] == 140
        )
        assert connection.execute(
            "SELECT volume FROM prices WHERE ticker = 'B' AND date = ?", [dates[10]]
        ).fetchone() == (0.0,)
        row = connection.execute(
            "SELECT MIN(date), COUNT(*), MIN(mom_63), MIN(mkt_rf) FROM factor_panel"
        ).fetchone()
        assert row[0] == dates[63].date()
        assert row[1] == 14
        assert row[2] == pytest.approx(0.63)
        assert row[3] == pytest.approx(0.01)
        assert connection.execute(
            "SELECT rmw FROM ff_factors WHERE model = 'ff3' LIMIT 1"
        ).fetchone() == (None,)

    log = (expected_logs / "data_pipeline.log").read_text()
    assert "日志文件夹已准备好" in log
    assert "数据文件夹已准备好" in log
    assert "模拟数据源的下载进度" in log
    assert "数据准备完成" in log
    assert ("自动在当前工作目录" in log) == (log_dir is None or data_dir is None)
    if log_dir is not None and data_dir is not None:
        assert not (tmp_path / "logs").exists()
        assert not (tmp_path / "data").exists()
    trace = (expected_logs / "trace.log").read_text()
    assert "_prepare_data" in trace
    assert "_fetch_french" in trace
    assert "调用链:" in trace


def test_repeat_run_updates_data_and_closes_pipeline_sink(
    tmp_path, sources, monkeypatch
):
    database = data_pipeline.main()
    log_path = tmp_path / "logs/data_pipeline.log"
    first = log_path.read_text()
    logger.info("运行结束后不再写入流水线文件")
    assert log_path.read_text() == first

    _, universe, prices, _ = sources
    universe.loc[0, "name"] = "Changed"
    prices.loc[:, "close"] = prices["close"] * 2
    monkeypatch.setattr(data_pipeline, "fetch_sp500_universe", lambda: universe)
    monkeypatch.setattr(data_pipeline, "get_prices", lambda **kwargs: prices)
    assert data_pipeline.main() == database
    with duckdb.connect(str(database), read_only=True) as connection:
        assert connection.execute("SELECT COUNT(*) FROM prices").fetchone()[0] == 140
        assert (
            connection.execute("SELECT COUNT(*) FROM ff_factors").fetchone()[0] == 140
        )
        assert connection.execute(
            "SELECT name FROM universe WHERE ticker = 'A'"
        ).fetchone() == ("Changed",)
        assert connection.execute("SELECT MIN(close) FROM prices").fetchone()[0] == 200
        assert (
            connection.execute("SELECT COUNT(*) FROM factor_panel").fetchone()[0] == 14
        )
    assert log_path.read_text().startswith(first)


def test_network_failure_is_logged_and_stops_before_prices(
    tmp_path, sources, monkeypatch
):
    database = data_pipeline.main()
    sentinel = TimeoutError("French 下载超时")

    def failed(*args, **kwargs):
        raise sentinel

    monkeypatch.setattr(data_pipeline, "urlopen", failed)
    sources[3].clear()
    with pytest.raises(TimeoutError) as caught:
        data_pipeline.main()
    assert caught.value is sentinel
    assert sources[3] == []
    with duckdb.connect(str(database), read_only=True) as connection:
        assert (
            connection.execute("SELECT COUNT(*) FROM ff_factors").fetchone()[0] == 140
        )
        assert (
            connection.execute("SELECT COUNT(*) FROM factor_panel").fetchone()[0] == 14
        )
    log_path = tmp_path / "logs/data_pipeline.log"
    log = log_path.read_text()
    assert "数据准备失败" in log
    assert "French 下载超时" in log
    logger.info("失败后也卸下流水线文件 sink")
    assert log_path.read_text() == log


def test_empty_prices_does_not_report_success(tmp_path, sources, monkeypatch):
    monkeypatch.setattr(
        data_pipeline, "get_prices", lambda **kwargs: sources[2].iloc[:0]
    )
    with pytest.raises(ValueError, match="日行情为空"):
        data_pipeline.main()
    log = (tmp_path / "logs/data_pipeline.log").read_text()
    assert "数据准备完成" not in log
    assert "[4/4]" not in log


def test_partial_prices_warns_about_missing_member(tmp_path, sources, monkeypatch):
    prices = sources[2].loc[lambda frame: frame.index.get_level_values("ticker") == "A"]
    monkeypatch.setattr(data_pipeline, "get_prices", lambda **kwargs: prices)
    database = data_pipeline.main()
    log = (tmp_path / "logs/data_pipeline.log").read_text()
    assert "以下 1 个成员没有本次行情: B" in log
    with duckdb.connect(str(database), read_only=True) as connection:
        assert connection.execute("SELECT COUNT(*) FROM prices").fetchone()[0] == 70


def test_empty_universe_stops_before_download(sources, monkeypatch):
    monkeypatch.setattr(
        data_pipeline, "fetch_sp500_universe", lambda: sources[1].iloc[:0]
    )
    with pytest.raises(ValueError, match="成员名单为空"):
        data_pipeline.main()
    assert sources[3] == []
