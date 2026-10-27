import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from loguru import logger

from quanttoolkit import tracelog
from quanttoolkit.tracelog import clear_log, configure_log, print_call_chain, trace

_ELAPSED = r"[0-9]+\.[0-9]{2}s"


@pytest.fixture(autouse=True)
def isolated_tracelog(monkeypatch, tmp_path):
    tracelog._reset()
    monkeypatch.setattr(tracelog, "_LOG_PATH", Path("logs/trace.log"))
    monkeypatch.chdir(tmp_path)
    yield
    tracelog._reset()


def _read_log() -> str:
    return (Path("logs") / "trace.log").read_text(encoding="utf-8")


def test_return_summaries_cover_frame_series_array_and_object():
    @trace(max_rows=1)
    def load_prices():
        return pd.DataFrame({"close": ["keep-close", "drop-close"], "volume": [1, 2]})

    @trace(max_rows=2)
    def build_factor():
        return pd.Series(["keep-a", "keep-b", "drop-c"], dtype="object")

    @trace(max_rows=2)
    def as_array():
        return np.array([["n0", "n1"], ["n2", "n3"]], dtype="object")

    @trace
    def plain():
        return "Q" * 400

    load_prices()
    build_factor()
    as_array()
    plain()
    log = _read_log()

    assert "DataFrame shape=(2, 2)" in log
    assert "['close', 'volume']" in log
    assert "keep-close" in log
    assert "drop-close" not in log

    assert "Series shape=(3,)" in log
    assert "dtype=object" in log
    assert "keep-a" in log
    assert "keep-b" in log
    assert "drop-c" not in log

    assert "ndarray shape=(2, 2)" in log
    assert "dtype=object" in log
    assert "head=['n0', 'n1']" in log
    assert "n2" not in log

    assert "Q" * 199 in log
    assert "Q" * 400 not in log

    text = print_call_chain()
    assert "DataFrame(2, 2)" in text
    assert "Series(3,)" in text
    assert "ndarray(2, 2)" in text


def test_exception_is_recorded_and_reraised():
    sentinel = ValueError("价格缺失")

    @trace
    def child():
        raise sentinel

    @trace
    def parent():
        return child()

    with pytest.raises(ValueError) as caught:
        parent()
    assert caught.value is sentinel

    log = _read_log()
    assert child.__qualname__ in log
    assert parent.__qualname__ in log
    assert "ValueError: 价格缺失" in log
    assert "Traceback" in log

    text = print_call_chain()
    assert "parent  (" in text
    assert "└── child" in text
    assert "ValueError: 价格缺失" in text


def test_nested_call_chain(capsys):
    @trace
    def zscore(values):
        return values

    @trace
    def build_factor(prices):
        return zscore(prices["close"])

    @trace
    def load_prices():
        return pd.DataFrame({"close": [1.0, 2.0, 3.0]})

    @trace
    def main():
        prices = load_prices()
        return build_factor(prices)

    assert list(main()) == [1.0, 2.0, 3.0]
    text = print_call_chain()
    assert capsys.readouterr().out == f"{text}\n"
    lines = text.splitlines()
    assert lines[0] == "调用链:"
    assert re.fullmatch(rf"main  \({_ELAPSED}\) -> Series\(3,\)", lines[1])
    assert re.fullmatch(
        rf"├── load_prices  \({_ELAPSED}\) -> DataFrame\(3, 1\)",
        lines[2],
    )
    assert re.fullmatch(
        rf"└── build_factor  \({_ELAPSED}\) -> Series\(3,\)",
        lines[3],
    )
    assert re.fullmatch(
        rf"    └── zscore  \({_ELAPSED}\) -> Series\(3,\)",
        lines[4],
    )
    assert len(lines) == 5


def test_trace_log_is_appended_to_default_file():
    assert not Path("logs/trace.log").exists()

    @trace
    def first():
        return "第一笔"

    @trace
    def second():
        return "第二笔"

    first()
    midpoint = _read_log()
    second()
    written = _read_log()
    assert first.__qualname__ in midpoint
    assert "第一笔" in midpoint
    assert written.startswith(midpoint)
    assert second.__qualname__ in written
    assert "第二笔" in written


def test_clear_log_empties_default_and_custom_files():
    @trace
    def once():
        return "第一次"

    once()
    path = Path("logs") / "trace.log"
    assert "第一次" in path.read_text(encoding="utf-8")

    cleared = clear_log()
    assert cleared == path
    assert cleared.read_text(encoding="utf-8") == ""

    once()
    assert "第一次" in path.read_text(encoding="utf-8")

    custom = Path("nested/other.log")
    custom.parent.mkdir(parents=True)
    custom.write_text("旧内容", encoding="utf-8")
    assert clear_log(custom) == custom
    assert custom.read_text(encoding="utf-8") == ""

    missing = "nested/missing.log"
    created = clear_log(missing)
    assert created == Path(missing)
    assert created.is_file()
    assert created.read_text(encoding="utf-8") == ""


def test_bare_and_parameterized_decorators_keep_metadata():
    @trace
    def bare():
        """裸装饰"""
        return pd.Series(
            ["row-0", "row-1", "row-2", "row-3", "row-4", "row-5"],
            dtype="object",
        )

    @trace(max_rows=1)
    def limited():
        """限制行数"""
        return pd.DataFrame({"name": ["keep-row", "hide-row"]})

    assert bare.__name__ == "bare"
    assert bare.__doc__ == "裸装饰"
    assert limited.__name__ == "limited"
    assert limited.__doc__ == "限制行数"

    bare()
    limited()
    log = _read_log()
    assert bare.__qualname__ in log
    assert limited.__qualname__ in log
    assert "row-4" in log
    assert "row-5" not in log
    assert "keep-row" in log
    assert "hide-row" not in log


def test_other_logger_output_stays_out_of_trace_log():
    @trace
    def load_prices():
        logger.info("来自现有模块的记录")
        return 1

    logger.warning("装饰之前的记录")
    assert load_prices() == 1
    logger.error("装饰之后的记录")

    log = _read_log()
    assert load_prices.__qualname__ in log
    assert "来自现有模块的记录" not in log
    assert "装饰之前的记录" not in log
    assert "装饰之后的记录" not in log


def test_configure_log_routes_existing_sink_and_clear_log(tmp_path):
    @trace
    def step():
        return "数据准备"

    step()
    original = _read_log()
    target = tmp_path / "custom/trace.log"
    assert configure_log(target) == target
    assert not target.exists()
    step()
    text = print_call_chain()
    written = target.read_text()
    assert "数据准备" in written
    assert text in written
    assert _read_log() == original
    assert clear_log() == target
    assert target.read_text() == ""


def test_idle_tracer_does_not_create_log_or_print(capsys):
    assert not Path("logs/trace.log").exists()
    assert print_call_chain() == ""
    tracelog._print_call_chain_at_exit()
    assert capsys.readouterr().out == ""
    assert not Path("logs/trace.log").exists()


def test_exit_hook_prints_once_until_new_call(capsys):
    @trace
    def leaf():
        return "done"

    leaf()
    tracelog._print_call_chain_at_exit()
    first = capsys.readouterr().out
    assert "leaf" in first
    assert "done" in first

    tracelog._print_call_chain_at_exit()
    assert capsys.readouterr().out == ""

    leaf()
    tracelog._print_call_chain_at_exit()
    assert "leaf" in capsys.readouterr().out
