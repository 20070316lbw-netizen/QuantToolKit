import doctest

import numpy as np
import pandas as pd

from quanttoolkit import prices as prices_module
from quanttoolkit.prices import fill_price_gaps
from quanttoolkit.returns import log_returns


def _panel() -> pd.DataFrame:
    # A 覆盖全部交易日, B 晚上市并在中间停牌, C 早退市
    # 2024-01-06 是周六, 数据里没有这一天
    rows = [
        ("2024-01-02", "A", 10.0, 9.0, 100.0),
        ("2024-01-03", "A", 11.0, 10.0, 110.0),
        ("2024-01-05", "A", 12.0, 11.0, 120.0),
        ("2024-01-08", "A", 13.0, 12.0, 130.0),
        ("2024-01-03", "B", 20.0, 19.0, 200.0),
        ("2024-01-08", "B", 22.0, 21.0, 220.0),
        ("2024-01-02", "C", 30.0, 29.0, 300.0),
        ("2024-01-03", "C", 31.0, 30.0, 310.0),
    ]
    frame = pd.DataFrame(
        rows, columns=["date", "ticker", "close", "adj_close", "volume"]
    )
    frame["date"] = pd.to_datetime(frame["date"])
    frame["open"] = frame["close"] - 1
    frame["high"] = frame["close"] + 1
    frame["low"] = frame["close"] - 2
    return frame.set_index(["date", "ticker"]).sample(frac=1, random_state=0)


def test_fill_price_gaps_docstring_example():
    result = doctest.testmod(prices_module)
    assert result.failed == 0
    assert result.attempted > 0


def test_fills_interior_gaps_and_keeps_existing_rows():
    raw = _panel()
    raw.loc[(pd.Timestamp("2024-01-03"), "A"), "adj_close"] = np.nan
    before = raw.copy(deep=True)

    filled = fill_price_gaps(raw)

    pd.testing.assert_frame_equal(raw, before)
    assert filled.index.names == ["date", "ticker"]
    assert filled.index.is_monotonic_increasing
    assert "2024-01-06" not in filled.index.get_level_values("date").strftime(
        "%Y-%m-%d"
    )

    b = filled.xs("B", level="ticker")
    assert b.index.strftime("%Y-%m-%d").tolist() == [
        "2024-01-03",
        "2024-01-05",
        "2024-01-08",
    ]
    assert b["close"].tolist() == [20.0, 20.0, 22.0]
    assert b["adj_close"].tolist() == [19.0, 19.0, 21.0]
    assert b["open"].tolist() == [19.0, 19.0, 21.0]
    assert b["high"].tolist() == [21.0, 21.0, 23.0]
    assert b["low"].tolist() == [18.0, 18.0, 20.0]
    assert b["volume"].tolist() == [200.0, 0.0, 220.0]
    assert (
        log_returns(df=filled, price_col="close").xs("B", level="ticker").iloc[1] == 0.0
    )

    assert filled.xs("C", level="ticker").index.strftime("%Y-%m-%d").tolist() == [
        "2024-01-02",
        "2024-01-03",
    ]
    assert np.isnan(filled.loc[(pd.Timestamp("2024-01-03"), "A"), "adj_close"])
    pd.testing.assert_series_equal(
        filled.xs("A", level="ticker")["close"],
        before.xs("A", level="ticker").sort_index()["close"],
    )


def test_empty_and_complete_panels_keep_their_rows():
    empty = pd.DataFrame(
        columns=["close", "volume"],
        index=pd.MultiIndex.from_arrays([[], []], names=["date", "ticker"]),
    )
    filled_empty = fill_price_gaps(empty)
    assert filled_empty.empty
    assert filled_empty.index.names == ["date", "ticker"]
    assert filled_empty.columns.tolist() == ["close", "volume"]

    complete = _panel().xs("A", level="ticker", drop_level=False)
    filled = fill_price_gaps(complete)
    pd.testing.assert_frame_equal(filled, complete.sort_index())
