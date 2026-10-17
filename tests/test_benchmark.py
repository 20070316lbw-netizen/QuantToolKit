import doctest

import numpy as np
import pandas as pd

from quanttoolkit import benchmark
from quanttoolkit.benchmark import equal_weight_hold


def _prices(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["date", "ticker", "adj_close"])
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.set_index(["date", "ticker"])


def test_benchmark_docstring_example():
    result = doctest.testmod(benchmark)
    assert result.failed == 0
    assert result.attempted > 0


def test_equal_weight_hold_keeps_the_initial_shares():
    prices = _prices(
        [
            ("2024-01-03", "B", 1.0),
            ("2024-01-02", "A", 1.0),
            ("2024-01-02", "B", 1.0),
            ("2024-01-03", "A", 2.0),
            ("2024-01-04", "A", 2.0),
            ("2024-01-04", "B", 4.0),
            ("2024-01-03", "C", 100.0),
        ]
    )
    before = prices.copy(deep=True)
    held = equal_weight_hold(prices=prices)
    assert held.name == "equal_weight_hold"
    assert held.index.equals(pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]))
    assert np.isnan(held.iloc[0])
    assert held.iloc[1] == 0.5
    assert held.iloc[2] == 1.0
    pd.testing.assert_frame_equal(prices, before)


def test_equal_weight_hold_missing_price_breaks_the_next_return():
    prices = _prices(
        [
            ("2024-01-02", "A", 1.0),
            ("2024-01-02", "B", 1.0),
            ("2024-01-03", "A", 2.0),
            ("2024-01-03", "B", np.nan),
            ("2024-01-04", "A", 2.0),
            ("2024-01-04", "B", 2.0),
        ]
    )
    held = equal_weight_hold(prices=prices)
    assert held.iloc[1:].isna().all()


def test_equal_weight_hold_empty_input():
    empty = pd.DataFrame(
        columns=["adj_close"],
        index=pd.MultiIndex.from_arrays([[], []], names=["date", "ticker"]),
    )
    held = equal_weight_hold(prices=empty)
    assert held.empty
    assert held.name == "equal_weight_hold"
