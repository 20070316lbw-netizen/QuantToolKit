import doctest

import numpy as np
import pandas as pd

from quanttoolkit.factors import rsi_base
from quanttoolkit.factors.rsi_base import base_rsifactor


def test_rsi_docstring_example():
    result = doctest.testmod(rsi_base)
    assert result.failed == 0
    assert result.attempted > 0


def test_rsi_wilder_smoothing_and_ticker_isolation(price_panel):
    before = price_panel.copy(deep=True)
    result = base_rsifactor(price_panel, window=2)
    assert isinstance(result, pd.Series)
    assert result.name == "rsi_2"
    assert result.index.equals(price_panel.sort_index().index)
    np.testing.assert_allclose(
        result.xs("A", level="ticker"),
        [np.nan, np.nan, 50.0, 250.0 / 3.0, 850.0 / 9.0],
    )
    np.testing.assert_allclose(
        result.xs("B", level="ticker"),
        [np.nan, np.nan, 100.0],
    )
    pd.testing.assert_frame_equal(price_panel, before)


def test_rsi_default_window_needs_fifteen_prices(price_panel):
    result = base_rsifactor(price_panel)
    assert result.name == "rsi_14"
    assert result.isna().all()


def test_rsi_missing_price_restarts_after_the_gap(price_panel):
    panel = price_panel.xs("A", level="ticker", drop_level=False).sort_index().copy()
    panel.iloc[1, panel.columns.get_loc("adj_close")] = np.nan
    np.testing.assert_allclose(
        base_rsifactor(panel, window=2),
        [np.nan, np.nan, np.nan, np.nan, 100.0],
    )


def test_rsi_flat_prices_are_nan_and_one_sided_moves_hit_bounds():
    index = pd.MultiIndex.from_product(
        [pd.date_range("2024-01-01", periods=4), ["A"]],
        names=["date", "ticker"],
    )
    flat = pd.DataFrame({"adj_close": [1.0, 1.0, 1.0, 1.0]}, index=index)
    rising = pd.DataFrame({"adj_close": [1.0, 2.0, 4.0, 8.0]}, index=index)
    falling = pd.DataFrame({"adj_close": [8.0, 4.0, 2.0, 1.0]}, index=index)
    assert base_rsifactor(flat, window=2).isna().all()
    np.testing.assert_allclose(
        base_rsifactor(rising, window=2),
        [np.nan, np.nan, 100.0, 100.0],
    )
    np.testing.assert_allclose(
        base_rsifactor(falling, window=2),
        [np.nan, np.nan, 0.0, 0.0],
    )


def test_rsi_custom_price_column(price_panel):
    panel = price_panel.copy()
    panel["custom"] = 100.0
    result = base_rsifactor(panel, window=2, price_col="custom")
    assert result.name == "rsi_2"
    assert result.isna().all()


def test_rsi_empty_input_keeps_name():
    empty = pd.DataFrame(
        columns=["adj_close"],
        index=pd.MultiIndex.from_arrays([[], []], names=["date", "ticker"]),
    )
    result = base_rsifactor(empty, window=14)
    assert result.empty
    assert result.name == "rsi_14"
    assert result.index.names == ["date", "ticker"]
