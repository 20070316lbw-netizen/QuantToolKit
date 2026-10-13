import doctest

import numpy as np
import pandas as pd
import pytest

from quanttoolkit.factors import mom_base
from quanttoolkit.factors.mom_base import base_momfactor


@pytest.mark.parametrize(
    "skip, expected_a, expected_b",
    [
        (0, [np.nan, np.nan, 0, np.log(1.5), np.log(6)], [np.nan, np.nan, np.log(2)]),
        (1, [np.nan, np.nan, np.nan, 0, np.log(1.5)], [np.nan] * 3),
        (2, [np.nan, np.nan, np.nan, np.nan, 0], [np.nan] * 3),
    ],
)
def test_momentum_window_skip_and_ticker_isolation(
    price_panel, skip, expected_a, expected_b
):
    before = price_panel.copy(deep=True)
    result = base_momfactor(price_panel, window=2, skip=skip)
    assert isinstance(result, pd.Series)
    assert result.name == (f"mom_2_{skip}" if skip else "mom_2")
    assert result.index.equals(price_panel.sort_index().index)
    np.testing.assert_allclose(result.xs("A", level="ticker"), expected_a, atol=1e-15)
    np.testing.assert_allclose(result.xs("B", level="ticker"), expected_b)
    pd.testing.assert_frame_equal(price_panel, before)


@pytest.mark.parametrize(
    "window, expected",
    [
        (1, [np.nan, np.log(2), -np.log(2), np.log(3), np.log(2)]),
        (3, [np.nan, np.nan, np.nan, np.log(3), np.log(3)]),
        (5, [np.nan] * 5),
    ],
)
def test_momentum_holding_interval_and_insufficient_history(
    price_panel, window, expected
):
    result = base_momfactor(price_panel, window=window)
    np.testing.assert_allclose(result.xs("A", level="ticker"), expected)


def test_momentum_missing_price_invalidates_whole_window(price_panel):
    panel = price_panel.xs("A", level="ticker", drop_level=False).sort_index().copy()
    panel.iloc[1, panel.columns.get_loc("adj_close")] = np.nan
    np.testing.assert_allclose(
        base_momfactor(panel, window=2), [np.nan] * 4 + [np.log(6)]
    )


def test_momentum_custom_price_column(price_panel):
    panel = price_panel.copy()
    panel["custom"] = 100.0
    result = base_momfactor(panel, window=2, price_col="custom")
    np.testing.assert_allclose(
        result.xs("A", level="ticker"), [np.nan, np.nan, 0, 0, 0]
    )
    np.testing.assert_allclose(result.xs("B", level="ticker"), [np.nan, np.nan, 0])


def test_momentum_skip_excludes_recent_price_changes(price_panel):
    before = base_momfactor(price_panel, window=2, skip=1)
    changed = price_panel.copy()
    changed.loc[(pd.Timestamp("2024-01-09"), "A"), "adj_close"] = 6000.0
    pd.testing.assert_series_equal(base_momfactor(changed, window=2, skip=1), before)


def test_momentum_docstring_example():
    result = doctest.testmod(mom_base)
    assert result.failed == 0
    assert result.attempted > 0
