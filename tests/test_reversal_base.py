import doctest

import numpy as np
import pandas as pd
import pytest

from quanttoolkit.factors import reversal_base
from quanttoolkit.factors.mom_base import base_momfactor
from quanttoolkit.factors.reversal_base import base_reversalfactor


@pytest.mark.parametrize(
    "skip, expected_a, expected_b",
    [
        (
            0,
            [np.nan, np.nan, 0, -np.log(1.5), -np.log(6)],
            [np.nan, np.nan, -np.log(2)],
        ),
        (1, [np.nan, np.nan, np.nan, 0, -np.log(1.5)], [np.nan] * 3),
        (2, [np.nan, np.nan, np.nan, np.nan, 0], [np.nan] * 3),
    ],
)
def test_reversal_window_skip_and_ticker_isolation(
    price_panel, skip, expected_a, expected_b
):
    before = price_panel.copy(deep=True)
    result = base_reversalfactor(price_panel, window=2, skip=skip)
    assert isinstance(result, pd.Series)
    assert result.name == (f"reversal_2_{skip}" if skip else "reversal_2")
    assert result.index.equals(price_panel.sort_index().index)
    np.testing.assert_allclose(result.xs("A", level="ticker"), expected_a, atol=1e-15)
    np.testing.assert_allclose(result.xs("B", level="ticker"), expected_b)
    pd.testing.assert_frame_equal(price_panel, before)


def test_rising_and_falling_prices_have_opposite_signs(price_panel):
    result = base_reversalfactor(price_panel, window=1)
    np.testing.assert_allclose(
        result.xs("A", level="ticker"),
        [np.nan, -np.log(2), np.log(2), -np.log(3), -np.log(2)],
    )


def test_missing_price_invalidates_window_then_recovers(price_panel):
    panel = price_panel.xs("A", level="ticker", drop_level=False).sort_index().copy()
    panel.iloc[1, panel.columns.get_loc("adj_close")] = np.nan
    result = base_reversalfactor(panel, window=2)
    np.testing.assert_allclose(result, [np.nan] * 4 + [-np.log(6)])


def test_skip_excludes_recent_price_changes(price_panel):
    before = base_reversalfactor(price_panel, window=2, skip=1)
    changed = price_panel.copy()
    changed.loc[(pd.Timestamp("2024-01-09"), "A"), "adj_close"] = 6000.0
    pd.testing.assert_series_equal(
        base_reversalfactor(changed, window=2, skip=1), before
    )


def test_custom_price_column(price_panel):
    custom = price_panel.rename(columns={"adj_close": "custom"})
    pd.testing.assert_series_equal(
        base_reversalfactor(custom, window=2, price_col="custom"),
        base_reversalfactor(price_panel, window=2),
    )


@pytest.mark.parametrize("calculation", [base_momfactor, base_reversalfactor])
@pytest.mark.parametrize("skip", [0, 1])
def test_future_records_do_not_change_historical_factor(price_panel, calculation, skip):
    dates = price_panel.index.get_level_values("date")
    history = price_panel.loc[dates < pd.Timestamp("2024-01-09")]
    before = calculation(history, window=2, skip=skip)
    after = calculation(price_panel, window=2, skip=skip)
    pd.testing.assert_series_equal(after.loc[before.index], before)


@pytest.mark.parametrize("calculation", [base_momfactor, base_reversalfactor])
def test_constant_price_has_zero_factor(price_panel, calculation):
    panel = price_panel.copy()
    panel["adj_close"] = 100.0
    result = calculation(panel, window=2, skip=1)
    np.testing.assert_allclose(
        result.xs("A", level="ticker"), [np.nan, np.nan, np.nan, 0, 0]
    )
    assert result.xs("B", level="ticker").isna().all()


def test_reversal_requires_full_history(price_panel):
    assert base_reversalfactor(price_panel, window=5).isna().all()


def test_reversal_docstring_example():
    result = doctest.testmod(reversal_base)
    assert result.failed == 0
    assert result.attempted > 0
