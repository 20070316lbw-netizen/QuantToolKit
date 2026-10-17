import numpy as np
import pandas as pd
import pytest

from quanttoolkit.panel.returns import future_returns, historical_return, log_returns


def test_historical_return_uses_each_tickers_records(price_panel):
    before = price_panel.copy(deep=True)
    result = historical_return(df=price_panel, n_periods=2)
    assert result.name == "historical_return_2"
    assert result.index.equals(price_panel.sort_index().index)
    np.testing.assert_allclose(
        result.xs("A", level="ticker"), [np.nan, np.nan, 0, 0.5, 5]
    )
    np.testing.assert_allclose(result.xs("B", level="ticker"), [np.nan, np.nan, 1])
    pd.testing.assert_frame_equal(price_panel, before)


@pytest.mark.parametrize(
    "gap, expected_a, expected_b",
    [
        (0, [0, 0.5, 5, np.nan, np.nan], [1, np.nan, np.nan]),
        (1, [0.5, 5, np.nan, np.nan, np.nan], [np.nan] * 3),
        (2, [5, np.nan, np.nan, np.nan, np.nan], [np.nan] * 3),
    ],
)
def test_future_return_entry_offset(price_panel, gap, expected_a, expected_b):
    before = price_panel.copy(deep=True)
    result = future_returns(df=price_panel, n_periods=2, gap=gap)
    assert result.name == f"forward_return_2_gap_{gap}"
    assert result.index.equals(price_panel.sort_index().index)
    np.testing.assert_allclose(result.xs("A", level="ticker"), expected_a)
    np.testing.assert_allclose(result.xs("B", level="ticker"), expected_b)
    pd.testing.assert_frame_equal(price_panel, before)


def test_log_returns_isolated_and_sorted(price_panel):
    before = price_panel.copy(deep=True)
    result = log_returns(df=price_panel)
    assert result.name == "log_return"
    assert result.index.equals(price_panel.sort_index().index)
    np.testing.assert_allclose(
        result.xs("A", level="ticker"),
        [np.nan, np.log(2), np.log(0.5), np.log(3), np.log(2)],
    )
    np.testing.assert_allclose(result.xs("B", level="ticker"), [np.nan, 0, np.log(2)])
    pd.testing.assert_frame_equal(price_panel, before)


def test_missing_prices_are_not_filled_and_period_returns_only_need_endpoints(
    price_panel,
):
    panel = price_panel.xs("A", level="ticker", drop_level=False).sort_index()
    panel.iloc[1, panel.columns.get_loc("adj_close")] = np.nan
    np.testing.assert_allclose(
        log_returns(df=panel), [np.nan, np.nan, np.nan, np.log(3), np.log(2)]
    )
    np.testing.assert_allclose(
        historical_return(df=panel, n_periods=2), [np.nan, np.nan, 0, np.nan, 5]
    )
    np.testing.assert_allclose(
        future_returns(df=panel, n_periods=2, gap=0), [0, np.nan, 5, np.nan, np.nan]
    )


def test_return_defaults_and_custom_price_column(price_panel):
    assert historical_return(df=price_panel).isna().all()
    assert future_returns(df=price_panel).isna().all()
    custom = price_panel.rename(columns={"adj_close": "custom"})
    pd.testing.assert_series_equal(
        log_returns(df=custom, price_col="custom"), log_returns(df=price_panel)
    )
    pd.testing.assert_series_equal(
        future_returns(df=price_panel, n_periods=np.int64(1), gap=np.int64(0)),
        future_returns(df=price_panel, n_periods=1, gap=0),
    )
