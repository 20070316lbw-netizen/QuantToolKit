import numpy as np
import pandas as pd
import pytest

from quanttoolkit.volatility import forward_volatility, history_vol


def test_history_vol_sample_std_not_annualized(price_panel):
    before = price_panel.copy(deep=True)
    result = history_vol(df=price_panel, window=2)
    assert result.name == "volatility_2"
    assert result.index.equals(price_panel.sort_index().index)
    np.testing.assert_allclose(
        result.xs("A", level="ticker"),
        [np.nan, np.nan, 1.5 / np.sqrt(2), 2.5 / np.sqrt(2), 1 / np.sqrt(2)],
    )
    np.testing.assert_allclose(
        result.xs("B", level="ticker"), [np.nan, np.nan, 1 / np.sqrt(2)]
    )
    pd.testing.assert_frame_equal(price_panel, before)


@pytest.mark.parametrize(
    "gap, expected_a, expected_b",
    [
        (
            0,
            [1.5 / np.sqrt(2), 2.5 / np.sqrt(2), 1 / np.sqrt(2), np.nan, np.nan],
            [1 / np.sqrt(2), np.nan, np.nan],
        ),
        (1, [2.5 / np.sqrt(2), 1 / np.sqrt(2), np.nan, np.nan, np.nan], [np.nan] * 3),
        (2, [1 / np.sqrt(2), np.nan, np.nan, np.nan, np.nan], [np.nan] * 3),
    ],
)
def test_forward_volatility_uses_returns_after_entry(
    price_panel, gap, expected_a, expected_b
):
    before = price_panel.copy(deep=True)
    result = forward_volatility(df=price_panel, window=2, gap=gap)
    assert result.name == f"forward_volatility_2_gap_{gap}"
    assert result.index.equals(price_panel.sort_index().index)
    np.testing.assert_allclose(result.xs("A", level="ticker"), expected_a)
    np.testing.assert_allclose(result.xs("B", level="ticker"), expected_b)
    pd.testing.assert_frame_equal(price_panel, before)


def test_missing_prices_invalidate_whole_window_and_then_recover(price_panel):
    panel = price_panel.xs("A", level="ticker", drop_level=False).sort_index()
    panel.iloc[1, panel.columns.get_loc("close")] = np.nan
    np.testing.assert_allclose(
        history_vol(df=panel, window=2), [np.nan] * 4 + [1 / np.sqrt(2)]
    )
    np.testing.assert_allclose(
        forward_volatility(df=panel, window=2, gap=0),
        [np.nan, np.nan, 1 / np.sqrt(2), np.nan, np.nan],
    )
    # gap=1 的标签区间从第二条价格开始, 基准日自身缺失不会污染该窗口。
    panel.iloc[0, panel.columns.get_loc("close")] = np.nan
    panel.iloc[1, panel.columns.get_loc("close")] = 40.0
    assert forward_volatility(df=panel, window=2, gap=1).iloc[0] == pytest.approx(
        2.5 / np.sqrt(2)
    )


def test_constant_returns_have_zero_volatility(price_panel):
    panel = price_panel.xs("A", level="ticker", drop_level=False).sort_index()
    panel["close"] = [1, 2, 4, 8, 16]
    np.testing.assert_allclose(
        history_vol(df=panel, window=2), [np.nan, np.nan, 0, 0, 0]
    )
    np.testing.assert_allclose(
        forward_volatility(df=panel, window=2, gap=0), [0, 0, 0, np.nan, np.nan]
    )


def test_volatility_defaults_and_custom_price_column(price_panel):
    assert history_vol(df=price_panel).isna().all()
    assert forward_volatility(df=price_panel).isna().all()
    custom = price_panel.rename(columns={"close": "custom"})
    pd.testing.assert_series_equal(
        history_vol(df=custom, window=np.int64(2), price_col="custom"),
        history_vol(df=price_panel, window=2),
    )
    pd.testing.assert_series_equal(
        forward_volatility(df=price_panel, window=np.int64(2), gap=np.int64(0)),
        forward_volatility(df=price_panel, window=2, gap=0),
    )
