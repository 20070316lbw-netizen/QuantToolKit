import doctest

import numpy as np
import pandas as pd
import pytest

from quanttoolkit.factors import exposure
from quanttoolkit.factors.exposure import FF5, cma_exposure


@pytest.mark.parametrize("window", [8, 13])
@pytest.mark.parametrize("missing", [None, "volume", "price", "factor"])
def test_cma_matches_independent_window_regressions(window, missing):
    rng = np.random.default_rng(31)
    dates = pd.bdate_range("2024-01-01", periods=40)
    ff = pd.DataFrame(rng.normal(0, 0.01, (40, 5)), index=dates, columns=FF5)
    ff["rf"] = 0.0001
    pieces = []
    for ticker, beta in [("A", 0.8), ("B", -1.2)]:
        daily = ff["rf"] + beta * ff["cma"] + 0.3 * ff["hml"]
        index = pd.MultiIndex.from_product([dates, [ticker]], names=["date", "ticker"])
        pieces.append(
            pd.DataFrame(
                {"adj_close": 100 * (1 + daily).cumprod().to_numpy(), "volume": 100.0},
                index=index,
            )
        )
    prices = pd.concat(pieces).sort_index()
    key = (dates[16], "A")
    if missing == "volume":
        prices.loc[key, "volume"] = 0
    elif missing == "price":
        prices.loc[key, "adj_close"] = np.nan
    elif missing == "factor":
        ff.loc[dates[16], "smb"] = np.nan
    original_prices, original_ff = prices.copy(), ff.copy()
    actual = cma_exposure(prices, ff, window)
    expected = pd.Series(np.nan, index=prices.index, name="b_cma")
    for ticker in ["A", "B"]:
        stock = prices.xs(ticker, level="ticker")
        target = (stock["adj_close"].pct_change() - ff["rf"]).where(stock["volume"] > 0)
        for end in range(window - 1, len(dates)):
            span = dates[end - window + 1 : end + 1]
            y = target.loc[span].to_numpy()
            x = np.column_stack([np.ones(window), ff.loc[span, FF5]])
            if np.isfinite(y).all() and np.isfinite(x).all():
                expected.loc[(dates[end], ticker)] = np.linalg.lstsq(x, y, rcond=None)[
                    0
                ][-1]
    pd.testing.assert_series_equal(actual, expected, atol=1e-7, rtol=1e-7)
    pd.testing.assert_frame_equal(prices, original_prices)
    pd.testing.assert_frame_equal(ff, original_ff)


def test_exposure_docstring():
    assert doctest.testmod(exposure).failed == 0
