import doctest

import numpy as np
import pandas as pd

from quanttoolkit.factors import logvol_base
from quanttoolkit.factors.logvol_base import base_logvolfactor


def _rolling_logvol(prices: np.ndarray, window: int) -> list[float]:
    returns = np.diff(np.log(prices))
    expected = [np.nan] * window
    for end in range(window, len(returns) + 1):
        expected.append(float(np.std(returns[end - window : end], ddof=1)))
    return expected


def test_logvol_docstring_example():
    result = doctest.testmod(logvol_base)
    assert result.failed == 0
    assert result.attempted > 0


def test_logvol_sample_std_and_ticker_isolation(price_panel):
    before = price_panel.copy(deep=True)
    result = base_logvolfactor(price_panel, window=2)
    assert isinstance(result, pd.Series)
    assert result.name == "logvol_2"
    assert result.index.equals(price_panel.sort_index().index)
    np.testing.assert_allclose(
        result.xs("A", level="ticker"),
        _rolling_logvol(np.array([10.0, 20.0, 10.0, 30.0, 60.0]), 2),
    )
    np.testing.assert_allclose(
        result.xs("B", level="ticker"),
        _rolling_logvol(np.array([100.0, 100.0, 200.0]), 2),
    )
    pd.testing.assert_frame_equal(price_panel, before)


def test_logvol_missing_price_invalidates_window(price_panel):
    panel = price_panel.xs("A", level="ticker", drop_level=False).sort_index().copy()
    panel.iloc[1, panel.columns.get_loc("adj_close")] = np.nan
    result = base_logvolfactor(panel, window=2)
    np.testing.assert_allclose(result.iloc[:4], [np.nan] * 4)
    np.testing.assert_allclose(
        result.iloc[4],
        np.std(np.log([30.0 / 10.0, 60.0 / 30.0]), ddof=1),
    )


def test_logvol_constant_log_returns_are_zero(price_panel):
    panel = price_panel.xs("A", level="ticker", drop_level=False).sort_index().copy()
    panel["adj_close"] = [1.0, 2.0, 4.0, 8.0, 16.0]
    np.testing.assert_allclose(
        base_logvolfactor(panel, window=2), [np.nan, np.nan, 0, 0, 0], atol=1e-15
    )


def test_logvol_custom_price_column(price_panel):
    panel = price_panel.copy()
    panel["custom"] = 100.0
    result = base_logvolfactor(panel, window=2, price_col="custom")
    np.testing.assert_allclose(
        result.xs("A", level="ticker"), [np.nan, np.nan, 0, 0, 0]
    )
    np.testing.assert_allclose(result.xs("B", level="ticker"), [np.nan, np.nan, 0])


def test_logvol_empty_input_keeps_name():
    empty = pd.DataFrame(
        columns=["adj_close"],
        index=pd.MultiIndex.from_arrays([[], []], names=["date", "ticker"]),
    )
    result = base_logvolfactor(empty, window=2)
    assert result.empty
    assert result.name == "logvol_2"
    assert result.index.names == ["date", "ticker"]
