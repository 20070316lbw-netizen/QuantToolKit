import doctest

import numpy as np
import pandas as pd

from quanttoolkit.factors import zscore
from quanttoolkit.factors.zscore import zscore_by_date


def _panel(
    values: dict[str, list[float]], dates: list[str], tickers: list[str]
) -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(dates), tickers],
        names=["date", "ticker"],
    )
    return pd.DataFrame(values, index=index)


def test_zscore_docstring_example():
    result = doctest.testmod(zscore)
    assert result.failed == 0
    assert result.attempted > 0


def test_zscore_is_computed_inside_each_date():
    factors = _panel(
        {"f1": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0]},
        ["2024-01-01", "2024-01-02"],
        ["A", "B", "C"],
    )
    before = factors.copy(deep=True)
    result = zscore_by_date(factors)
    assert result.index.equals(factors.index)
    assert result.columns.tolist() == ["f1"]
    np.testing.assert_allclose(result.loc["2024-01-01", "f1"], [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(result.loc["2024-01-02", "f1"], [1.0, 0.0, -1.0])
    pd.testing.assert_frame_equal(factors, before)


def test_zscore_ignores_missing_values_in_the_cross_section():
    factors = _panel(
        {"f1": [1.0, np.nan, 3.0]},
        ["2024-01-01"],
        ["A", "B", "C"],
    )
    result = zscore_by_date(factors)["f1"]
    scale = np.std([1.0, 3.0], ddof=1)
    np.testing.assert_allclose(result.iloc[0], (1.0 - 2.0) / scale)
    assert np.isnan(result.iloc[1])
    np.testing.assert_allclose(result.iloc[2], (3.0 - 2.0) / scale)


def test_zscore_zero_std_and_single_name_are_nan():
    constant = _panel({"f1": [2.0, 2.0]}, ["2024-01-01"], ["A", "B"])
    single = constant.iloc[:1]
    assert zscore_by_date(constant)["f1"].isna().all()
    assert zscore_by_date(single)["f1"].isna().all()


def test_zscore_empty_input_keeps_columns():
    empty = pd.DataFrame(
        columns=["f1"],
        index=pd.MultiIndex.from_arrays([[], []], names=["date", "ticker"]),
    )
    result = zscore_by_date(empty)
    assert result.empty
    assert result.columns.tolist() == ["f1"]
    assert result.index.names == ["date", "ticker"]
