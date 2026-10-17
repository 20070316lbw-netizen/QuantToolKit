import doctest

import numpy as np
import pandas as pd

from quanttoolkit.factors import neutral
from quanttoolkit.factors.neutral import neutralize_by_industry


def _factors(values: dict[str, list[float]], dates: list[str], tickers: list[str]):
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(dates), tickers],
        names=["date", "ticker"],
    )
    return pd.DataFrame(values, index=index)


def test_neutral_docstring_example():
    result = doctest.testmod(neutral)
    assert result.failed == 0
    assert result.attempted > 0


def test_neutral_subtracts_the_industry_mean_inside_each_date():
    factors = _factors(
        {"f1": [1.0, 3.0, 10.0, 3.0, 1.0, 8.0]},
        ["2024-01-02", "2024-01-03"],
        ["A", "B", "C"],
    )
    industry = pd.Series(["Tech", "Tech", "Bank"], index=["A", "B", "C"])
    before = factors.copy(deep=True)
    result = neutralize_by_industry(factors, industry)
    assert result.index.equals(factors.index)
    assert result.columns.tolist() == ["f1"]
    np.testing.assert_allclose(result.loc["2024-01-02", "f1"], [-1.0, 1.0, 0.0])
    np.testing.assert_allclose(result.loc["2024-01-03", "f1"], [1.0, -1.0, 0.0])
    pd.testing.assert_frame_equal(factors, before)


def test_neutral_keeps_columns_independent_and_skips_missing_values():
    factors = _factors(
        {"f1": [1.0, np.nan, 5.0], "f2": [2.0, 4.0, 9.0]},
        ["2024-01-02"],
        ["A", "B", "C"],
    )
    industry = pd.Series(["Tech", "Tech", "Bank"], index=["A", "B", "C"])
    result = neutralize_by_industry(factors, industry)
    assert np.isnan(result.loc[("2024-01-02", "B"), "f1"])
    assert result.loc[("2024-01-02", "A"), "f1"] == 0.0
    np.testing.assert_allclose(
        result["f2"].tolist(),
        [-1.0, 1.0, 0.0],
    )


def test_neutral_missing_industry_is_nan():
    factors = _factors({"f1": [1.0, 3.0, 8.0]}, ["2024-01-02"], ["A", "B", "C"])
    industry = pd.Series(["Tech", "Tech"], index=["A", "B"])
    result = neutralize_by_industry(factors, industry)["f1"]
    np.testing.assert_allclose(result.iloc[:2], [-1.0, 1.0])
    assert np.isnan(result.iloc[2])


def test_neutral_empty_input_keeps_columns():
    empty = pd.DataFrame(
        columns=["f1"],
        index=pd.MultiIndex.from_arrays([[], []], names=["date", "ticker"]),
    )
    result = neutralize_by_industry(empty, pd.Series(dtype=object))
    assert result.empty
    assert result.columns.tolist() == ["f1"]
    assert result.index.names == ["date", "ticker"]
