import doctest

import numpy as np
import pandas as pd
import pytest

from quanttoolkit.portfolio import sizing
from quanttoolkit.portfolio.sizing import simple_quantile_long_short


def _score(values: list[float], dates: list[str], tickers: list[str]) -> pd.Series:
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(dates), tickers],
        names=["date", "ticker"],
    )
    return pd.Series(values, index=index, name="score")


def test_sizing_docstring_example():
    result = doctest.testmod(sizing)
    assert result.failed == 0
    assert result.attempted > 0


def test_long_short_weights_sum_to_opposite_unit_exposure():
    score = _score(
        [5.0, 3.0, 1.0, -1.0, -3.0, -5.0],
        ["2024-01-01"],
        ["A", "B", "C", "D", "E", "F"],
    )
    before = score.copy()
    weights = simple_quantile_long_short(score=score, n_quantiles=3)
    assert weights.name == "weight"
    assert weights.index.names == ["date", "ticker"]
    assert weights[weights > 0].sum() == 1
    assert weights[weights < 0].sum() == -1
    assert (weights == 0).sum() == 2
    pd.testing.assert_series_equal(score, before)


def test_dates_are_sized_separately_and_thin_dates_drop_out():
    thick = _score([5.0, 3.0, 1.0, -1.0], ["2024-01-02"], ["A", "B", "C", "D"])
    thin = _score([9.0, 8.0], ["2024-01-01"], ["A", "B"])
    weights = simple_quantile_long_short(score=pd.concat([thin, thick]), n_quantiles=4)
    dates = weights.index.get_level_values("date").unique()
    assert dates.strftime("%Y-%m-%d").tolist() == ["2024-01-02"]
    assert weights.tolist() == [1.0, 0.0, 0.0, -1.0]


def test_missing_score_is_removed():
    score = _score(
        [5.0, np.nan, 1.0, -1.0, -3.0, -5.0],
        ["2024-01-01"],
        ["A", "B", "C", "D", "E", "F"],
    )
    weights = simple_quantile_long_short(score=score, n_quantiles=5)
    assert "B" not in weights.index.get_level_values("ticker")
    assert weights[weights > 0].sum() == 1
    assert weights[weights < 0].sum() == -1


def test_collapsed_bins_and_empty_input_have_no_weights():
    tied = _score([1.0, 1.0, 1.0, 1.0], ["2024-01-01"], ["A", "B", "C", "D"])
    assert simple_quantile_long_short(score=tied, n_quantiles=2).empty
    empty = tied.iloc[:0]
    result = simple_quantile_long_short(score=empty, n_quantiles=2)
    assert result.empty
    assert result.name == "weight"
    assert result.index.names == ["date", "ticker"]


def test_quantile_count_below_two_raises():
    score = _score([1.0, 2.0], ["2024-01-01"], ["A", "B"])
    with pytest.raises(ValueError, match="n_quantiles"):
        simple_quantile_long_short(score=score, n_quantiles=1)
