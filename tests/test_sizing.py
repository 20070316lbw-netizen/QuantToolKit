import doctest

import numpy as np
import pandas as pd
import pytest

from quanttoolkit.portfolio import sizing
from quanttoolkit.portfolio.sizing import (
    simple_quantile_long_short,
    vol_quantile_long_short,
)


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


def _aligned(score_values: list[float], vol_values: list[float], tickers: list[str]):
    score = _score(score_values, ["2024-01-02"], tickers)
    vol = score.copy()
    vol.iloc[:] = vol_values
    vol.name = "vol"
    return score, vol


def test_vol_layers_are_equal_weighted_after_selection():
    score, vol = _aligned(
        [9, 7, 5, 3, 20, 16, 12, 4],
        [1, 1, 1, 1, 9, 9, 9, 9],
        ["A", "B", "C", "D", "E", "F", "G", "H"],
    )
    before_score = score.copy()
    before_vol = vol.copy()
    weights = vol_quantile_long_short(
        score=score, vol=vol, n_vol_groups=2, n_quantiles=2
    )
    assert weights.name == "weight"
    assert weights[weights > 0].sum() == 1
    assert weights[weights < 0].sum() == -1
    assert (weights == 0).sum() == 0
    assert sorted(weights[weights > 0].index.get_level_values("ticker")) == [
        "A",
        "B",
        "E",
        "F",
    ]
    pd.testing.assert_series_equal(score, before_score)
    pd.testing.assert_series_equal(vol, before_vol)


def test_small_vol_layer_is_skipped_and_thin_dates_drop_out():
    score, vol = _aligned(
        [3, 2, 1, 8, 7],
        [1, 2, 3, 4, 10],
        ["A", "B", "C", "D", "E"],
    )
    thin_score = _score([5.0, 4.0], ["2024-01-01"], ["A", "B"])
    thin_vol = thin_score.copy()
    thin_vol.iloc[:] = [1.0, 9.0]
    weights = vol_quantile_long_short(
        score=pd.concat([thin_score, score]),
        vol=pd.concat([thin_vol, vol]),
        n_vol_groups=2,
        n_quantiles=3,
    )
    dates = weights.index.get_level_values("date").strftime("%Y-%m-%d")
    assert dates.unique().tolist() == ["2024-01-02"]
    assert weights.tolist() == [1.0, -1.0]
    assert weights.index.get_level_values("ticker").tolist() == ["A", "C"]


def test_duplicate_vol_edges_collapse_before_score_quantiles():
    score, vol = _aligned(
        [8, 7, 6, 5, 4, 3, 2, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        ["A", "B", "C", "D", "E", "F", "G", "H"],
    )
    weights = vol_quantile_long_short(
        score=score, vol=vol, n_vol_groups=2, n_quantiles=4
    )
    assert weights.tolist() == [0.5, 0.5, -0.5, -0.5]
    assert weights.index.get_level_values("ticker").tolist() == ["A", "B", "G", "H"]


def test_missing_score_or_vol_is_removed():
    score = _score([5.0, 1.0, 3.0], ["2024-01-02"], ["A", "B", "C"])
    vol = _score([np.nan, 1.0, 1.0, 4.0], ["2024-01-02"], ["A", "B", "C", "D"])
    weights = vol_quantile_long_short(
        score=score, vol=vol, n_vol_groups=1, n_quantiles=2
    )
    assert "A" not in weights.index.get_level_values("ticker")
    assert "D" not in weights.index.get_level_values("ticker")
    assert weights.index.get_level_values("ticker").tolist() == ["B", "C"]
    assert weights.tolist() == [-1.0, 1.0]


def test_tied_scores_inside_a_vol_layer_are_skipped():
    score, vol = _aligned(
        [1, 1, 1, 1, 4, 3, 2, 1],
        [1, 1, 1, 1, 9, 9, 9, 9],
        ["A", "B", "C", "D", "E", "F", "G", "H"],
    )
    weights = vol_quantile_long_short(
        score=score, vol=vol, n_vol_groups=2, n_quantiles=2
    )
    assert weights.tolist() == [0.5, 0.5, -0.5, -0.5]
    assert weights.index.get_level_values("ticker").tolist() == ["E", "F", "G", "H"]


def test_vol_quantile_empty_input_and_bad_count():
    empty = _score([], [], [])
    result = vol_quantile_long_short(
        score=empty, vol=empty, n_vol_groups=2, n_quantiles=2
    )
    assert result.empty
    assert result.name == "weight"
    assert result.index.names == ["date", "ticker"]
    score, vol = _aligned([1.0, 2.0], [1.0, 2.0], ["A", "B"])
    with pytest.raises(ValueError, match="n_quantiles"):
        vol_quantile_long_short(score=score, vol=vol, n_vol_groups=2, n_quantiles=1)
