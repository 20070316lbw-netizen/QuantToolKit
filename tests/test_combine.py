import doctest

import numpy as np
import pandas as pd
import pytest

from quanttoolkit.factors import combine
from quanttoolkit.factors.combine import combine_scores


def _z() -> pd.DataFrame:
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-01"]), ["A", "B", "C"]],
        names=["date", "ticker"],
    )
    return pd.DataFrame(
        {"f1": [-1.0, 0.0, 1.0], "f2": [-0.71, np.nan, 0.71]},
        index=index,
    )


def test_combine_docstring_example():
    result = doctest.testmod(combine)
    assert result.failed == 0
    assert result.attempted > 0


def test_combine_renormalizes_present_factors_and_keeps_sign():
    z = _z()
    before = z.copy(deep=True)
    score = combine_scores(z, {"f1": 0.25, "f2": -0.25})
    assert score.name == "score"
    assert score.index.equals(z.index)
    np.testing.assert_allclose(
        score.tolist(),
        [-0.145, 0.0, 0.145],
    )
    pd.testing.assert_frame_equal(z, before)


def test_combine_ignores_columns_left_out_of_weights():
    z = _z()
    z["f3"] = [10.0, 10.0, 10.0]
    score = combine_scores(z, {"f1": 1.0})
    np.testing.assert_allclose(score.tolist(), [-1.0, 0.0, 1.0])


def test_combine_all_missing_row_is_nan():
    z = _z()
    z.loc[pd.IndexSlice[:, "B"], :] = np.nan
    score = combine_scores(z)
    assert np.isnan(score.xs("B", level="ticker").iloc[0])


def test_combine_unknown_factor_raises():
    with pytest.raises(KeyError, match="missing"):
        combine_scores(_z(), {"missing": 1.0})
