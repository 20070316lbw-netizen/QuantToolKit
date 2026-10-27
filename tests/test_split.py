import numpy as np
import pandas as pd
import pytest

from quanttoolkit.model.dataset import label_end_dates
from quanttoolkit.model.split import split_dates, split_samples
from quanttoolkit.panel.returns import future_returns


def test_split_purges_tails_and_sorts_unique_dates():
    calendar = pd.bdate_range("2024-01-01", periods=8, name="date")
    raw = calendar[[3, 0, 0, 7, 1, 2, 4, 5, 6]]
    result = split_dates(raw, train_end="2024-01-04", valid_end="2024-01-09", embargo=1)
    expected = pd.Series(
        ["train", "train", "train", pd.NA, "valid", "valid", pd.NA, "test"],
        index=pd.DatetimeIndex(calendar.to_numpy(), name="date"),
        dtype="object",
        name="split",
    )
    pd.testing.assert_series_equal(result, expected)
    assert raw.equals(calendar[[3, 0, 0, 7, 1, 2, 4, 5, 6]])


def test_zero_embargo_keeps_boundary_dates():
    calendar = pd.bdate_range("2024-01-01", periods=8, name="date")
    result = split_dates(
        calendar, train_end="2024-01-04", valid_end="2024-01-09", embargo=0
    )
    assert result.tolist() == [
        "train",
        "train",
        "train",
        "train",
        "valid",
        "valid",
        "valid",
        "test",
    ]


def test_short_segment_becomes_entirely_missing():
    calendar = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
    calendar = pd.DatetimeIndex(calendar, name="date")
    result = split_dates(
        calendar, train_end="2024-01-03", valid_end="2024-01-03", embargo=2
    )
    assert result.isna().tolist() == [True, True, False]
    assert result.iloc[2] == "test"


def test_naive_endpoint_follows_calendar_timezone_and_precision():
    calendar = (
        pd.bdate_range("2024-01-01", periods=8, name="date")
        .tz_localize("Asia/Shanghai")
        .as_unit("us")
    )
    result = split_dates(
        calendar, train_end="2024-01-04", valid_end="2024-01-09", embargo=1
    )
    assert result.index.dtype == calendar.dtype
    assert result.index.name == "date"
    assert result.loc[pd.Timestamp("2024-01-03", tz="Asia/Shanghai")] == "train"
    assert pd.isna(result.loc[pd.Timestamp("2024-01-04", tz="Asia/Shanghai")])
    assert result.loc[pd.Timestamp("2024-01-05", tz="Asia/Shanghai")] == "valid"


def test_empty_calendar_retains_timezone_precision_and_name():
    dates = pd.DatetimeIndex([], dtype="datetime64[ms, Asia/Shanghai]", name="date")
    result = split_dates(
        dates, train_end="2024-01-04", valid_end="2024-01-09", embargo=1
    )
    pd.testing.assert_series_equal(
        result,
        pd.Series(index=dates, dtype="object", name="split"),
    )


def test_split_samples_removes_sparse_labels_crossing_both_boundaries():
    dates = pd.bdate_range("2024-01-01", periods=12, name="date")
    complete = pd.MultiIndex.from_product([dates, ["A"]], names=["date", "ticker"])
    sparse = pd.MultiIndex.from_product(
        [dates[[0, 1, 5, 6, 10, 11]], ["B"]], names=["date", "ticker"]
    )
    prices = pd.DataFrame(
        {"adj_close": np.arange(18) + 10.0}, index=complete.append(sparse)
    )
    label = future_returns(df=prices, n_periods=1, gap=1)
    ends = label_end_dates(prices.index, n_periods=1, gap=1)
    samples = label.dropna().to_frame().assign(x=1).sample(frac=1, random_state=3)
    original = samples.copy()
    parts = split_samples(
        samples=samples,
        dates=dates,
        label_end=ends,
        train_end=dates[4],
        valid_end=dates[9],
        embargo=2,
    )
    assert (dates[0], "B") not in parts["train"].index
    assert (dates[5], "B") not in parts["valid"].index
    assert len(parts["train"]) == 3
    assert len(parts["valid"]) == 3
    for role, boundary in (("train", dates[4]), ("valid", dates[9])):
        assert ends.reindex(parts[role].index).le(boundary).all()
        assert parts[role].index.is_monotonic_increasing
    assert parts["test"].empty
    pd.testing.assert_frame_equal(samples, original)


def test_split_samples_excludes_unknown_dates_and_missing_ends():
    dates = pd.bdate_range("2024-01-01", periods=8, tz="Asia/Shanghai", name="date")
    index = pd.MultiIndex.from_product([dates, ["A"]], names=["date", "ticker"])
    samples = pd.DataFrame({"x": 1}, index=index)
    ends = label_end_dates(index, n_periods=1, gap=0).drop(index[0])
    parts = split_samples(
        samples=samples,
        dates=dates[1:],
        label_end=ends,
        train_end=dates[3],
        valid_end=dates[5],
        embargo=0,
    )
    assert len(parts["train"]) == 2
    assert len(parts["valid"]) == 1
    assert len(parts["test"]) == 1


@pytest.mark.parametrize(
    "embargo,train,valid",
    [(-1, "2024-01-01", "2024-01-02"), (0, "2024-01-02", "2024-01-01")],
)
def test_invalid_split_boundaries_raise(embargo, train, valid):
    with pytest.raises(ValueError):
        split_dates(
            pd.bdate_range("2024-01-01", periods=3),
            train_end=train,
            valid_end=valid,
            embargo=embargo,
        )
