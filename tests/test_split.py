import pandas as pd

from quanttoolkit.model.split import split_dates


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
