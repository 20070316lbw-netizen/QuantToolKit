import numpy as np
import pandas as pd
import pytest
import yfinance as yf

from quanttoolkit.sources import get_prices


@pytest.mark.parametrize(
    "options, expected_progress",
    [({}, True), ({"progress": True}, True), ({"progress": False}, False)],
    ids=["default", "enabled", "disabled"],
)
def test_progress_preserves_prices(monkeypatch, options, expected_progress):
    raw = pd.DataFrame(
        [
            [
                200.0,
                205.0,
                198.0,
                203.0,
                202.0,
                2000.0,
                100.0,
                105.0,
                99.0,
                102.0,
                101.0,
                1000.0,
            ]
        ],
        index=pd.DatetimeIndex(["2024-01-02"], name="Date"),
        columns=pd.MultiIndex.from_product(
            [["MSFT", "AAPL"], ["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
        ),
    )
    calls = []

    def download(tickers, **kwargs):
        calls.append((tickers, kwargs))
        return raw.copy()

    monkeypatch.setattr(yf, "download", download)
    result = get_prices(
        tickers=["MSFT", "AAPL"], start="2024-01-01", end="2024-02-01", **options
    )

    assert len(calls) == 1
    tickers, kwargs = calls[0]
    assert tickers == ["MSFT", "AAPL"]
    assert kwargs["progress"] is expected_progress
    assert kwargs["threads"] is True
    expected = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02"]),
            "ticker": ["AAPL", "MSFT"],
            "open": [100.0, 200.0],
            "high": [105.0, 205.0],
            "low": [99.0, 198.0],
            "close": [102.0, 203.0],
            "adj_close": [101.0, 202.0],
            "volume": [1000.0, 2000.0],
        }
    )
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize("raw", [None, pd.DataFrame()])
def test_empty_download_preserves_output_columns(monkeypatch, raw):
    monkeypatch.setattr(yf, "download", lambda *args, **kwargs: raw)
    result = get_prices("AAPL", start="2024-01-01")
    assert result.empty
    assert result.columns.tolist() == [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
    ]


def test_empty_ticker_list_fails_before_download(monkeypatch):
    def unexpected_download(*args, **kwargs):
        pytest.fail("无证券时不应调用下载")

    monkeypatch.setattr(yf, "download", unexpected_download)
    with pytest.raises(ValueError, match="不能为空"):
        get_prices([], start="2024-01-01")


@pytest.mark.parametrize("auto_adjust", [False, True])
def test_single_ticker_cleaning_adjustment_timezone_and_options(
    monkeypatch, auto_adjust
):
    raw = pd.DataFrame(
        {
            "Close": ["bad", "102", "101"],
            "Open": [1, "bad", 3],
            "Volume": [100, 200, 300],
        },
        index=pd.DatetimeIndex(
            ["2024-01-04 09:30", "2024-01-03 09:30", "2024-01-02 09:30"],
            name="Datetime",
            tz="America/New_York",
        ),
    )
    before = raw.copy(deep=True)
    calls = []

    def download(tickers, **kwargs):
        calls.append((tickers, kwargs))
        return raw

    monkeypatch.setattr(yf, "download", download)
    result = get_prices(
        "AAPL",
        start="2024-01-01",
        end="2024-02-01",
        interval="1h",
        auto_adjust=auto_adjust,
        progress=False,
    )
    assert calls == [
        (
            ["AAPL"],
            {
                "start": "2024-01-01",
                "end": "2024-02-01",
                "interval": "1h",
                "auto_adjust": auto_adjust,
                "group_by": "ticker",
                "progress": False,
                "threads": True,
            },
        )
    ]
    assert result["date"].tolist() == [
        pd.Timestamp("2024-01-02 09:30"),
        pd.Timestamp("2024-01-03 09:30"),
    ]
    assert result["ticker"].tolist() == ["AAPL", "AAPL"]
    assert result["close"].tolist() == [101, 102]
    np.testing.assert_allclose(result["open"], [3, np.nan])
    assert result[["high", "low"]].isna().all().all()
    if auto_adjust:
        np.testing.assert_allclose(result["adj_close"], [101, 102])
    else:
        assert result["adj_close"].isna().all()
    pd.testing.assert_frame_equal(raw, before)


@pytest.mark.parametrize(
    "tickers, expected_rows", [(["MISSING", "AAPL"], 1), (["MISSING"], 0)]
)
def test_missing_requested_ticker_is_skipped(monkeypatch, tickers, expected_rows):
    raw = pd.DataFrame(
        [[101.0]],
        index=pd.DatetimeIndex(["2024-01-02"], name="Date"),
        columns=pd.MultiIndex.from_tuples([("AAPL", "Close")]),
    )
    monkeypatch.setattr(yf, "download", lambda *args, **kwargs: raw)
    result = get_prices(tickers, start="2024-01-01")
    assert len(result) == expected_rows
    if expected_rows:
        assert result["ticker"].tolist() == ["AAPL"]


def test_single_level_columns_cannot_identify_multiple_tickers(monkeypatch):
    raw = pd.DataFrame(
        {"Close": [101.0]}, index=pd.DatetimeIndex(["2024-01-02"], name="Date")
    )
    monkeypatch.setattr(yf, "download", lambda *args, **kwargs: raw)
    with pytest.raises(ValueError, match="身份"):
        get_prices(["AAPL", "MSFT"], start="2024-01-01")


def test_all_bad_closes_produce_empty_frame(monkeypatch):
    raw = pd.DataFrame(
        {"Close": [None, "bad"]},
        index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="Date"),
    )
    monkeypatch.setattr(yf, "download", lambda *args, **kwargs: raw)
    result = get_prices("AAPL", start="2024-01-01", auto_adjust=True)
    assert result.empty and len(result.columns) == 8


def test_download_error_is_not_hidden(monkeypatch):
    def failed_download(*args, **kwargs):
        raise RuntimeError("download failed")

    monkeypatch.setattr(yf, "download", failed_download)
    with pytest.raises(RuntimeError, match="download failed"):
        get_prices("AAPL", start="2024-01-01")
