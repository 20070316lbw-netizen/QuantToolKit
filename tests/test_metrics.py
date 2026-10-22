import numpy as np
import pandas as pd

from quanttoolkit.metrics import (
    ic,
    ic_summary,
    max_drawdown,
    rank_ic,
    sharpe,
    total_return,
    win_rate,
)


def _panel(dates, tickers, values) -> pd.Series:
    index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(dates), tickers],
        names=["date", "ticker"],
    )
    return pd.Series(values, index=index)


def test_ic_and_rank_ic_use_cross_section_not_row_order():
    dates = ["2024-01-03"] * 3 + ["2024-01-02"] * 3
    tickers = ["C", "A", "B", "B", "C", "A"]
    factor = _panel(dates, tickers, [4.0, 1.0, 2.0, 2.0, 3.0, 1.0])
    forward = _panel(dates, tickers, [0.1, 0.0, 0.1, 0.2, 0.3, 0.1])
    factor_before = factor.copy()
    forward_before = forward.copy()

    pearson = ic(factor=factor, forward_return=forward)
    spearman = rank_ic(factor=factor, forward_return=forward)

    assert pearson.name == "ic"
    assert spearman.name == "rank_ic"
    assert pearson.index.name == "date"
    assert pearson.index.equals(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    assert float(pearson.iloc[0]) == 1.0
    assert float(spearman.iloc[0]) == 1.0
    assert np.isclose(pearson.iloc[1], 2 / np.sqrt(7))
    assert np.isclose(spearman.iloc[1], np.sqrt(0.75))
    pd.testing.assert_series_equal(factor, factor_before)
    pd.testing.assert_series_equal(forward, forward_before)


def test_ic_aligns_on_index_and_skips_incomplete_pairs():
    day = "2024-01-02"
    factor = _panel([day] * 3, ["A", "B", "C"], [1.0, 2.0, 9.0])
    forward = _panel([day, day], ["B", "A"], [3.0, 1.0])
    result = ic(factor=factor, forward_return=forward)
    assert np.isclose(result.iloc[0], 1.0)

    missing = _panel([day] * 3, ["A", "B", "C"], [0.1, np.nan, 0.3])
    kept = ic(
        factor=_panel([day] * 3, ["A", "B", "C"], [1.0, 2.0, 3.0]),
        forward_return=missing,
    )
    assert np.isclose(kept.iloc[0], 1.0)

    lone = ic(
        factor=_panel([day] * 2, ["A", "B"], [1.0, 2.0]),
        forward_return=_panel([day] * 2, ["A", "B"], [0.1, np.nan]),
    )
    assert lone.index.equals(pd.to_datetime([day]))
    assert lone.isna().all()


def test_ic_constant_cross_section_and_empty_input_stay_nan():
    day = "2024-01-02"
    result = ic(
        factor=_panel([day] * 3, ["A", "B", "C"], [1.0, 1.0, 1.0]),
        forward_return=_panel([day] * 3, ["A", "B", "C"], [0.1, 0.2, 0.4]),
    )
    assert result.isna().all()

    empty = pd.Series(
        dtype=float,
        index=pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex([]), pd.Index([])],
            names=["date", "ticker"],
        ),
    )
    blank = rank_ic(factor=empty, forward_return=empty)
    assert blank.empty
    assert blank.name == "rank_ic"
    assert blank.index.name == "date"


def test_ic_summary_skips_missing_dates_and_zero_dispersion():
    summary = ic_summary(ic=pd.Series([1.0, np.nan, -1.0], name="ic"))
    assert summary.name == "ic_summary"
    assert summary.index.tolist() == ["ic_mean", "ic_std", "ic_ir", "ic_pos"]
    assert summary["ic_mean"] == 0
    assert np.isclose(summary["ic_std"], np.sqrt(2))
    assert summary["ic_ir"] == 0
    assert summary["ic_pos"] == 0.5

    one = ic_summary(ic=pd.Series([0.4, np.nan]))
    assert one["ic_mean"] == 0.4
    assert np.isnan(one["ic_std"])
    assert np.isnan(one["ic_ir"])
    assert one["ic_pos"] == 1.0
    assert ic_summary(ic=pd.Series([0.2, 0.2]))["ic_ir"] == np.inf
    assert np.isnan(ic_summary(ic=pd.Series([0.0, 0.0]))["ic_ir"])
    assert np.isclose(
        ic_summary(ic=pd.Series([0.0, 0.2, -0.1]))["ic_pos"],
        1 / 3,
    )
    assert np.isnan(ic_summary(ic=pd.Series([np.nan], dtype=float))["ic_pos"])


def test_return_metrics_compound_in_index_order():
    returns = pd.Series(
        [-0.5, 1.0, -0.5],
        index=pd.to_datetime(["2024-01-04", "2024-01-02", "2024-01-03"]),
    )
    before = returns.copy()
    assert total_return(returns=returns) == -0.5
    assert max_drawdown(returns=returns) == -0.75
    assert max_drawdown(returns=pd.Series([-0.25, 0.1])) == -0.25
    assert max_drawdown(returns=pd.Series([0.1, 0.2])) == 0.0
    assert np.isclose(sharpe(returns=pd.Series([0.1, 0.3])), np.sqrt(2))
    assert sharpe(returns=pd.Series([0.1, 0.1])) == np.inf
    assert np.isnan(sharpe(returns=pd.Series([0.0, 0.0])))
    assert win_rate(returns=pd.Series([0.1, -0.2, 0.0, 0.3])) == 0.5
    pd.testing.assert_series_equal(returns, before)


def test_return_metrics_reject_missing_or_empty_samples():
    missing = pd.Series([0.1, np.nan, 0.2])
    empty = pd.Series(dtype=float)
    for metric in (total_return, sharpe, max_drawdown, win_rate):
        assert np.isnan(metric(returns=missing))
        assert np.isnan(metric(returns=empty))
