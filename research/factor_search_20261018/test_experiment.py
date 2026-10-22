"""验证研究公式的复权口径, 时间因果性与标签边界"""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from quanttoolkit.panel.returns import future_returns

SPEC = importlib.util.spec_from_file_location(
    "factor_experiment", Path(__file__).with_name("experiment.py")
)
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


def prices():
    index = pd.bdate_range("2021-01-01", periods=360, name="date")
    rng = np.random.default_rng(3)
    adj = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.001, 0.02, (360, 3)), axis=0)),
        index=index,
        columns=pd.Index(["A", "B", "C"], name="ticker"),
    )
    scale = pd.DataFrame(1.0, index=index, columns=adj.columns)
    scale.iloc[:180] = 2.0
    close = adj * scale
    opening = close * np.exp(rng.normal(0, 0.01, close.shape))
    return {
        "adj_close": adj,
        "close": close,
        "open": opening,
        "high": np.maximum(close, opening) * 1.01,
        "low": np.minimum(close, opening) * 0.99,
        "volume": pd.DataFrame(
            rng.integers(1000, 10000, close.shape), index=index, columns=adj.columns
        ),
    }


def test_intraday_and_adjusted_overnight_reconstruct_total_return():
    p = prices()
    factors = experiment._make_factors(p)
    reconstructed = factors["intraday_strength_63"] - factors["overnight_reversal_63"]
    expected = np.log(p["adj_close"] / p["adj_close"].shift(63))
    np.testing.assert_allclose(reconstructed.iloc[63:], expected.iloc[63:], atol=1e-12)


def test_all_factors_use_only_information_available_at_signal_date():
    p = prices()
    expected = experiment._make_factors(p)
    changed = {name: frame.copy() for name, frame in p.items()}
    for frame in changed.values():
        frame.iloc[300:] *= 7
    actual = experiment._make_factors(changed)
    for name in expected:
        assert_frame_equal(expected[name].iloc[:300], actual[name].iloc[:300])


def test_label_matches_gap_one_and_filled_day_is_excluded():
    p = prices()
    factors = experiment._make_factors(p)
    label, mask = experiment._label_and_mask(p, factors)
    np.testing.assert_allclose(
        label.iloc[270], p["adj_close"].iloc[292] / p["adj_close"].iloc[271] - 1
    )
    assert mask.iloc[270].all()
    assert not mask.iloc[:252].any().any()
    assert not mask.iloc[-22:].any().any()
    p["volume"].iloc[270, 0] = 0
    _, mask = experiment._label_and_mask(p, experiment._make_factors(p))
    assert not mask.iloc[270, 0]
    assert not mask.iloc[269, 0]


def test_period_purges_labels_crossing_boundary():
    calendar = pd.bdate_range("2022-11-01", "2023-02-01")
    selected = experiment._date_mask(calendar, "selection")
    exits = pd.Series(calendar).shift(-22)
    assert selected.any()
    assert (exits.loc[selected] <= pd.Timestamp("2022-12-31")).all()
    assert not selected[calendar >= "2023-01-01"].any()


def test_research_labels_match_package_calculation():
    p = prices()
    actual, _ = experiment._label_and_mask(p, experiment._make_factors(p))
    panel = p["adj_close"].stack().to_frame("adj_close")
    expected = future_returns(df=panel, n_periods=21, gap=1).unstack("ticker")
    assert_frame_equal(actual, expected)


def test_daily_rank_ic_handles_ties_and_paired_masks():
    x = pd.DataFrame([np.tile([1.0, 1.0, 3.0], 40)])
    y = pd.DataFrame([np.tile([3.0, 2.0, 1.0], 40)])
    mask = x.notna()
    assert np.isclose(experiment._daily_ic(x, y, mask).iloc[0], -np.sqrt(0.75))
    mask.iloc[0, :30] = False
    assert np.isnan(experiment._daily_ic(x, y, mask).iloc[0])
