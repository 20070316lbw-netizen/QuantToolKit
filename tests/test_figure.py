import doctest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from quanttoolkit.figure import plot_report, plots, report
from quanttoolkit.metrics import max_drawdown, prepare_portfolio_returns


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_vectorized_report_saves_charts_and_matches_drawdown(tmp_path):
    dates = pd.bdate_range("2024-01-02", periods=3, tz="Asia/Shanghai")
    returns = pd.Series([-0.10, 0.20, -0.25], index=dates)
    index = pd.MultiIndex.from_product(
        [dates[[0, 2]], ["A", "B"]], names=["date", "ticker"]
    )
    weights = pd.Series([0.7, -0.3, 0.5, -0.5], index=index)
    output = tmp_path / "backtest.png"
    figure = plot_report(
        "vectorized",
        portfolio_returns=prepare_portfolio_returns(returns.iloc[::-1]),
        target_weights=weights,
        output=output,
    )

    nav, drawdown, exposure = figure.axes
    assert nav.lines[0].get_ydata() == pytest.approx([0.9, 1.08, 0.81])
    assert drawdown.lines[0].get_ydata() == pytest.approx([-0.1, 0, -0.25])
    assert min(drawdown.lines[0].get_ydata()) == pytest.approx(
        max_drawdown(returns=returns)
    )
    assert exposure.lines[0].get_ydata() == pytest.approx([1, 1])
    assert exposure.lines[1].get_ydata() == pytest.approx([0.4, 0])
    assert nav.get_xlim() == drawdown.get_xlim() == exposure.get_xlim()
    assert output.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_event_report_keeps_fills_at_the_same_time():
    dates = pd.date_range("2024-01-02", periods=2)
    returns = pd.Series([0.01, -0.02], index=dates)
    fills = pd.Series([1.5, -1.0], index=dates[[1, 1]])
    figure = plot_report(
        "event_driven",
        portfolio_returns=prepare_portfolio_returns(returns),
        slippage_bps=fills,
    )
    offsets = figure.axes[-1].collections[0].get_offsets()
    assert offsets[0, 0] == offsets[1, 0]
    assert offsets[:, 1].tolist() == [1.5, -1.0]

    empty = plot_report(
        "event_driven",
        portfolio_returns=prepare_portfolio_returns(returns),
        slippage_bps=fills.iloc[:0],
    )
    assert empty.axes[-1].texts[0].get_text() == "No fills"
    assert empty.axes[-2].get_xticklabels()


def test_factor_report_preserves_uncompounded_spread_and_ic_gaps():
    dates = pd.date_range("2024-01-02", periods=3)
    ic = pd.Series([0.1, np.nan, 0.2], index=dates, name="rank_ic")
    spread = pd.Series([0.06, 0.04, -0.08], index=dates)
    figure = plot_report("factor_research", ic=ic, forward_spread=spread)
    ic_ax, spread_ax = figure.axes
    assert np.isnan(ic_ax.lines[0].get_ydata()[1])
    assert spread_ax.lines[0].get_ydata().tolist() == spread.tolist()


def test_report_rejects_wrong_artifacts_and_closes_on_plot_error():
    dates = pd.date_range("2024-01-02", periods=2)
    returns = pd.Series([0.01, -0.02], index=dates)
    with pytest.raises(ValueError, match="portfolio_returns"):
        plot_report("vectorized", forward_spread=returns)
    with pytest.raises(ValueError, match="slippage_bps"):
        plot_report("vectorized", portfolio_returns=returns, slippage_bps=returns)
    prepared = prepare_portfolio_returns(returns)
    prepared.iloc[-1] = np.nan
    with pytest.raises(ValueError, match="portfolio_returns"):
        plot_report("event_driven", portfolio_returns=prepared)
    assert plt.get_fignums() == []


@pytest.mark.parametrize("module", [plots, report])
def test_figure_docstrings(module):
    result = doctest.testmod(module)
    assert result.failed == 0
    assert result.attempted > 0
