from importlib import import_module
from inspect import signature

import pandas as pd
import pytest

from quanttoolkit.figure import plot_nav, plot_report
from quanttoolkit.metrics import (
    max_drawdown,
    prepare_forward_spread,
    prepare_portfolio_returns,
    rank_ic,
    total_return,
    win_rate,
)
from quanttoolkit.model.dataset import (
    assemble_samples,
    label_end_dates,
    prepare_label_end_dates,
)
from quanttoolkit.model.split import split_samples
from quanttoolkit.panel.returns import future_returns, historical_return
from quanttoolkit.panel.transfer import prepare_price_panel, to_date_ticker_frame
from quanttoolkit.registry import get_contract, list_contracts


def _resolve(name):
    module, _, member = name.rpartition(".")
    if module.rsplit(".", 1)[-1][0].isupper():
        module, _, owner = module.rpartition(".")
        return getattr(getattr(import_module(f"quanttoolkit.{module}"), owner), member)
    return getattr(import_module(f"quanttoolkit.{module}"), member)


def test_catalog_references_match_public_functions():
    contracts = list_contracts()
    for name, spec in contracts.items():
        function = _resolve(name)
        assert spec["stage"] and spec["definition"] and spec["output"], name
        assert set(spec["inputs"]) - {"self"} == set(signature(function).parameters) - {
            "self"
        }, name
        for rule in spec["inputs"].values():
            assert rule["requirements"], name
            assert set(rule["upstream"]) <= contracts.keys(), name
        if spec["track"] or any(
            rule["check"] or rule["source_policy"] != "described"
            for rule in spec["inputs"].values()
        ):
            assert hasattr(function, "__wrapped__"), name


def test_report_modes_and_catalog_lookup():
    vectorized = list_contracts(stage="报告", workflow="vectorized")
    event = list_contracts(stage="报告", workflow="event_driven")
    assert "figure.plots.plot_target_exposure" in vectorized
    assert "figure.plots.plot_target_exposure" not in event
    assert "figure.plots.plot_execution_slippage" in event
    assert "figure.plots.plot_execution_slippage" not in vectorized
    assert "figure.plots.plot_nav" in vectorized.keys() & event.keys()

    name = "panel.returns.future_returns"
    entry = get_contract(name)
    assert entry["inputs"]["df"]["source_policy"] == "required"
    entry["inputs"].clear()
    assert "df" in get_contract(name)["inputs"]


def test_price_origin_is_bound_to_the_input_and_selected_column(price_panel):
    copied = price_panel.copy()
    with pytest.raises(ValueError, match="future_returns.df 缺少上游"):
        future_returns(df=copied)
    prepared = prepare_price_panel(copied)
    assert future_returns(df=prepared, n_periods=1, gap=0).notna().any()

    prepared.iloc[0, prepared.columns.get_loc("close")] = -1.0
    assert future_returns(df=prepared, n_periods=1, gap=0).notna().any()
    with pytest.raises(ValueError, match="大于 0"):
        future_returns(df=prepared, price_col="close")
    duplicate = pd.concat([copied, copied.iloc[:1]])
    with pytest.raises(ValueError, match="键必须唯一"):
        prepare_price_panel(duplicate)


def test_factor_metrics_reject_swapped_project_outputs(price_panel):
    historical = historical_return(df=price_panel, n_periods=1)
    future = future_returns(df=price_panel, n_periods=1, gap=0)
    assert len(rank_ic(factor=historical, forward_return=future)) > 0
    with pytest.raises(ValueError, match="rank_ic.forward_return 不接受"):
        rank_ic(factor=historical, forward_return=historical)
    with pytest.raises(ValueError, match="rank_ic.factor 不接受"):
        rank_ic(factor=future, forward_return=future)
    with pytest.raises(TypeError, match="MultiIndex"):
        rank_ic(factor=historical, forward_return=future.reset_index(drop=True))
    assert len(rank_ic(factor=historical, forward_return=future.copy())) > 0


def test_portfolio_reports_reject_known_forward_spreads():
    dates = pd.date_range("2024-01-02", periods=2)
    raw = pd.Series([0.05, -0.02], index=dates)
    spread = prepare_forward_spread(raw)
    assert win_rate(returns=spread) == 0.5
    for metric in (total_return, max_drawdown):
        with pytest.raises(ValueError, match="不接受 metrics.prepare_forward_spread"):
            metric(returns=spread)
    with pytest.raises(ValueError, match="不接受 metrics.prepare_forward_spread"):
        prepare_portfolio_returns(spread)
    with pytest.raises(ValueError, match="缺少上游"):
        plot_nav(raw)
    with pytest.raises(ValueError, match="不接受 metrics.prepare_forward_spread"):
        plot_report("vectorized", portfolio_returns=spread)


def test_split_checks_label_window_and_accepts_reloaded_samples(price_panel):
    label = future_returns(df=price_panel, n_periods=1, gap=0)
    samples = assemble_samples(features=price_panel[["close"]], label=label)
    index = price_panel.index
    calendar = index.get_level_values("date").unique().sort_values()
    kwargs = {
        "dates": calendar,
        "train_end": calendar[1],
        "valid_end": calendar[3],
        "embargo": 0,
    }
    wrong_ends = label_end_dates(index, n_periods=1, gap=1)
    with pytest.raises(ValueError, match="标签期限.*不一致"):
        split_samples(samples=samples, label_end=wrong_ends, **kwargs)
    ends = label_end_dates(index, n_periods=1, gap=0)
    with pytest.raises(ValueError, match="split_samples.label_end 缺少上游"):
        split_samples(samples=samples, label_end=ends.copy(), **kwargs)

    expected = split_samples(samples=samples, label_end=ends, **kwargs)
    reloaded = to_date_ticker_frame(df=samples.reset_index())
    actual = split_samples(
        samples=reloaded, label_end=prepare_label_end_dates(ends.copy()), **kwargs
    )
    for role in expected:
        pd.testing.assert_frame_equal(actual[role], expected[role])
