import numpy as np
import pandas as pd
import pytest

from quanttoolkit.returns import future_returns, historical_return, log_returns
from quanttoolkit.volatility import forward_volatility, history_vol

CALCULATIONS = [
    future_returns,
    historical_return,
    log_returns,
    history_vol,
    forward_volatility,
]


@pytest.mark.parametrize("calculation", CALCULATIONS)
def test_empty_nullable_missing_and_timezone_inputs(calculation, price_panel):
    empty = price_panel.iloc[:0]
    result = calculation(df=empty)
    assert isinstance(result, pd.Series)
    assert result.empty and result.dtype == np.dtype("float64")
    assert result.index.equals(empty.index)
    assert result.name

    for dtype in ["Float64", "Int64", "object"]:
        missing = price_panel.copy()
        missing["close"] = missing["adj_close"] = pd.Series(
            pd.NA, index=missing.index, dtype=dtype
        )
        assert calculation(df=missing).isna().all()

    nullable = price_panel.astype("Float64")
    pd.testing.assert_series_equal(
        calculation(df=nullable), calculation(df=price_panel)
    )

    mixed = price_panel.astype("object")
    mixed.iloc[0, :] = pd.NA
    expected = price_panel.copy()
    expected.iloc[0, :] = np.nan
    pd.testing.assert_series_equal(calculation(df=mixed), calculation(df=expected))

    zoned = price_panel.copy()
    zoned.index = pd.MultiIndex.from_arrays(
        [
            zoned.index.get_level_values("date").tz_localize("Asia/Shanghai"),
            zoned.index.get_level_values("ticker"),
        ],
        names=["date", "ticker"],
    )
    assert calculation(df=zoned).index.equals(zoned.sort_index().index)


@pytest.mark.parametrize("calculation", CALCULATIONS)
def test_reject_non_dataframe(calculation):
    with pytest.raises(TypeError, match="DataFrame"):
        calculation(df=pd.Series([1, 2]))


@pytest.mark.parametrize("calculation", CALCULATIONS)
@pytest.mark.parametrize(
    "case",
    [
        "flat",
        "names",
        "three_levels",
        "duplicate",
        "string_date",
        "nat",
        "missing_ticker",
        "blank_ticker",
    ],
)
def test_reject_invalid_index(calculation, price_panel, case):
    panel = price_panel.copy()
    dates = panel.index.get_level_values("date")
    tickers = panel.index.get_level_values("ticker")
    if case == "flat":
        panel = panel.reset_index()
    elif case == "names":
        panel.index = panel.index.set_names(["day", "symbol"])
    elif case == "three_levels":
        panel.index = pd.MultiIndex.from_arrays(
            [dates, tickers, range(len(panel))], names=["date", "ticker", "extra"]
        )
    elif case == "duplicate":
        panel = pd.concat([panel, panel.iloc[:1]])
    else:
        dates, tickers = dates.tolist(), tickers.tolist()
        if case == "string_date":
            dates = [str(value) for value in dates]
        elif case == "nat":
            dates[0] = pd.NaT
        elif case == "missing_ticker":
            tickers[0] = None
        elif case == "blank_ticker":
            tickers[0] = " "
        panel.index = pd.MultiIndex.from_arrays(
            [dates, tickers], names=["date", "ticker"]
        )
    with pytest.raises(ValueError):
        calculation(df=panel)


@pytest.mark.parametrize("calculation", CALCULATIONS)
@pytest.mark.parametrize("value", [0, -1, np.inf, -np.inf, "12", True, 1 + 2j])
def test_reject_invalid_price(calculation, price_panel, value):
    panel = price_panel.copy()
    # 整列替换以保留无效输入本身的类型, 避免 pandas 自动转为浮点数。
    panel["adj_close"] = panel["close"] = value
    with pytest.raises(ValueError, match="价格"):
        calculation(df=panel)


@pytest.mark.parametrize("calculation", CALCULATIONS)
@pytest.mark.parametrize("value", [True, np.bool_(False), "12"])
def test_object_prices_do_not_coerce_booleans_or_strings(
    calculation, price_panel, value
):
    panel = price_panel.astype("object")
    panel.iloc[0, :] = value
    with pytest.raises(ValueError, match="实数"):
        calculation(df=panel)


@pytest.mark.parametrize("calculation", CALCULATIONS)
def test_reject_missing_or_duplicate_selected_column(calculation, price_panel):
    with pytest.raises(ValueError, match="价格列"):
        calculation(df=price_panel, price_col="missing")
    with pytest.raises(ValueError, match="价格列"):
        calculation(df=price_panel, price_col=None)
    duplicated = pd.concat([price_panel, price_panel], axis=1)
    with pytest.raises(ValueError, match="唯一"):
        calculation(df=duplicated)


@pytest.mark.parametrize(
    "calculation, parameter, minimum",
    [
        (future_returns, "n_periods", 1),
        (historical_return, "n_periods", 1),
        (future_returns, "gap", 0),
        (history_vol, "window", 2),
        (forward_volatility, "window", 2),
        (forward_volatility, "gap", 0),
    ],
)
def test_period_validation_even_for_empty_input(
    calculation, parameter, minimum, price_panel
):
    for value in [True, np.bool_(False), 1.5, "2", None]:
        with pytest.raises(TypeError, match=parameter):
            calculation(df=price_panel.iloc[:0], **{parameter: value})
    for value in [-1, minimum - 1]:
        with pytest.raises(ValueError, match=parameter):
            calculation(df=price_panel.iloc[:0], **{parameter: value})
