import numpy as np
import pandas as pd
import pytest

from quanttoolkit.panel.returns import future_returns, historical_return, log_returns
from quanttoolkit.panel.volatility import forward_volatility, history_vol

CALCULATIONS = [
    future_returns,
    historical_return,
    log_returns,
    history_vol,
    forward_volatility,
]


@pytest.mark.parametrize("calculation", CALCULATIONS)
def test_empty_missing_and_timezone_inputs(calculation, price_panel):
    empty = price_panel.iloc[:0]
    result = calculation(df=empty)
    assert isinstance(result, pd.Series)
    assert result.empty and result.dtype == np.dtype("float64")
    assert result.index.equals(empty.index)
    assert result.name

    missing = price_panel.copy()
    missing["close"] = missing["adj_close"] = np.nan
    assert calculation(df=missing).isna().all()

    zoned = price_panel.copy()
    zoned.index = pd.MultiIndex.from_arrays(
        [
            zoned.index.get_level_values("date").tz_localize("Asia/Shanghai"),
            zoned.index.get_level_values("ticker"),
        ],
        names=["date", "ticker"],
    )
    assert calculation(df=zoned).index.equals(zoned.sort_index().index)
