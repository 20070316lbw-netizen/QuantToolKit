import numpy as np
import pandas as pd
import yfinance as yf

from quanttoolkit.db import get_duckdb, ingestion_prices
from quanttoolkit.rebalance import rebalance_dates_belong
from quanttoolkit.returns import future_returns, historical_return, log_returns
from quanttoolkit.sources import get_prices
from quanttoolkit.transfer import to_date_ticker_frame
from quanttoolkit.volatility import forward_volatility, history_vol


def test_download_ingest_reopen_and_calculate(tmp_path, monkeypatch):
    raw = pd.DataFrame(
        [[100.0, 50.0], [110.0, 50.0], [99.0, 100.0], [118.8, 100.0]],
        index=pd.bdate_range("2024-01-02", periods=4, name="Date"),
        columns=pd.MultiIndex.from_tuples([("A", "Close"), ("B", "Close")]),
    )
    monkeypatch.setattr(yf, "download", lambda *args, **kwargs: raw.copy())
    downloaded = get_prices(
        ["B", "A"],
        start="2024-01-02",
        end="2024-01-08",
        auto_adjust=True,
        progress=False,
    )
    path = tmp_path / "nested" / "prices.db"
    with get_duckdb(path=path, read_only=False) as connection:
        assert ingestion_prices(df=downloaded, connection=connection) == 8
    with get_duckdb(path=path) as connection:
        stored = connection.execute(
            "SELECT * FROM prices ORDER BY ticker DESC, date DESC"
        ).df()
    panel = to_date_ticker_frame(df=stored)
    np.testing.assert_allclose(
        historical_return(df=panel, n_periods=2).xs("A", level="ticker"),
        [np.nan, np.nan, -0.01, 0.08],
    )
    np.testing.assert_allclose(
        future_returns(df=panel, n_periods=2, gap=1).xs("A", level="ticker"),
        [0.08, np.nan, np.nan, np.nan],
    )
    np.testing.assert_allclose(
        log_returns(df=panel).xs("B", level="ticker"), [np.nan, 0, np.log(2), 0]
    )
    np.testing.assert_allclose(
        history_vol(df=panel, window=2).xs("A", level="ticker"),
        [np.nan, np.nan, 0.2 / np.sqrt(2), 0.3 / np.sqrt(2)],
    )
    np.testing.assert_allclose(
        forward_volatility(df=panel, window=2, gap=1).xs("A", level="ticker"),
        [0.3 / np.sqrt(2), np.nan, np.nan, np.nan],
    )
    belongs = rebalance_dates_belong(panel.index.get_level_values("date"), freq=2)
    assert len(belongs) == 4
    assert belongs.tolist() == [raw.index[0], raw.index[0], raw.index[2], raw.index[2]]
