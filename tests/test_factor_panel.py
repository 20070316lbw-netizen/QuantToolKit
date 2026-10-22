import numpy as np
import pandas as pd

from quanttoolkit.db.connection import get_duckdb
from quanttoolkit.db.operation import SQL_DIR, operation_sql, read_sql
from quanttoolkit.factors.logvol_base import base_logvolfactor
from quanttoolkit.factors.mom_base import base_momfactor
from quanttoolkit.panel.transfer import to_date_ticker_frame


def _prices():
    dates = pd.bdate_range("2024-01-02", periods=70)
    frame = pd.DataFrame({"date": dates, "ticker": "A"})
    frame["adj_close"] = np.exp(np.arange(len(dates)) * 0.01)
    frame["open"] = frame["adj_close"] * 0.99
    frame["close"] = frame["adj_close"] * 2
    frame["high"] = frame["adj_close"]
    frame["low"] = frame["adj_close"]
    frame["volume"] = 1000.0
    frame.loc[65, "volume"] = 0.0
    late = pd.DataFrame({"date": dates[-10:], "ticker": "B"})
    late["adj_close"] = 10.0
    late["open"] = 10.0
    late["close"] = 10.0
    late["high"] = 10.0
    late["low"] = 10.0
    late["volume"] = 50.0
    return pd.concat([frame, late], ignore_index=True)


def _ff5(dates):
    return pd.DataFrame(
        {
            "date": dates,
            "model": "ff5",
            "mkt_rf": 0.01,
            "smb": 0.02,
            "hml": -0.01,
            "rmw": 0.03,
            "cma": 0.04,
            "rf": 0.001,
        }
    )


def test_factor_panel_starts_with_mom_63_and_matches_python(tmp_path):
    prices = _prices()
    dates = pd.bdate_range("2024-01-02", periods=70)
    start = dates[63]
    with get_duckdb(path=tmp_path / "factors.db", read_only=False) as connection:
        operation_sql(
            read_sql(SQL_DIR / "prices.sql"),
            read_sql(SQL_DIR / "prices_ddl.sql"),
            connection,
            prices,
        )
        operation_sql(
            read_sql(SQL_DIR / "ff_factors.sql"),
            read_sql(SQL_DIR / "ff_factors_ddl.sql"),
            connection,
            _ff5(dates[63:]),
        )
        connection.execute(read_sql(SQL_DIR / "factor_panel.sql"))
        panel = connection.execute(
            "SELECT * FROM factor_panel ORDER BY ticker, date"
        ).df()

    assert panel["date"].min() == start
    assert set(panel["ticker"]) == {"A", "B"}
    late = panel[panel["ticker"] == "B"]
    assert late["mom_63"].isna().all()
    assert late["mom_21"].isna().all()
    assert late["logvol_21"].isna().all()
    assert np.allclose(late["overnight"], 0)
    assert np.allclose(late["intraday"], 0)
    assert np.allclose(late["dvol"], 0)

    indexed = to_date_ticker_frame(df=prices.copy())
    mom63 = base_momfactor(indexed, window=63)
    mom21 = base_momfactor(indexed, window=21)
    logvol = base_logvolfactor(indexed, window=21)
    got = panel[panel["ticker"] == "A"].set_index("date")
    raw = prices[prices["ticker"] == "A"].set_index("date").sort_index()
    traded = raw["volume"] > 0
    previous_volume = raw["volume"].shift(1)
    both_traded = traded & (previous_volume > 0)
    overnight = pd.Series(np.nan, index=raw.index)
    intraday = pd.Series(np.nan, index=raw.index)
    dvol = pd.Series(np.nan, index=raw.index)
    overnight.loc[traded] = np.log(
        raw.loc[traded, "open"] / raw["close"].shift(1).loc[traded]
    )
    intraday.loc[traded] = np.log(raw.loc[traded, "close"] / raw.loc[traded, "open"])
    dvol.loc[both_traded] = np.log(
        raw.loc[both_traded, "volume"] / previous_volume.loc[both_traded]
    )
    assert "rev_5" not in panel.columns
    for column, series in (
        ("mom_63", mom63.xs("A", level="ticker")),
        ("mom_21", mom21.xs("A", level="ticker")),
        ("logvol_21", logvol.xs("A", level="ticker")),
        ("overnight", overnight),
        ("intraday", intraday),
        ("dvol", dvol),
    ):
        expected = series.copy()
        expected.index = pd.DatetimeIndex(expected.index).tz_localize(None)
        aligned = expected.reindex(pd.DatetimeIndex(pd.to_datetime(got.index)))
        assert np.allclose(got[column].to_numpy(), aligned.to_numpy(), equal_nan=True)

    zero_volume_day = pd.Timestamp(dates[65])
    assert pd.isna(got.loc[zero_volume_day, "dvol"])
    assert pd.isna(got.loc[zero_volume_day, "overnight"])
    assert pd.isna(got.loc[zero_volume_day, "intraday"])
    assert pd.isna(got.loc[pd.Timestamp(dates[66]), "dvol"])
    for offset in range(65, 70):
        assert pd.isna(got.loc[pd.Timestamp(dates[offset]), "rel_volume_21"])
    assert got.loc[start, "mkt_rf"] == 0.01
    assert got.loc[start, "rel_volume_21"] == 1
