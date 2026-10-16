from datetime import date
from pathlib import Path
from zipfile import ZipFile

import duckdb
import pandas as pd
import pytest

from quanttoolkit.db.operation import SQL_DIR, operation_sql, read_sql
from quanttoolkit.sources.french import read_french_factors

ROOT = Path(__file__).resolve().parents[1]
FF3_PATH = ROOT / "data" / "ff3_daily.zip"
FF5_PATH = ROOT / "data" / "ff5_daily.zip"

FF3_TEXT = """This file was created by using the 202608 CRSP database.

,Mkt-RF,SMB,HML,RF
19260701,    0.09,   -0.23,   -0.28,    0.01
19260702,    0.45,   -0.34,   -0.03,    0.01

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""

FF5_TEXT = """This file was created by using the 202608 CRSP database.

,Mkt-RF,SMB,HML,RMW,CMA,RF
19630701,   -0.67,    0.00,   -0.33,   -0.01,    0.16,    0.01
19630702,    0.79,   -0.26,    0.26,   -0.07,   -0.20,    0.01

Copyright 2026 Eugene F. Fama and Kenneth R. French
"""


def write_zip(path: Path, text: str) -> None:
    with ZipFile(path, "w") as archive:
        archive.writestr("factors.csv", text)


def test_read_ff3_scales_percent_and_drops_the_copyright(tmp_path):
    path = tmp_path / "ff3.zip"
    write_zip(path, FF3_TEXT)
    factors = read_french_factors(path)

    assert factors.index.name == "date"
    assert factors.columns.tolist() == ["mkt_rf", "smb", "hml", "rf"]
    assert factors.index.tolist() == list(pd.to_datetime(["1926-07-01", "1926-07-02"]))
    assert factors.loc[pd.Timestamp("1926-07-01"), "mkt_rf"] == pytest.approx(0.0009)
    assert factors.loc[pd.Timestamp("1926-07-01"), "smb"] == pytest.approx(-0.0023)
    assert factors.loc[pd.Timestamp("1926-07-01"), "rf"] == pytest.approx(0.0001)


def test_read_ff5_keeps_profitability_and_investment(tmp_path):
    path = tmp_path / "ff5.zip"
    write_zip(path, FF5_TEXT)
    factors = read_french_factors(path)

    assert factors.columns.tolist() == ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]
    assert factors.loc[pd.Timestamp("1963-07-01"), "mkt_rf"] == pytest.approx(-0.0067)
    assert factors.loc[pd.Timestamp("1963-07-01"), "rmw"] == pytest.approx(-0.0001)
    assert factors.loc[pd.Timestamp("1963-07-01"), "cma"] == pytest.approx(0.0016)


def test_missing_factor_table_raises(tmp_path):
    path = tmp_path / "empty.zip"
    write_zip(path, "no table here\n")
    with pytest.raises(ValueError, match="没有因子表"):
        read_french_factors(path)


def test_ff_factors_replace_both_models(tmp_path):
    write_zip(tmp_path / "ff3.zip", FF3_TEXT)
    write_zip(tmp_path / "ff5.zip", FF5_TEXT)
    ff3 = read_french_factors(tmp_path / "ff3.zip")
    ff5 = read_french_factors(tmp_path / "ff5.zip")
    ff3 = ff3.reset_index()
    ff5 = ff5.reset_index()
    ff3.insert(1, "model", "ff3")
    ff5.insert(1, "model", "ff5")
    factors = pd.concat([ff3, ff5], ignore_index=True)
    revised = ff3.iloc[:1].copy()
    revised.loc[revised.index[0], "mkt_rf"] = 0.02

    with duckdb.connect(":memory:") as connection:
        assert write(connection, factors) == 4
        assert write(connection, revised) == 1
        rows = connection.execute(
            """
            SELECT model, date, mkt_rf, rmw, cma
            FROM ff_factors
            ORDER BY model, date
            """
        ).fetchall()

    assert rows == [("ff3", date(1926, 7, 1), 0.02, None, None)]


def write(connection, df):
    return operation_sql(
        read_sql(SQL_DIR / "ff_factors.sql"),
        read_sql(SQL_DIR / "ff_factors_ddl.sql"),
        connection,
        df,
    )


@pytest.mark.skipif(
    not FF3_PATH.is_file() or not FF5_PATH.is_file(),
    reason="本地没有官方因子压缩包",
)
def test_official_daily_files():
    ff3 = read_french_factors(FF3_PATH)
    ff5 = read_french_factors(FF5_PATH)
    overlap = ff3.index.intersection(ff5.index)

    assert ff3.columns.tolist() == ["mkt_rf", "smb", "hml", "rf"]
    assert ff5.columns.tolist() == ["mkt_rf", "smb", "hml", "rmw", "cma", "rf"]
    assert ff3.index[0] == pd.Timestamp("1926-07-01")
    assert ff5.index[0] == pd.Timestamp("1963-07-01")
    assert ff3.loc[pd.Timestamp("1926-07-01"), "mkt_rf"] == pytest.approx(0.0009)
    assert ff5.loc[pd.Timestamp("1963-07-01"), "mkt_rf"] == pytest.approx(-0.0067)
    assert ff3.index.is_monotonic_increasing and ff3.index.is_unique
    assert ff5.index.is_monotonic_increasing and ff5.index.is_unique
    assert bool((ff3.abs() < 1).all().all())
    assert bool((ff5.abs() < 1).all().all())
    pd.testing.assert_series_equal(
        ff3.loc[overlap, "mkt_rf"],
        ff5.loc[overlap, "mkt_rf"],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        ff3.loc[overlap, "rf"],
        ff5.loc[overlap, "rf"],
        check_names=False,
    )
    assert not ff3.loc[overlap, "smb"].equals(ff5.loc[overlap, "smb"])
