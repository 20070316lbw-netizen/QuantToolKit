from datetime import date

import duckdb
import pandas as pd
import pytest

from quanttoolkit.db.connection import get_duckdb
from quanttoolkit.db.operation import SQL_DIR, operation_sql, read_sql


def write(connection, name, df):
    return operation_sql(
        read_sql(SQL_DIR / f"{name}.sql"),
        read_sql(SQL_DIR / f"{name}_ddl.sql"),
        connection,
        df,
    )


@pytest.fixture
def universe():
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "BRK-B"],
            "name": ["Apple Inc.", "Berkshire Hathaway"],
            "sub_industry": ["Technology Hardware", "Financial Services"],
        }
    )


def test_universe_persists_data(tmp_path, universe):
    path = tmp_path / "data" / "universe.db"
    with get_duckdb(path=path, read_only=False) as connection:
        assert write(connection, "universe", universe) == 2

    with get_duckdb(path=path) as connection:
        rows = connection.execute("SELECT * FROM universe ORDER BY ticker").fetchall()

    assert rows == [
        ("AAPL", "Apple Inc.", "Technology Hardware"),
        ("BRK-B", "Berkshire Hathaway", "Financial Services"),
    ]


@pytest.fixture
def prices():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-02", "2024-01-02", "2024-01-03"]),
            "ticker": ["AAPL", "MSFT", "AAPL"],
            "open": [100.0, 200.0, 103.0],
            "high": [105.0, 205.0, 108.0],
            "low": [99.0, 198.0, 102.0],
            "close": [102.0, 203.0, 107.0],
            "adj_close": [101.0, float("nan"), 106.0],
            "volume": [1000.0, 2000.0, 1500.0],
        }
    )


def test_prices_persist_data(tmp_path, prices):
    path = tmp_path / "prices.db"
    with get_duckdb(path=path, read_only=False) as connection:
        assert write(connection, "prices", prices) == 3

    with get_duckdb(path=path) as connection:
        rows = connection.execute(
            "SELECT * FROM prices ORDER BY ticker, date"
        ).fetchall()

    assert rows == [
        (date(2024, 1, 2), "AAPL", 100.0, 105.0, 99.0, 102.0, 101.0, 1000.0),
        (date(2024, 1, 3), "AAPL", 103.0, 108.0, 102.0, 107.0, 106.0, 1500.0),
        (date(2024, 1, 2), "MSFT", 200.0, 205.0, 198.0, 203.0, None, 2000.0),
    ]


def test_prices_update_and_append(prices, universe):
    latest = prices.iloc[[0, 2]].copy()
    latest.loc[0, "close"] = 104.0
    latest.loc[0, "adj_close"] = float("nan")
    latest.loc[2, "date"] = pd.Timestamp("2024-01-04")
    latest = latest[latest.columns[::-1]]

    with duckdb.connect(":memory:") as connection:
        write(connection, "universe", universe)
        write(connection, "prices", prices)
        assert write(connection, "prices", latest) == 2
        assert write(connection, "prices", latest) == 2
        assert connection.execute(
            "SELECT date, ticker, close, adj_close FROM prices ORDER BY ticker, date"
        ).fetchall() == [
            (date(2024, 1, 2), "AAPL", 104.0, None),
            (date(2024, 1, 3), "AAPL", 107.0, 106.0),
            (date(2024, 1, 4), "AAPL", 107.0, 106.0),
            (date(2024, 1, 2), "MSFT", 203.0, None),
        ]
        assert connection.execute("SELECT COUNT(*) FROM universe").fetchone() == (2,)


def test_prices_failure_rolls_back(prices):
    invalid = prices.copy()
    invalid.loc[0, "close"] = 104.0
    invalid.loc[1, "date"] = pd.Timestamp("2024-01-04")
    invalid.loc[2, "close"] = float("nan")

    with duckdb.connect(":memory:") as connection:
        write(connection, "prices", prices)
        before = connection.execute(
            "SELECT * FROM prices ORDER BY ticker, date"
        ).fetchall()

        with pytest.raises(duckdb.ConstraintException):
            write(connection, "prices", invalid)

        assert (
            connection.execute("SELECT * FROM prices ORDER BY ticker, date").fetchall()
            == before
        )
        assert write(connection, "prices", prices) == 3


def test_empty_prices_create_table_and_preserve_existing_rows(prices):
    with duckdb.connect(":memory:") as connection:
        assert write(connection, "prices", prices.iloc[:0]) == 0
        assert connection.execute("SELECT COUNT(*) FROM prices").fetchone() == (0,)
        write(connection, "prices", prices)
        assert write(connection, "prices", prices.iloc[:0]) == 0
        assert connection.execute("SELECT COUNT(*) FROM prices").fetchone() == (3,)


@pytest.mark.parametrize("column", ["date", "ticker", "close"])
def test_prices_required_columns_rollback_even_on_initial_creation(prices, column):
    invalid = prices.copy()
    invalid.loc[0, column] = None
    with duckdb.connect(":memory:") as connection:
        with pytest.raises(duckdb.ConstraintException):
            write(connection, "prices", invalid)
        assert connection.execute("SHOW TABLES").fetchall() == []
        assert write(connection, "prices", prices) == 3


def test_prices_date_is_daily_and_schema_rejects_duplicate_keys(prices):
    prices = prices.copy()
    prices.loc[0, "date"] = pd.Timestamp("2024-01-02 15:30")
    with duckdb.connect(":memory:") as connection:
        write(connection, "prices", prices)
        assert connection.execute(
            "SELECT date FROM prices WHERE ticker='AAPL' ORDER BY date"
        ).fetchall() == [(date(2024, 1, 2),), (date(2024, 1, 3),)]
        with pytest.raises(duckdb.ConstraintException):
            connection.execute("INSERT INTO prices SELECT * FROM prices")


def test_prices_bad_columns_and_read_only_database_preserve_data(tmp_path, prices):
    path = tmp_path / "prices.db"
    with get_duckdb(path=path, read_only=False) as connection:
        write(connection, "prices", prices)
        before = connection.execute(
            "SELECT * FROM prices ORDER BY ticker, date"
        ).fetchall()
        with pytest.raises(duckdb.Error):
            write(connection, "prices", prices.rename(columns={"close": "unknown"}))
        assert (
            connection.execute("SELECT * FROM prices ORDER BY ticker, date").fetchall()
            == before
        )
    with get_duckdb(path=path) as connection:
        with pytest.raises(duckdb.Error):
            write(connection, "prices", prices)
        assert (
            connection.execute("SELECT * FROM prices ORDER BY ticker, date").fetchall()
            == before
        )


def test_universe_replaces_current_members(universe):
    latest = universe.iloc[:1].copy()
    latest["name"] = "APPLE INC"
    latest = latest[latest.columns[::-1]]

    with duckdb.connect(":memory:") as connection:
        connection.execute("CREATE TABLE other_data AS SELECT 42 AS value")
        write(connection, "universe", universe)
        assert write(connection, "universe", latest) == 1
        assert write(connection, "universe", latest) == 1
        assert connection.execute("SELECT ticker, name FROM universe").fetchall() == [
            ("AAPL", "APPLE INC")
        ]
        assert connection.execute("SELECT value FROM other_data").fetchone() == (42,)


def test_universe_failure_rolls_back(universe):
    invalid = universe.rename(columns={"sub_industry": "unknown_column"})
    with duckdb.connect(":memory:") as connection:
        write(connection, "universe", universe)
        before = connection.execute("SELECT * FROM universe ORDER BY ticker").fetchall()

        with pytest.raises(duckdb.Error):
            write(connection, "universe", invalid)

        assert (
            connection.execute("SELECT * FROM universe ORDER BY ticker").fetchall()
            == before
        )
        assert write(connection, "universe", universe) == 2
