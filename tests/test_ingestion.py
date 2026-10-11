import duckdb
import pandas as pd
import pytest

from quanttoolkit.db.connection import get_duckdb
from quanttoolkit.db.ingestion import ingestion_universe


@pytest.fixture
def universe():
    return pd.DataFrame(
        {
            "ticker": ["AAPL", "BRK-B"],
            "name": ["Apple Inc.", "Berkshire Hathaway"],
            "sub_industry": ["Technology Hardware", "Financial Services"],
            "cik": ["0000320193", "0001067983"],
            "sec_title": ["Apple Inc.", pd.NA],
        }
    )


def test_ingestion_persists_data(tmp_path, universe):
    path = tmp_path / "data" / "universe.db"
    with get_duckdb(path=path, read_only=False) as connection:
        assert ingestion_universe(df=universe, connection=connection) == 2

    with get_duckdb(path=path) as connection:
        rows = connection.execute("SELECT * FROM universe ORDER BY ticker").fetchall()

    assert rows == [
        ("AAPL", "Apple Inc.", "Technology Hardware", "0000320193", "Apple Inc."),
        ("BRK-B", "Berkshire Hathaway", "Financial Services", "0001067983", None),
    ]


def test_ingestion_replaces_current_members(universe):
    latest = universe.iloc[:1].copy()
    latest["sec_title"] = "APPLE INC"
    latest = latest[latest.columns[::-1]]

    with duckdb.connect(":memory:") as connection:
        connection.execute("CREATE TABLE other_data AS SELECT 42 AS value")
        ingestion_universe(df=universe, connection=connection)
        assert ingestion_universe(df=latest, connection=connection) == 1
        assert ingestion_universe(df=latest, connection=connection) == 1
        assert connection.execute(
            "SELECT ticker, sec_title FROM universe"
        ).fetchall() == [("AAPL", "APPLE INC")]
        assert connection.execute("SELECT value FROM other_data").fetchone() == (42,)


def test_ingestion_failure_rolls_back(universe):
    invalid = universe.rename(columns={"cik": "unknown_column"})
    with duckdb.connect(":memory:") as connection:
        ingestion_universe(df=universe, connection=connection)
        before = connection.execute("SELECT * FROM universe ORDER BY ticker").fetchall()

        with pytest.raises(duckdb.Error):
            ingestion_universe(df=invalid, connection=connection)

        assert (
            connection.execute("SELECT * FROM universe ORDER BY ticker").fetchall()
            == before
        )
        assert ingestion_universe(df=universe, connection=connection) == 2
