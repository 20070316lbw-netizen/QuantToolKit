from io import BytesIO
from urllib.error import URLError

import duckdb
import pandas as pd
import pytest

from quanttoolkit.db.operation import SQL_DIR, operation_sql, read_sql
from quanttoolkit.sources import wikipedia


@pytest.fixture
def page_requests(monkeypatch):
    html = b"""
    <table><tr><th>Other table</th></tr><tr><td>Ignore</td></tr></table>
    <table id="constituents">
      <tr><th>Symbol</th><th>Security</th>
          <th>GICS Sub-Industry</th><th>CIK</th>
          <th>Headquarters Location</th></tr>
      <tr><td>NA</td><td>National Bank</td>
          <td>Banks</td><td>0000000001</td><td>City</td></tr>
      <tr><td>BRK.B</td><td>Berkshire Hathaway</td>
          <td>Financial Services</td>
          <td>0001067983</td><td>Omaha</td></tr>
    </table>
    <table id="changes"><tr><th>Removed</th></tr>
      <tr><td>OLD</td></tr></table>
    """
    calls = []

    def fake_urlopen(request, timeout):
        calls.append((request, timeout))
        return BytesIO(html)

    monkeypatch.setattr(wikipedia, "urlopen", fake_urlopen)
    return calls


def test_fetch_sp500_universe(page_requests):
    result = wikipedia.fetch_sp500_universe()

    assert len(page_requests) == 1
    request, timeout = page_requests[0]
    assert request.full_url == (
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )
    assert request.get_header("User-agent")
    assert timeout == 15
    assert result.index.equals(pd.RangeIndex(2))
    assert result["ticker"].tolist() == ["BRK-B", "NA"]
    assert result.columns.tolist() == ["ticker", "name", "sub_industry"]
    members = result.set_index("ticker")
    assert members.loc["BRK-B", "name"] == "Berkshire Hathaway"
    assert members.loc["BRK-B", "sub_industry"] == "Financial Services"


def test_universe_ddl_round_trip(page_requests):
    result = wikipedia.fetch_sp500_universe()

    ddl = read_sql(SQL_DIR / "universe_ddl.sql")
    with duckdb.connect(":memory:") as connection:
        connection.execute(ddl)
        connection.execute(ddl)
        columns = connection.execute("DESCRIBE universe").fetchall()
        assert [column[0] for column in columns] == result.columns.tolist()
        assert [column[1] for column in columns] == ["VARCHAR"] * 3
        assert (
            operation_sql(read_sql(SQL_DIR / "universe.sql"), ddl, connection, result)
            == 2
        )
        rows = connection.execute("SELECT * FROM universe ORDER BY ticker").fetchall()

    assert rows == [
        ("BRK-B", "Berkshire Hathaway", "Financial Services"),
        ("NA", "National Bank", "Banks"),
    ]


def test_request_error_propagates(monkeypatch):
    def fail_request(request, timeout):
        raise URLError("offline")

    monkeypatch.setattr(wikipedia, "urlopen", fail_request)
    with pytest.raises(URLError, match="offline"):
        wikipedia.fetch_sp500_universe()
