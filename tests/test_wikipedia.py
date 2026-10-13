import json
from io import BytesIO
from pathlib import Path
from urllib.error import URLError

import duckdb
import pandas as pd
import pytest

from quanttoolkit.db.operation import SQL_DIR, operation_sql, read_sql
from quanttoolkit.sources import wikipedia


@pytest.fixture
def json_path(tmp_path):
    path = tmp_path / "company_tickers.json"
    path.write_text(
        json.dumps(
            {
                "0": {
                    "cik_str": 1067983,
                    "ticker": "BRK-A",
                    "title": "BERKSHIRE HATHAWAY INC",
                },
                "1": {
                    "cik_str": 1067983,
                    "ticker": "BRK-B",
                    "title": "BERKSHIRE HATHAWAY INC",
                },
                "2": {"cik_str": 1872302, "ticker": "NA", "title": "Nano Labs Ltd"},
            }
        ),
        encoding="utf-8",
    )
    return path


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


@pytest.mark.parametrize("path_type", [str, Path])
def test_fetch_sp500_universe(page_requests, json_path, path_type):
    result = wikipedia.fetch_sp500_universe(path_type(json_path))

    assert len(page_requests) == 1
    request, timeout = page_requests[0]
    assert request.full_url == (
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )
    assert request.get_header("User-agent")
    assert timeout == 15
    assert result.index.equals(pd.RangeIndex(2))
    assert result["ticker"].tolist() == ["BRK-B", "NA"]
    assert result.columns.tolist() == [
        "ticker",
        "name",
        "sub_industry",
        "cik",
        "sec_title",
    ]
    members = result.set_index("ticker")
    assert members.loc["BRK-B", "cik"] == "0001067983"
    assert members.loc["NA", "cik"] == "0000000001"
    assert members.loc["BRK-B", "name"] == "Berkshire Hathaway"
    assert members.loc["BRK-B", "sub_industry"] == "Financial Services"
    assert members.loc["BRK-B", "sec_title"] == "BERKSHIRE HATHAWAY INC"
    assert pd.isna(members.loc["NA", "sec_title"])


def test_universe_ddl_round_trip(page_requests, json_path):
    result = wikipedia.fetch_sp500_universe(json_path)

    ddl = read_sql(SQL_DIR / "universe_ddl.sql")
    with duckdb.connect(":memory:") as connection:
        connection.execute(ddl)
        connection.execute(ddl)
        columns = connection.execute("DESCRIBE universe").fetchall()
        assert [column[0] for column in columns] == result.columns.tolist()
        assert [column[1] for column in columns] == ["VARCHAR"] * 5
        assert (
            operation_sql(read_sql(SQL_DIR / "universe.sql"), ddl, connection, result)
            == 2
        )
        rows = connection.execute("SELECT * FROM universe ORDER BY ticker").fetchall()

    assert rows == [
        (
            "BRK-B",
            "Berkshire Hathaway",
            "Financial Services",
            "0001067983",
            "BERKSHIRE HATHAWAY INC",
        ),
        ("NA", "National Bank", "Banks", "0000000001", None),
    ]


def test_missing_mapping_file(page_requests, tmp_path):
    with pytest.raises(FileNotFoundError, match="没有映射文件") as error:
        wikipedia.fetch_sp500_universe(tmp_path / "missing.json")

    assert "src/quanttoolkit/sources/USAGE.md" in str(error.value)
    assert page_requests == []


def test_request_error_propagates(monkeypatch, json_path):
    def fail_request(request, timeout):
        raise URLError("offline")

    monkeypatch.setattr(wikipedia, "urlopen", fail_request)
    with pytest.raises(URLError, match="offline"):
        wikipedia.fetch_sp500_universe(json_path)
