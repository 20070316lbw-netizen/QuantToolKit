from io import BytesIO
from urllib.error import URLError

import pandas as pd
import pytest

from quanttoolkit.sources import wikipedia


def test_fetch_sp500_universe(monkeypatch):
    html = b"""
    <table><tr><th>Other table</th></tr><tr><td>Ignore</td></tr></table>
    <table id="constituents">
      <tr><th>Symbol</th><th>Security</th><th>GICS Sector</th>
          <th>GICS Sub-Industry</th><th>Date added</th><th>CIK</th>
          <th>Headquarters Location</th></tr>
      <tr><td>NA</td><td>National Bank</td><td>Financials</td>
          <td>Banks</td><td></td><td>0000000001</td><td>City</td></tr>
      <tr><td>BRK.B</td><td>Berkshire Hathaway</td><td>Financials</td>
          <td>Financial Services</td><td>2010-02-16</td>
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
    before = pd.Timestamp.now().normalize()
    result = wikipedia.fetch_sp500_universe()
    after = pd.Timestamp.now().normalize()

    assert len(calls) == 1
    request, timeout = calls[0]
    assert request.full_url == (
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    )
    assert request.get_header("User-agent")
    assert timeout == 15
    assert result.index.names == ["date", "ticker"]
    assert result.index.is_monotonic_increasing
    assert result.index.get_level_values("ticker").tolist() == ["BRK-B", "NA"]
    dates = result.index.get_level_values("date")
    assert ((dates >= before) & (dates <= after)).all()
    assert result.columns.tolist() == [
        "name",
        "sector",
        "sub_industry",
        "date_added",
        "cik",
    ]
    members = result.droplevel("date")
    assert members.loc["BRK-B", "cik"] == "0001067983"
    assert members.loc["BRK-B", "date_added"] == pd.Timestamp("2010-02-16")
    assert pd.isna(members.loc["NA", "date_added"])


def test_request_error_propagates(monkeypatch):
    def fail_request(request, timeout):
        raise URLError("offline")

    monkeypatch.setattr(wikipedia, "urlopen", fail_request)
    with pytest.raises(URLError, match="offline"):
        wikipedia.fetch_sp500_universe()
