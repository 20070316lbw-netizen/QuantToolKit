import runpy
from pathlib import Path

import duckdb
import pandas as pd

from quanttoolkit.db import ingestion_prices

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ingest_prices.py"


def test_import_is_safe_and_reader_works_outside_repo(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    script = runpy.run_path(str(SCRIPT))
    assert capsys.readouterr().out == ""
    assert script["DB"] == SCRIPT.parents[1] / "data" / "quanttoolkit.db"
    prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-03", "2024-01-02"]),
            "ticker": ["A", "A"],
            "open": [1.0, 2.0],
            "high": [1.0, 2.0],
            "low": [1.0, 2.0],
            "close": [1.0, 2.0],
            "adj_close": [1.0, 2.0],
            "volume": [1.0, 2.0],
        }
    )
    path = tmp_path / "prices.db"
    with duckdb.connect(str(path)) as connection:
        ingestion_prices(df=prices, connection=connection)
    script["main"](path=path)
    output = capsys.readouterr().out
    assert "date" in output and "ticker" in output
    assert output.index("2024-01-02") < output.index("2024-01-03")
    with duckdb.connect(str(path), read_only=True) as connection:
        assert connection.execute("SELECT COUNT(*) FROM prices").fetchone() == (2,)
