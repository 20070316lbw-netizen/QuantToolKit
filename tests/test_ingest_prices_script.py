import runpy
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ingest_prices.py"


def test_import_does_not_download(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    script = runpy.run_path(str(SCRIPT))
    assert capsys.readouterr().out == ""
    assert script["DB"] == SCRIPT.parents[1] / "data" / "quanttoolkit.db"
