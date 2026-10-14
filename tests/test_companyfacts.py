import json
import runpy
from datetime import date
from pathlib import Path
from zipfile import ZipFile

import duckdb
import pandas as pd
import pytest

from quanttoolkit.db.operation import SQL_DIR, operation_sql, read_sql
from quanttoolkit.sources.companyfacts import read_company_facts

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ingest_companyfacts.py"


def write(connection, name, df):
    return operation_sql(
        read_sql(SQL_DIR / f"{name}.sql"),
        read_sql(SQL_DIR / f"{name}_ddl.sql"),
        connection,
        df,
    )


def company_zip(path: Path) -> Path:
    apple = {
        "cik": 320193,
        "entityName": "Apple Inc.",
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            {
                                "start": "2024-09-29",
                                "end": "2025-09-27",
                                "val": 112010000000,
                                "accn": "0000320193-25-000079",
                                "fy": 2025,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2025-10-31",
                                "frame": "CY2025",
                            },
                            {
                                "start": "2024-09-29",
                                "end": "2025-09-27",
                                "val": 100,
                                "accn": "0000320193-24-000001",
                                "form": "10-K",
                                "filed": "2024-11-01",
                            },
                        ]
                    }
                },
                "Assets": {
                    "units": {
                        "USD": [
                            {
                                "end": "2025-09-27",
                                "val": 359241000000,
                                "accn": "0000320193-25-000079",
                                "fy": 2025,
                                "fp": "FY",
                                "form": "10-K",
                                "filed": "2025-10-31",
                            }
                        ]
                    }
                },
            },
            "ifrs-full": {
                "Revenue": {
                    "units": {
                        "USD": [
                            {
                                "start": "2022-01-01",
                                "end": "2022-12-31",
                                "val": 10.5,
                                "accn": "0000849395-23-000001",
                                "form": "20-F",
                                "filed": "2023-03-10",
                                "fp": "",
                            }
                        ]
                    }
                }
            },
        },
    }
    empty = {"cik": "0000002110", "entityName": "", "facts": {}}
    zip_path = path / "companyfacts.zip"
    with ZipFile(zip_path, "w") as archive:
        archive.writestr("CIK0000320193.json", json.dumps(apple))
        archive.writestr("CIK0000002110.json", json.dumps(empty))
    return zip_path


def test_read_company_facts_splits_period_and_instant(tmp_path):
    period, instant = read_company_facts(
        company_zip(tmp_path), ["0000320193", "0000002110"]
    )

    assert period.columns.tolist() == [
        "cik",
        "taxonomy",
        "concept",
        "unit",
        "period_start",
        "period_end",
        "val",
        "accn",
        "form",
        "filed",
        "fy",
        "fp",
        "frame",
    ]
    assert "period_start" not in instant.columns
    assert len(period) == 3
    assert len(instant) == 1

    restated = period.loc[period["accn"] == "0000320193-24-000001"].iloc[0]
    assert restated["cik"] == "0000320193"
    assert restated["val"] == 100
    assert pd.isna(restated["fy"])
    assert pd.isna(restated["fp"])
    assert pd.isna(restated["frame"])

    revenue = period.loc[period["taxonomy"] == "ifrs-full"].iloc[0]
    assert revenue["concept"] == "Revenue"
    assert revenue["val"] == 10.5
    assert revenue["fp"] == ""
    assert pd.isna(revenue["frame"])

    asset = instant.iloc[0]
    assert asset["concept"] == "Assets"
    assert asset["period_end"] == pd.Timestamp("2025-09-27")
    assert pd.isna(asset["frame"])


def test_read_company_facts_missing_member(tmp_path):
    with pytest.raises(KeyError):
        read_company_facts(company_zip(tmp_path), ["0000000001"])


def test_facts_replace_and_append(tmp_path):
    period, instant = read_company_facts(company_zip(tmp_path), ["0000320193"])
    revised = period.loc[period["frame"].eq("CY2025")].copy()
    revised["val"] = 1.0
    extra = revised.copy()
    extra["accn"] = "0000320193-26-000001"
    extra["val"] = 2.0

    with duckdb.connect(":memory:") as connection:
        assert write(connection, "period_facts", period) == 3
        assert write(connection, "instant_facts", instant) == 1
        assert write(connection, "period_facts", revised) == 1
        assert write(connection, "period_facts", extra) == 1
        rows = connection.execute(
            """
            SELECT period_start, period_end, val, accn, filed, fy, fp, frame
            FROM period_facts
            WHERE concept = 'NetIncomeLoss'
            ORDER BY accn
            """
        ).fetchall()

    assert rows == [
        (
            date(2024, 9, 29),
            date(2025, 9, 27),
            100.0,
            "0000320193-24-000001",
            date(2024, 11, 1),
            None,
            None,
            None,
        ),
        (
            date(2024, 9, 29),
            date(2025, 9, 27),
            1.0,
            "0000320193-25-000079",
            date(2025, 10, 31),
            2025,
            "FY",
            "CY2025",
        ),
        (
            date(2024, 9, 29),
            date(2025, 9, 27),
            2.0,
            "0000320193-26-000001",
            date(2025, 10, 31),
            2025,
            "FY",
            "CY2025",
        ),
    ]


def test_facts_failure_rolls_back(tmp_path):
    period, _instant = read_company_facts(company_zip(tmp_path), ["0000320193"])
    invalid = period.copy()
    invalid.loc[0, "val"] = float("nan")

    with duckdb.connect(":memory:") as connection:
        write(connection, "period_facts", period)
        before = connection.execute("SELECT COUNT(*) FROM period_facts").fetchone()
        with pytest.raises(duckdb.ConstraintException):
            write(connection, "period_facts", invalid)
        assert (
            connection.execute("SELECT COUNT(*) FROM period_facts").fetchone() == before
        )


def test_import_does_not_ingest(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    script = runpy.run_path(str(SCRIPT))
    assert capsys.readouterr().out == ""
    assert script["DB"] == SCRIPT.parents[1] / "data" / "quanttoolkit.db"
    assert script["ZIP_PATH"] == SCRIPT.parents[1] / "data" / "companyfacts.zip"
