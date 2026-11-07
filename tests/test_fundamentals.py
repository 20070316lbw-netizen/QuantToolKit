import doctest

import duckdb
import numpy as np
import pandas as pd
import pytest

from quanttoolkit import fundamentals as fundamentals_module
from quanttoolkit.fundamentals import (
    align_filings,
    filing_panel,
    fundamental_fact_mapping,
    read_filing_panel,
    read_fundamental_facts,
)

INSTANT_COLUMNS = [
    "cik",
    "taxonomy",
    "concept",
    "unit",
    "period_end",
    "filed",
    "val",
    "accn",
]
PERIOD_COLUMNS = [
    "cik",
    "taxonomy",
    "concept",
    "unit",
    "period_start",
    "period_end",
    "filed",
    "val",
    "accn",
]


def members(*pairs):
    return pd.DataFrame(list(pairs), columns=["ticker", "cik"])


def instant_row(
    concept,
    end,
    filed,
    val,
    *,
    cik="1",
    unit="USD",
    taxonomy="us-gaap",
    accn="a",
):
    return {
        "cik": cik,
        "taxonomy": taxonomy,
        "concept": concept,
        "unit": unit,
        "period_end": end,
        "filed": filed,
        "val": val,
        "accn": accn,
    }


def period_row(
    concept,
    start,
    end,
    filed,
    val,
    *,
    cik="1",
    accn="a",
):
    return {
        "cik": cik,
        "taxonomy": "us-gaap",
        "concept": concept,
        "unit": "USD",
        "period_start": start,
        "period_end": end,
        "filed": filed,
        "val": val,
        "accn": accn,
    }


def panel_from(instant=(), period=(), pairs=(("A", "1"),)):
    instant_frame = pd.DataFrame(list(instant), columns=INSTANT_COLUMNS)
    period_frame = pd.DataFrame(list(period), columns=PERIOD_COLUMNS)
    return filing_panel(instant_frame, period_frame, members(*pairs))


def value_at(panel, filed, column, ticker="A"):
    row = panel.loc[(panel["ticker"] == ticker) & (panel["filed"] == filed), column]
    return float(row.iloc[0])


def prices_for(*rows):
    frame = pd.DataFrame(list(rows), columns=["date", "ticker", "close"])
    frame["date"] = pd.to_datetime(frame["date"])
    frame["adj_close"] = 999.0
    return frame.set_index(["date", "ticker"])


def test_current_statement_ignores_older_comparative_on_the_same_filing():
    panel = panel_from(
        instant=[
            instant_row("Assets", "2023-12-31", "2025-02-01", 8),
            instant_row("Assets", "2024-12-31", "2025-02-01", 10),
        ]
    )
    assert value_at(panel, "2025-02-01", "assets") == 10
    assert value_at(panel, "2025-02-01", "asset_growth") == pytest.approx(0.25)


def test_later_amendment_of_an_old_period_does_not_replace_current_assets():
    panel = panel_from(
        instant=[
            instant_row("Assets", "2024-09-30", "2024-11-01", 100, accn="1"),
            instant_row("Assets", "2024-12-31", "2025-02-01", 110, accn="2"),
            instant_row("Assets", "2024-09-30", "2025-03-01", 90, accn="3"),
        ]
    )
    assert value_at(panel, "2025-03-01", "assets") == 110


def test_restatement_of_the_prior_year_updates_asset_growth():
    panel = panel_from(
        instant=[
            instant_row("Assets", "2023-12-31", "2024-02-01", 100),
            instant_row("Assets", "2024-12-31", "2025-02-01", 150),
            instant_row("Assets", "2023-12-31", "2025-03-01", 120),
        ]
    )
    assert value_at(panel, "2025-02-01", "asset_growth") == pytest.approx(0.5)
    assert value_at(panel, "2025-03-01", "assets") == 150
    assert value_at(panel, "2025-03-01", "asset_growth") == pytest.approx(0.25)


def test_asset_growth_picks_the_report_closest_to_one_year():
    panel = panel_from(
        instant=[
            instant_row("Assets", "2023-11-26", "2024-02-01", 80),
            instant_row("Assets", "2024-01-05", "2024-03-01", 100),
            instant_row("Assets", "2024-12-31", "2025-02-01", 150),
        ]
    )
    assert value_at(panel, "2025-02-01", "asset_growth") == pytest.approx(0.5)


def test_missing_year_ago_assets_clears_growth_until_the_next_one():
    panel = panel_from(
        instant=[
            instant_row("Assets", "2022-12-31", "2023-02-01", 100),
            instant_row("Assets", "2023-12-31", "2024-02-01", 110),
            instant_row("Assets", "2024-06-30", "2024-08-01", 120),
            instant_row("Assets", "2024-12-31", "2025-02-01", 132),
        ]
    )
    assert value_at(panel, "2024-02-01", "asset_growth") == pytest.approx(0.1)
    assert np.isnan(value_at(panel, "2024-08-01", "asset_growth"))
    assert value_at(panel, "2025-02-01", "asset_growth") == pytest.approx(0.2)


def test_equity_prefers_common_equity_over_the_noncontrolling_total():
    panel = panel_from(
        instant=[
            instant_row(
                "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
                "2024-12-31",
                "2025-02-01",
                80,
            ),
            instant_row("StockholdersEquity", "2024-12-31", "2025-02-01", 70),
        ]
    )
    assert value_at(panel, "2025-02-01", "equity") == 70


def test_shares_follow_their_own_filing_and_carry_across_later_facts():
    panel = panel_from(
        instant=[
            instant_row(
                "EntityCommonStockSharesOutstanding",
                "2024-12-31",
                "2025-02-01",
                4,
                taxonomy="dei",
                unit="shares",
            ),
            instant_row("Assets", "2024-12-31", "2025-05-01", 10),
        ]
    )
    assert value_at(panel, "2025-05-01", "shares") == 4
    assert value_at(panel, "2025-05-01", "assets") == 10


def test_dual_share_classes_each_receive_the_company_fact():
    panel = panel_from(
        instant=[instant_row("Assets", "2024-12-31", "2025-02-01", 10)],
        pairs=(("GOOG", "1"), ("GOOGL", "1")),
    )
    assert set(panel["ticker"]) == {"GOOG", "GOOGL"}
    assert panel["assets"].tolist() == [10, 10]


def test_revenue_alias_and_year_to_date_span_are_not_mixed_into_the_quarter():
    panel = panel_from(
        period=[
            period_row("SalesRevenueNet", "2024-01-01", "2024-03-31", "2024-05-01", 99),
            period_row("Revenues", "2024-01-01", "2024-03-31", "2024-05-01", 10),
            period_row("Revenues", "2024-01-01", "2024-06-30", "2024-08-01", 999),
        ]
    )
    assert value_at(panel, "2024-05-01", "revenue_q") == 10
    assert panel["filed"].max() == pd.Timestamp("2024-05-01")


def test_fourth_quarter_is_annual_minus_the_first_three_quarters():
    panel = panel_from(
        period=[
            period_row("NetIncomeLoss", "2024-01-01", "2024-03-31", "2024-05-01", 1),
            period_row("NetIncomeLoss", "2024-04-01", "2024-06-30", "2024-08-01", 2),
            period_row("NetIncomeLoss", "2024-07-01", "2024-09-30", "2024-11-01", 3),
            period_row("NetIncomeLoss", "2024-01-01", "2024-12-31", "2025-02-15", 10),
        ]
    )
    assert np.isnan(value_at(panel, "2024-11-01", "net_income_ttm"))
    assert value_at(panel, "2025-02-15", "net_income_q") == 4
    assert value_at(panel, "2025-02-15", "net_income_ttm") == 10


def test_standalone_fourth_quarter_replaces_the_derived_one():
    panel = panel_from(
        period=[
            period_row("NetIncomeLoss", "2024-01-01", "2024-03-31", "2024-05-01", 1),
            period_row("NetIncomeLoss", "2024-04-01", "2024-06-30", "2024-08-01", 2),
            period_row("NetIncomeLoss", "2024-07-01", "2024-09-30", "2024-11-01", 3),
            period_row("NetIncomeLoss", "2024-01-01", "2024-12-31", "2025-02-15", 10),
            period_row("NetIncomeLoss", "2024-10-01", "2024-12-31", "2025-02-15", 3.5),
        ]
    )
    assert value_at(panel, "2025-02-15", "net_income_q") == 3.5
    assert value_at(panel, "2025-02-15", "net_income_ttm") == 9.5


def test_restatement_updates_ttm_and_prefers_net_income_over_profit_loss():
    panel = panel_from(
        period=[
            period_row("ProfitLoss", "2024-01-01", "2024-03-31", "2024-05-01", 9),
            period_row("NetIncomeLoss", "2024-01-01", "2024-03-31", "2024-05-01", 1),
            period_row("NetIncomeLoss", "2024-04-01", "2024-06-30", "2024-08-01", 2),
            period_row("NetIncomeLoss", "2024-07-01", "2024-09-30", "2024-11-01", 3),
            period_row("NetIncomeLoss", "2024-10-01", "2024-12-31", "2025-02-15", 4),
            period_row("NetIncomeLoss", "2024-04-01", "2024-06-30", "2025-03-01", 5),
        ]
    )
    assert value_at(panel, "2025-02-15", "net_income_ttm") == 10
    assert value_at(panel, "2025-03-01", "net_income_ttm") == 13


def test_other_filing_does_not_erase_a_carried_level():
    panel = panel_from(
        instant=[instant_row("Assets", "2024-12-31", "2025-02-01", 10)],
        period=[
            period_row("Revenues", "2025-01-01", "2025-03-31", "2025-05-01", 3),
        ],
    )
    assert value_at(panel, "2025-05-01", "assets") == 10
    assert value_at(panel, "2025-05-01", "revenue_q") == 3


def test_filing_panel_does_not_mutate_inputs_and_empty_inputs_keep_columns():
    instant = pd.DataFrame(columns=INSTANT_COLUMNS)
    period = pd.DataFrame(columns=PERIOD_COLUMNS)
    universe = members(("A", "1"))
    before = [frame.copy() for frame in (instant, period, universe)]
    panel = filing_panel(instant, period, universe)
    for frame, original in zip((instant, period, universe), before, strict=True):
        pd.testing.assert_frame_equal(frame, original)
    assert panel.columns.tolist() == [
        "ticker",
        "filed",
        "assets",
        "equity",
        "shares",
        "revenue_q",
        "net_income_q",
        "net_income_ttm",
        "asset_growth",
    ]
    assert panel.empty


def test_read_filing_panel_maps_database_columns():
    with duckdb.connect() as connection:
        connection.execute(
            """
            CREATE TABLE instant_facts(
                cik VARCHAR, taxonomy VARCHAR, concept VARCHAR, unit VARCHAR,
                period_end DATE, filed DATE, val DOUBLE, accn VARCHAR
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE period_facts(
                cik VARCHAR, taxonomy VARCHAR, concept VARCHAR, unit VARCHAR,
                period_start DATE, period_end DATE, filed DATE, val DOUBLE,
                accn VARCHAR
            )
            """
        )
        connection.execute(
            """
            INSERT INTO instant_facts VALUES
            ('1', 'us-gaap', 'Assets', 'USD', '2023-12-31', '2025-02-01', 8, 'a'),
            ('1', 'us-gaap', 'Assets', 'USD', '2024-12-31', '2025-02-01', 10, 'b')
            """
        )
        panel = read_filing_panel(connection, members(("A", "1")))
    assert value_at(panel, "2025-02-01", "assets") == 10


def test_align_filings_only_uses_a_filing_after_its_date():
    prices = prices_for(("2024-01-02", "A", 5.0), ("2024-01-03", "A", 10.0))
    filings = pd.DataFrame(
        {
            "ticker": ["A", "A"],
            "filed": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "equity": [50.0, 75.0],
        }
    )
    aligned = align_filings(prices, filings)
    np.testing.assert_allclose(aligned["equity"], [np.nan, 50.0])


def test_align_filings_on_empty_panel_keeps_the_price_index():
    prices = prices_for(("2024-01-02", "A", 5.0))
    filings = panel_from()
    aligned = align_filings(prices, filings)
    assert aligned.index.equals(prices.sort_index().index)
    assert np.isnan(aligned["equity"].iloc[0])


def test_fundamental_docstrings():
    result = doctest.testmod(fundamentals_module)
    assert result.failed == 0
    assert result.attempted > 0


def test_normalized_facts_cover_both_taxonomies_and_keep_filing_context():
    mapping = fundamental_fact_mapping()
    assert not mapping.duplicated(["fact_kind", "taxonomy", "concept"]).any()

    with duckdb.connect() as connection:
        connection.execute("""
            CREATE TABLE instant_facts (
                cik VARCHAR, taxonomy VARCHAR, concept VARCHAR, unit VARCHAR,
                period_end DATE, filed DATE, accn VARCHAR, form VARCHAR, val DOUBLE
            );
            CREATE TABLE period_facts AS
                SELECT *, NULL::DATE AS period_start FROM instant_facts;
            INSERT INTO instant_facts VALUES
                ('0000849395', 'ifrs-full', 'Equity', 'EUR', '2019-12-31', '2020-03-06', 'a', '20-F', 110),
                ('0000849395', 'ifrs-full', 'EquityAttributableToOwnersOfParent', 'EUR',
                 '2019-12-31', '2020-03-06', 'a', '20-F', 100),
                ('0000849395', 'us-gaap', 'StockholdersEquity', 'USD',
                 '2023-12-31', '2024-02-29', 'c', '10-K', 150),
                ('0000000001', 'us-gaap', 'Assets', 'USD', '2023-12-31', '2024-02-29', 'd', '10-K', 200);
            INSERT INTO period_facts VALUES
                ('0000849395', 'ifrs-full', 'Revenue', 'EUR', '2019-12-31', '2020-03-06', 'a', '20-F', 500, '2019-01-01'),
                ('0000849395', 'ifrs-full', 'RevenueFromContractsWithCustomers', 'EUR',
                 '2019-12-31', '2020-03-06', 'a', '20-F', 450, '2019-01-01'),
                ('0000849395', 'ifrs-full', 'Revenue', 'USD', '2019-12-31', '2021-03-12', 'b', '20-F', 610, '2019-01-01'),
                ('0000849395', 'us-gaap', 'RevenueFromContractWithCustomerExcludingAssessedTax', 'USD',
                 '2023-12-31', '2024-02-29', 'c', '10-K', 700, '2023-01-01'),
                ('0000849395', 'ifrs-full', 'ProfitLossBeforeTax', 'EUR',
                 '2019-12-31', '2020-03-06', 'a', '20-F', 60, '2019-01-01');
        """)
        facts = read_fundamental_facts(connection, ["0000849395"])
        assert read_fundamental_facts(connection, []).empty

    assert set(facts["cik"]) == {"0000849395"}
    assert set(facts["metric"]) == {"equity", "total_equity", "revenue"}
    revenue = facts.loc[facts["metric"] == "revenue"]
    assert list(
        revenue[["accn", "unit", "taxonomy", "val"]].itertuples(index=False, name=None)
    ) == [
        ("a", "EUR", "ifrs-full", 500),
        ("b", "USD", "ifrs-full", 610),
        ("c", "USD", "us-gaap", 700),
    ]
    assert facts.loc[facts["fact_kind"] == "instant", "period_start"].isna().all()
