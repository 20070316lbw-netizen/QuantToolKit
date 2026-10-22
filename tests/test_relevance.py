import pandas as pd

from quanttoolkit.model.relevance import relevance_by_date


def test_higher_return_gets_higher_grade_within_each_date():
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-03"), "B"),
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-02"), "B"),
            (pd.Timestamp("2024-01-03"), "A"),
        ],
        names=["date", "ticker"],
    )
    label = pd.Series([0.2, 0.1, 0.4, float("nan")], index=index, name="forward")
    original = label.copy()
    relevance = relevance_by_date(label, n_grades=4)
    assert relevance.name == "relevance_4"
    assert relevance.tolist() == [0, 2, pd.NA, 0]
    assert list(relevance.index) == [
        (pd.Timestamp("2024-01-02"), "A"),
        (pd.Timestamp("2024-01-02"), "B"),
        (pd.Timestamp("2024-01-03"), "A"),
        (pd.Timestamp("2024-01-03"), "B"),
    ]
    pd.testing.assert_series_equal(label, original)


def test_full_default_span_stays_inside_label_gain():
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-02"]), [f"S{i}" for i in range(40)]],
        names=["date", "ticker"],
    )
    label = pd.Series(range(40), index=index, dtype="float")
    relevance = relevance_by_date(label)
    assert relevance.name == "relevance_31"
    assert int(relevance.min()) == 0
    assert int(relevance.max()) == 30


def test_ties_follow_row_order_and_blank_dates_stay_missing():
    index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "B"),
            (pd.Timestamp("2024-01-02"), "A"),
            (pd.Timestamp("2024-01-03"), "A"),
            (pd.Timestamp("2024-01-03"), "B"),
        ],
        names=["date", "ticker"],
    )
    label = pd.Series([0.2, 0.2, float("nan"), float("nan")], index=index)
    relevance = relevance_by_date(label, n_grades=2)
    assert relevance.tolist() == [1, 0, pd.NA, pd.NA]


def test_empty_label_keeps_index_names():
    index = pd.MultiIndex.from_arrays([[], []], names=["date", "ticker"])
    relevance = relevance_by_date(pd.Series(index=index, dtype="float"))
    assert relevance.empty
    assert relevance.name == "relevance_31"
    assert relevance.index.names == ["date", "ticker"]
