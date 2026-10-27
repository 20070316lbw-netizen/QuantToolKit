import pandas as pd
import pytest

from quanttoolkit.model.dataset import assemble_samples, label_end_dates


def _index():
    return pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-03", "2024-01-02"]), ["B", "A"]],
        names=["date", "ticker"],
    )


def test_assemble_keeps_complete_rows_and_puts_label_last():
    index = _index()
    features = pd.DataFrame(
        {"mom_2": [0.4, 0.2, 0.3, float("nan")], "rsi_2": [1.0, 2.0, 3.0, 4.0]},
        index=index,
    )
    label = pd.Series(
        [0.04, float("nan"), 0.01, 0.02],
        index=index,
        name="forward_return_1_gap_1",
    )
    original = features.copy()
    samples = assemble_samples(features=features, label=label)
    expected_index = pd.MultiIndex.from_tuples(
        [
            (pd.Timestamp("2024-01-02"), "B"),
            (pd.Timestamp("2024-01-03"), "B"),
        ],
        names=["date", "ticker"],
    )
    expected = pd.DataFrame(
        {
            "mom_2": [0.3, 0.4],
            "rsi_2": [3.0, 1.0],
            "forward_return_1_gap_1": [0.01, 0.04],
        },
        index=expected_index,
    )
    pd.testing.assert_frame_equal(samples, expected)
    pd.testing.assert_frame_equal(features, original)


def test_assemble_drops_keys_missing_from_either_side():
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-02"), "A"), (pd.Timestamp("2024-01-02"), "B")],
        names=["date", "ticker"],
    )
    features = pd.DataFrame({"mom_2": [0.1, 0.2]}, index=index)
    label = pd.Series([0.01], index=index[:1], name="forward_return_1_gap_1")
    samples = assemble_samples(features=features, label=label)
    assert samples.index.tolist() == [(pd.Timestamp("2024-01-02"), "A")]


def test_empty_result_keeps_columns_and_index_names():
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-02"), "A")],
        names=["date", "ticker"],
    )
    features = pd.DataFrame({"mom_2": [float("nan")]}, index=index)
    label = pd.Series([0.01], index=index, name="forward_return_1_gap_1")
    samples = assemble_samples(features=features, label=label)
    assert samples.empty
    assert samples.columns.tolist() == ["mom_2", "forward_return_1_gap_1"]
    assert samples.index.names == ["date", "ticker"]


def test_label_must_have_a_name():
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-02"), "A")],
        names=["date", "ticker"],
    )
    features = pd.DataFrame({"mom_2": [0.1]}, index=index)
    label = pd.Series([0.01], index=index)
    with pytest.raises(ValueError, match="label 必须带 name 作为标签列名"):
        assemble_samples(features=features, label=label)


def test_index_names_must_match():
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-02"), "A")],
        names=["date", "ticker"],
    )
    features = pd.DataFrame({"mom_2": [0.1]}, index=index)
    label = pd.Series(
        [0.01],
        index=index.rename(["date", "symbol"]),
        name="forward_return_1_gap_1",
    )
    with pytest.raises(ValueError, match="索引名不一致"):
        assemble_samples(features=features, label=label)


def test_duplicate_date_ticker_is_rejected():
    duplicated = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-02"), "A")] * 2,
        names=["date", "ticker"],
    )
    unique = duplicated[:1]
    with pytest.raises(ValueError, match="索引必须唯一"):
        assemble_samples(
            features=pd.DataFrame({"mom_2": [0.1, 0.2]}, index=duplicated),
            label=pd.Series([0.01], index=unique, name="forward_return_1_gap_1"),
        )
    with pytest.raises(ValueError, match="索引必须唯一"):
        assemble_samples(
            features=pd.DataFrame({"mom_2": [0.1]}, index=unique),
            label=pd.Series(
                [0.01, 0.02], index=duplicated, name="forward_return_1_gap_1"
            ),
        )


def test_label_column_name_cannot_duplicate_a_feature():
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-02"), "A")],
        names=["date", "ticker"],
    )
    features = pd.DataFrame({"mom_2": [0.1]}, index=index)
    label = pd.Series([0.01], index=index, name="mom_2")
    with pytest.raises(KeyError, match="mom_2"):
        assemble_samples(features=features, label=label)


def test_label_end_tracks_each_security_record_and_preserves_timezone(price_panel):
    ends = label_end_dates(price_panel.index, n_periods=1, gap=1)
    assert ends.loc[("2024-01-02", "A")] == pd.Timestamp("2024-01-05")
    assert ends.loc[("2024-01-02", "B")] == pd.Timestamp("2024-01-09")
    assert ends.loc[("2024-01-05", "B")] is pd.NaT
    assert ends.index.equals(price_panel.sort_index().index)
    dates = pd.bdate_range("2024-01-01", periods=3, tz="Asia/Shanghai")
    index = pd.MultiIndex.from_product([dates, ["A"]], names=["date", "ticker"])
    assert label_end_dates(index, n_periods=1, gap=0).iloc[0] == dates[1]
    assert label_end_dates(index[:0]).empty
