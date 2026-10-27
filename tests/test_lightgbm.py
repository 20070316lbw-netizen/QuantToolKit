import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from quanttoolkit.model.lambdarank import fit_lambdarank, predict_scores
from quanttoolkit.model.lightgbm import LightGBMParams, fit_lightgbm


@pytest.fixture
def regression_data():
    dates = pd.bdate_range("2024-01-01", periods=20, name="date")
    index = pd.MultiIndex.from_product([dates, ["A", "B", "C", "D"]])
    index.names = ["date", "ticker"]
    rng = np.random.default_rng(7)
    features = pd.DataFrame(
        rng.normal(size=(len(index), 2)), index=index, columns=["a", "b"]
    )
    label = (features["a"] * 0.1 + features["b"] * 0.2).rename("target")
    return features, label


def test_regression_learns_aligned_labels_and_predicts_without_future_labels(
    regression_data,
):
    features, label = regression_data
    original = features.copy()
    params = LightGBMParams(num_boost_round=25, min_data_in_leaf=2, num_leaves=7)
    booster = fit_lightgbm(
        features=features, label=label.sample(frac=1, random_state=9), params=params
    )
    pred = predict_scores(booster=booster, features=features)
    assert np.mean((pred - label) ** 2) < np.mean((label - label.mean()) ** 2) / 2
    assert pred.name == "pred"
    pd.testing.assert_index_equal(pred.index, features.index)
    pd.testing.assert_frame_equal(features, original)
    assert booster.best_iteration == 0
    with pytest.raises(lgb.basic.LightGBMError):
        predict_scores(booster=booster, features=features[["b", "a"]])
    with pytest.raises(lgb.basic.LightGBMError):
        predict_scores(
            booster=booster, features=features.rename(columns={"a": "wrong"})
        )


def test_regression_early_stopping_and_model_roundtrip(regression_data, tmp_path):
    features, label = regression_data
    train_x, valid_x = features.iloc[:40], features.iloc[40:]
    params = LightGBMParams(
        num_boost_round=20, early_stopping_rounds=2, min_data_in_leaf=2, num_leaves=7
    )
    booster = fit_lightgbm(
        features=train_x,
        label=label.iloc[:40],
        valid_features=valid_x,
        valid_label=-label.iloc[40:],
        params=params,
    )
    assert booster.best_iteration == 1
    valid_mse = np.mean((booster.predict(valid_x) + label.iloc[40:]) ** 2)
    assert booster.best_score["valid"]["l2"] == pytest.approx(valid_mse)
    unselected = fit_lightgbm(
        features=train_x,
        label=label.iloc[:40],
        params=LightGBMParams(num_boost_round=20, min_data_in_leaf=2, num_leaves=7),
    )
    assert unselected.num_trees() == 20
    assert valid_mse < np.mean((unselected.predict(valid_x) + label.iloc[40:]) ** 2)
    path = tmp_path / "model.txt"
    booster.save_model(path)
    loaded = lgb.Booster(model_file=path)
    np.testing.assert_allclose(loaded.predict(valid_x), booster.predict(valid_x))


def test_regression_validation_without_early_stopping(regression_data):
    features, label = regression_data
    booster = fit_lightgbm(
        features=features,
        label=label,
        valid_features=features,
        valid_label=label,
        params=LightGBMParams(
            num_boost_round=3, min_data_in_leaf=2, early_stopping_rounds=0
        ),
    )
    assert booster.current_iteration() == 3
    assert "valid" in booster.best_score


def test_regression_rejects_missing_labels_or_incomplete_validation(regression_data):
    features, label = regression_data
    for kwargs in (
        {"valid_features": features},
        {"valid_label": label},
        {"valid_features": features[["b", "a"]], "valid_label": label},
    ):
        with pytest.raises(ValueError):
            fit_lightgbm(features=features, label=label, **kwargs)
    for bad in (label.iloc[1:], label * np.inf):
        with pytest.raises(ValueError, match="有限数值"):
            fit_lightgbm(features=features, label=bad)
    with pytest.raises(ValueError):
        fit_lightgbm(features=features.iloc[:0], label=label)


@pytest.mark.parametrize("value", [np.nan, np.inf, -1, 31, 0.5])
def test_ranking_rejects_invalid_grades(regression_data, value):
    features, label = regression_data
    relevance = pd.Series(1.0, index=label.index)
    relevance.iloc[0] = value
    with pytest.raises(ValueError, match="0 到 30"):
        fit_lambdarank(features=features, relevance=relevance)
