import doctest
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from quanttoolkit.metrics import rank_ic
from quanttoolkit.model.loss import pairwise_rank_loss
from quanttoolkit.model.mlp import (
    MLPParams,
    MLPRanker,
    fit_mlp_ranker,
    model,
    network,
    params,
    train,
)
from quanttoolkit.model.params import TrainLog
from quanttoolkit.portfolio.sizing import simple_quantile_long_short


@pytest.fixture
def ranking_panel():
    index = pd.MultiIndex.from_product(
        [pd.bdate_range("2024-01-01", periods=4), ["A", "B", "C", "D"]],
        names=["date", "ticker"],
    )
    signal = np.tile([-1.5, -0.5, 0.5, 1.5], 4)
    features = pd.DataFrame({"signal": signal, "constant": 2.0}, index=index)
    label = pd.Series(signal * 0.01, index=index, name="forward_return_1_gap_0")
    return features, label


def _params(**overrides):
    settings = {
        "hidden_dims": [],
        "dropout": 0.0,
        "learning_rate": 0.1,
        "weight_decay": 0.0,
        "max_epochs": 12,
        "early_stopping_rounds": 2,
    }
    settings.update(overrides)
    return MLPParams(**settings)


def test_params_roundtrip_and_independent_hidden_dims(tmp_path):
    config = MLPParams(hidden_dims=[8, 4], seed=3)
    path = tmp_path / "params.json"
    config.save(path)
    assert MLPParams.load(path) == config
    assert json.loads(path.read_text())["hidden_dims"] == [8, 4]
    copied = config.to_dict()
    copied["hidden_dims"].append(2)
    assert config.hidden_dims == [8, 4]
    first = MLPParams()
    first.hidden_dims.append(1)
    assert MLPParams().hidden_dims == [64, 32]


def test_network_scores_have_no_bounded_activation():
    net = network.MLP(input_dim=1, hidden_dims=[], dropout=0.0)
    with torch.no_grad():
        net.layers[0].weight.fill_(2.0)
        net.layers[0].bias.zero_()
    assert net(torch.tensor([[-2.0], [2.0]])).tolist() == [-4.0, 4.0]
    assert net(torch.ones(1, 1)).shape == (1,)
    assert net(torch.empty(0, 1)).shape == (0,)


def test_training_learns_order_and_connects_to_ic_and_positions(ranking_panel):
    features, label = ranking_panel
    before = features.copy(deep=True)
    result = fit_mlp_ranker(features=features, label=label, params=_params())
    scores = result.predict(features)
    assert result.history[-1]["train_loss"] < result.history[0]["train_loss"] * 0.2
    assert rank_ic(factor=scores, forward_return=label).eq(1.0).all()
    weights = simple_quantile_long_short(score=scores, n_quantiles=2)
    assert weights.groupby(level="date").sum().abs().lt(1e-12).all()
    assert result.best_epoch is None and result.best_valid_loss is None
    pd.testing.assert_frame_equal(features, before)


def test_index_alignment_prediction_order_and_training_only_scaling(ranking_panel):
    features, label = ranking_panel
    config = _params(max_epochs=3)
    validation = features + 100.0
    result = fit_mlp_ranker(
        features=features.sample(frac=1, random_state=2),
        label=label.sample(frac=1, random_state=3),
        valid_features=validation,
        valid_label=label,
        params=config,
    )
    expected = fit_mlp_ranker(features=features, label=label, params=config)
    np.testing.assert_array_equal(result.feature_mean, [0.0, 2.0])
    np.testing.assert_allclose(result.feature_scale, [np.sqrt(1.25), 1.0])
    # 验证数据不参与梯度更新或缩放拟合, 此处最佳轮为最后一轮
    assert result.best_epoch == config.max_epochs
    pd.testing.assert_series_equal(result.predict(features), expected.predict(features))
    shuffled = features.sample(frac=1, random_state=4)
    scores = result.predict(shuffled)
    assert scores.name == "pred" and scores.index.equals(shuffled.index)
    pd.testing.assert_series_equal(scores.sort_index(), result.predict(features))
    assert result.predict(features.iloc[:0]).empty


def test_dates_are_separate_and_days_have_equal_validation_weight():
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp("2024-01-01"), t) for t in ["A", "B"]]
        + [(pd.Timestamp("2024-01-02"), t) for t in ["A", "B", "C", "D"]],
        names=["date", "ticker"],
    )
    x = pd.DataFrame({"x": [0.0, 3.0, 0.0, 1.0, 2.0, 3.0]}, index=index)
    y = pd.Series([0.0, 1.0, 0.0, 1.0, 2.0, 3.0], index=index)
    fitted = fit_mlp_ranker(
        features=x,
        label=y,
        valid_features=x,
        valid_label=y,
        params=_params(max_epochs=1),
    )
    scores = fitted.predict(x)
    daily = []
    for date in index.get_level_values("date").unique():
        daily.append(
            pairwise_rank_loss(
                torch.tensor(scores.xs(date).to_numpy()),
                torch.tensor(y.xs(date).to_numpy()),
            ).item()
        )
    assert fitted.history[0]["valid_loss"] == pytest.approx(np.mean(daily))
    assert fitted.history[0]["valid_loss"] != pytest.approx(
        (daily[0] + 6 * daily[1]) / 7
    )
    # 每天分别平移标签不改变同日顺序, 也不应影响训练结果
    shifted = y.copy()
    shifted.loc[pd.Timestamp("2024-01-01")] += 100.0
    other = fit_mlp_ranker(features=x, label=shifted, params=_params(max_epochs=1))
    pd.testing.assert_series_equal(fitted.predict(x), other.predict(x))


@pytest.mark.parametrize("patience, epochs", [(2, 3), (0, 8)])
def test_early_stopping_restores_best_weights(ranking_panel, patience, epochs):
    features, label = ranking_panel
    result = fit_mlp_ranker(
        features=features,
        label=label,
        valid_features=features,
        valid_label=-label,
        params=_params(max_epochs=8, early_stopping_rounds=patience),
    )
    first = fit_mlp_ranker(features=features, label=label, params=_params(max_epochs=1))
    assert result.best_epoch == 1
    assert len(result.history) == epochs
    assert result.best_valid_loss == min(row["valid_loss"] for row in result.history)
    pd.testing.assert_series_equal(result.predict(features), first.predict(features))


def test_ineligible_days_are_skipped_and_close_returns_remain_distinct(ranking_panel):
    features, label = ranking_panel
    label = label.copy()
    dates = features.index.get_level_values("date").unique()
    label.loc[dates[0]] = 1.0
    # 两个值转成 float32 后相等, 原始 float64 收益仍有严格顺序
    label.loc[dates[1]] = [1.0, 1.0 + 1e-10, 1.0 + 2e-10, 1.0 + 3e-10]
    keep = np.ones(len(features), dtype=bool)
    keep[9:12] = False
    result = fit_mlp_ranker(
        features=features.loc[keep], label=label, params=_params(max_epochs=1)
    )
    assert result.training_summary == {"train_days": 2, "train_skipped_days": 2}
    assert result.history[0]["train_loss"] > 0


def test_reload_keeps_predictions_scaling_dropout_and_history(ranking_panel, tmp_path):
    features, label = ranking_panel
    config = _params(hidden_dims=[8, 4], dropout=0.3, max_epochs=3)
    fitted = fit_mlp_ranker(
        features=features,
        label=label,
        valid_features=features,
        valid_label=label,
        params=config,
    )
    # 调用方修改配置不会改动已训练模型的结构记录
    config.hidden_dims.append(2)
    assert fitted.params.hidden_dims == [8, 4]
    first = fitted.predict(features)
    pd.testing.assert_series_equal(first, fitted.predict(features))
    path = tmp_path / "model.pt"
    fitted.save(path)
    loaded = MLPRanker.load(path)
    assert not loaded.network.training
    assert next(loaded.network.parameters()).device.type == "cpu"
    pd.testing.assert_series_equal(first, loaded.predict(features))
    assert loaded.params == fitted.params
    assert loaded.history == fitted.history
    assert loaded.best_epoch == fitted.best_epoch
    assert loaded.best_valid_loss == fitted.best_valid_loss
    assert loaded.training_summary == fitted.training_summary
    np.testing.assert_array_equal(loaded.feature_mean, fitted.feature_mean)
    np.testing.assert_array_equal(loaded.feature_scale, fitted.feature_scale)


def test_seed_reproduces_network_dropout_and_date_shuffle(ranking_panel):
    features, label = ranking_panel
    config = _params(hidden_dims=[8], dropout=0.2, max_epochs=3)
    first = fit_mlp_ranker(features=features, label=label, params=config)
    second = fit_mlp_ranker(features=features, label=label, params=config)
    assert first.history == second.history
    pd.testing.assert_series_equal(first.predict(features), second.predict(features))


@pytest.mark.parametrize("invalid", ["missing", "infinite", "unaligned"])
def test_training_rejects_unusable_numbers(ranking_panel, invalid):
    features, label = ranking_panel
    features = features.copy()
    if invalid == "missing":
        features.iloc[0, 0] = np.nan
    elif invalid == "infinite":
        label = label.copy()
        label.iloc[0] = np.inf
    else:
        label = label.iloc[1:]
    with pytest.raises(ValueError, match="有限数值"):
        fit_mlp_ranker(features=features, label=label, params=_params(max_epochs=1))


def test_training_and_prediction_reject_invalid_configuration(ranking_panel):
    features, label = ranking_panel
    for overrides in ({"max_epochs": 0}, {"early_stopping_rounds": -1}):
        with pytest.raises(ValueError, match="max_epochs"):
            fit_mlp_ranker(features=features, label=label, params=_params(**overrides))
    for kwargs in ({"valid_features": features}, {"valid_label": label}):
        with pytest.raises(ValueError, match="同时提供"):
            fit_mlp_ranker(features=features, label=label, **kwargs)
    with pytest.raises(ValueError, match="列名及顺序"):
        fit_mlp_ranker(
            features=features,
            label=label,
            valid_features=features.iloc[:, ::-1],
            valid_label=label,
        )
    for x, y in ((features.iloc[:0], label), (features, label * 0)):
        with pytest.raises(ValueError):
            fit_mlp_ranker(features=x, label=y, params=_params(max_epochs=1))
    with pytest.raises(ValueError, match="没有可比较"):
        fit_mlp_ranker(
            features=features,
            label=label,
            valid_features=features,
            valid_label=label * 0,
        )
    fitted = fit_mlp_ranker(
        features=features, label=label, params=_params(max_epochs=1)
    )
    with pytest.raises(ValueError, match="特征列名和顺序"):
        fitted.predict(features.iloc[:, ::-1])
    with pytest.raises(ValueError, match="有限数值"):
        fitted.predict(features.assign(signal=np.nan))


@pytest.mark.parametrize("module", [params, network, train, model])
def test_mlp_docstrings(module):
    result = doctest.testmod(module)
    assert result.failed == 0 and result.attempted > 0


def test_mlp_workflow_script(tmp_path):
    root = Path(__file__).resolve().parents[1]
    main = runpy.run_path(str(root / "scripts/example_mlp_workflow.py"))["main"]
    output = main(tmp_path / "example")
    loaded = MLPRanker.load(output / "model.pt")
    log = TrainLog.load(output / "run.json")
    assert log.model == "mlp_ranker"
    assert log.params == loaded.params.to_dict()
    assert log.metrics["test"]["n_samples"] > 0
    assert log.metrics["best_epoch"] == loaded.best_epoch
    assert (output / "history.json").exists()
