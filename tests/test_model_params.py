import json

from quanttoolkit.model.lambdarank import LambdaRankParams
from quanttoolkit.model.lightgbm import LightGBMParams
from quanttoolkit.model.params import ModelParams, TrainLog


def test_booster_params_keep_tree_settings_and_copy_seed():
    params = LightGBMParams(num_leaves=15, seed=3)
    assert params.booster_params() == {
        "seed": 3,
        "objective": "regression",
        "metric": "l2",
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_data_in_leaf": 50,
        "feature_fraction": 1.0,
        "verbosity": -1,
        "feature_fraction_seed": 3,
        "bagging_seed": 3,
        "data_random_seed": 3,
    }
    assert params.num_boost_round == 200
    assert params.early_stopping_rounds == 20


def test_to_dict_is_a_copy():
    params = ModelParams(seed=1)
    copied = params.to_dict()
    copied["seed"] = 9
    assert params.seed == 1


def test_model_params_roundtrip(tmp_path):
    path = tmp_path / "lgb.json"
    LightGBMParams(num_leaves=15, seed=3).save(path)
    loaded = LightGBMParams.load(path)
    assert loaded == LightGBMParams(num_leaves=15, seed=3)
    assert json.loads(path.read_text())["num_leaves"] == 15


def test_lambdarank_uses_library_defaults_except_objective_and_seed(tmp_path):
    params = LambdaRankParams(seed=3)
    assert params.num_boost_round == 100
    assert params.booster_params() == {
        "objective": "lambdarank",
        "seed": 3,
        "feature_fraction_seed": 3,
        "bagging_seed": 3,
        "data_random_seed": 3,
    }
    path = tmp_path / "lambdarank.json"
    params.save(path)
    assert LambdaRankParams.load(path) == params


def test_train_log_roundtrip(tmp_path):
    path = tmp_path / "run.json"
    log = TrainLog(
        model="lightgbm",
        params=LightGBMParams(num_leaves=15).to_dict(),
        metrics={"best_iteration": 40},
    )
    log.save(path)
    assert TrainLog.load(path) == log
