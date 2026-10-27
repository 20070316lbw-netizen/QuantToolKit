"""用一份合成行情串起样本切分, 回归和排序训练, 评估与模型重载

直接运行本文件, 结果写到新的 logs/model_example/run_* 子目录
合成行情使用固定随机种子, 指标只用来检查各个接口能否接起来
"""

from pathlib import Path
from tempfile import mkdtemp

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from quanttoolkit.factors.mom_base import base_momfactor
from quanttoolkit.metrics import ic_summary, rank_ic
from quanttoolkit.model.dataset import assemble_samples, label_end_dates
from quanttoolkit.model.lambdarank import (
    LambdaRankParams,
    fit_lambdarank,
    predict_scores,
)
from quanttoolkit.model.lightgbm import LightGBMParams, fit_lightgbm
from quanttoolkit.model.params import TrainLog
from quanttoolkit.model.relevance import relevance_by_date
from quanttoolkit.model.split import split_samples
from quanttoolkit.panel.returns import future_returns


def main(output_dir: Path | None = None) -> Path:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2023-01-02", periods=200, name="date")
    tickers = [f"S{i:02}" for i in range(12)]
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    returns = rng.normal(0.0003, 0.015, (len(dates), len(tickers)))
    prices = pd.DataFrame(
        {"adj_close": (100 * np.exp(returns.cumsum(axis=0))).ravel()}, index=index
    )
    features = pd.concat(
        [base_momfactor(prices, window=5), base_momfactor(prices, window=20)], axis=1
    )
    horizon, gap = 5, 1
    label = future_returns(df=prices, n_periods=horizon, gap=gap)
    parts = split_samples(
        samples=assemble_samples(features=features, label=label),
        dates=dates,
        label_end=label_end_dates(prices.index, n_periods=horizon, gap=gap),
        train_end=dates[109],
        valid_end=dates[154],
        embargo=horizon + gap,
    )
    train, valid = parts["train"], parts["valid"]
    columns = features.columns.tolist()
    params = LightGBMParams(
        num_boost_round=30, early_stopping_rounds=5, min_data_in_leaf=10, num_leaves=7
    )
    regression = fit_lightgbm(
        features=train[columns],
        label=train[label.name],
        valid_features=valid[columns],
        valid_label=valid[label.name],
        params=params,
    )
    rank_params = LambdaRankParams(num_boost_round=10)
    ranking = fit_lambdarank(
        features=train[columns],
        relevance=relevance_by_date(train[label.name]),
        params=rank_params,
    )
    if output_dir is None:
        parent = Path(__file__).resolve().parents[1] / "logs" / "model_example"
        parent.mkdir(parents=True, exist_ok=True)
        output_dir = Path(mkdtemp(prefix="run_", dir=parent))
    for task, booster, model_params in (
        ("regression", regression, params),
        ("ranking", ranking, rank_params),
    ):
        reports = {}
        for role in ("valid", "test"):
            part = parts[role]
            pred = predict_scores(booster=booster, features=part[columns])
            summary = ic_summary(
                ic=rank_ic(factor=pred, forward_return=part[label.name])
            )
            reports[role] = {
                key: float(value) if np.isfinite(value) else None
                for key, value in summary.items()
            }
            reports[role]["n_samples"] = len(part)
            if task == "regression":
                reports[role]["mse"] = float(mean_squared_error(part[label.name], pred))
        path = output_dir / task
        path.mkdir(parents=True, exist_ok=False)
        booster.save_model(path / "model.txt")
        TrainLog(model=task, params=model_params.to_dict(), metrics=reports).save(
            path / "run.json"
        )
        loaded = lgb.Booster(model_file=path / "model.txt")
        test_features = parts["test"][columns]
        np.testing.assert_allclose(
            predict_scores(booster=loaded, features=test_features),
            predict_scores(booster=booster, features=test_features),
        )
        print(f"{task}: {reports}")
    print(f"saved: {output_dir}")
    return output_dir


if __name__ == "__main__":
    main()
