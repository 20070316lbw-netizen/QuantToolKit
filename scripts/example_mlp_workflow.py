"""用合成行情检查 MLP 排序训练, 评估和模型重载

直接运行本文件, 结果写到新的 logs/mlp_example/run_* 子目录
合成行情没有设定可预测规律, 指标只检查流程, 不证明模型或因子有效
"""

import json
from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pandas as pd

from quanttoolkit.factors.mom_base import base_momfactor
from quanttoolkit.metrics import ic_summary, rank_ic
from quanttoolkit.model.dataset import assemble_samples, label_end_dates
from quanttoolkit.model.mlp import MLPParams, MLPRanker, fit_mlp_ranker
from quanttoolkit.model.params import TrainLog
from quanttoolkit.model.split import split_samples
from quanttoolkit.panel.returns import future_returns
from quanttoolkit.panel.transfer import prepare_price_panel


def main(output_dir: Path | None = None) -> Path:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2023-01-02", periods=120, name="date")
    tickers = [f"S{i:02}" for i in range(12)]
    index = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    returns = rng.normal(0.0003, 0.015, (len(dates), len(tickers)))
    prices = prepare_price_panel(
        pd.DataFrame(
            {"adj_close": (100 * np.exp(returns.cumsum(axis=0))).ravel()}, index=index
        )
    )
    features = pd.concat(
        [base_momfactor(prices, window=5), base_momfactor(prices, window=10)], axis=1
    )
    horizon, gap = 3, 1
    label = future_returns(df=prices, n_periods=horizon, gap=gap)
    parts = split_samples(
        samples=assemble_samples(features=features, label=label),
        dates=dates,
        label_end=label_end_dates(prices.index, n_periods=horizon, gap=gap),
        train_end=dates[69],
        valid_end=dates[94],
        embargo=horizon + gap,
    )
    columns = features.columns.tolist()
    train, valid = parts["train"], parts["valid"]
    params = MLPParams(
        hidden_dims=[16, 8], max_epochs=12, early_stopping_rounds=3, learning_rate=0.003
    )
    model = fit_mlp_ranker(
        features=train[columns],
        label=train[label.name],
        valid_features=valid[columns],
        valid_label=valid[label.name],
        params=params,
    )
    metrics = {
        "best_epoch": model.best_epoch,
        "best_valid_loss": model.best_valid_loss,
        "training_summary": model.training_summary,
        "data": {
            "source": "synthetic_prices",
            "seed": 7,
            "features": columns,
            "label": label.name,
            "n_periods": horizon,
            "gap": gap,
            "train_end": str(dates[69].date()),
            "valid_end": str(dates[94].date()),
            "embargo": horizon + gap,
        },
    }
    for role in ("valid", "test"):
        part = parts[role]
        scores = model.predict(part[columns])
        summary = ic_summary(ic=rank_ic(factor=scores, forward_return=part[label.name]))
        metrics[role] = {
            key: float(value) if np.isfinite(value) else None
            for key, value in summary.items()
        }
        metrics[role]["n_samples"] = len(part)
        metrics[role]["n_dates"] = part.index.get_level_values("date").nunique()
    if output_dir is None:
        parent = Path(__file__).resolve().parents[1] / "logs" / "mlp_example"
        parent.mkdir(parents=True, exist_ok=True)
        output_dir = Path(mkdtemp(prefix="run_", dir=parent))
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
    model.save(output_dir / "model.pt")
    TrainLog(model="mlp_ranker", params=model.params.to_dict(), metrics=metrics).save(
        output_dir / "run.json"
    )
    (output_dir / "history.json").write_text(
        json.dumps(model.history, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    loaded = MLPRanker.load(output_dir / "model.pt")
    test_features = parts["test"][columns]
    pd.testing.assert_series_equal(
        model.predict(test_features), loaded.predict(test_features)
    )
    print(f"best_epoch: {model.best_epoch}, best_valid_loss: {model.best_valid_loss}")
    print(f"valid: {metrics['valid']}")
    print(f"test: {metrics['test']}")
    print(f"saved: {output_dir}")
    return output_dir


if __name__ == "__main__":
    main()
