"""在五日标签上比较直接用组合分数和交给 LambdaRank 排序的结果

先计算 logvol_21, logvol_63, logvol_126, logvol_252 和 b_cma
四个波动率窗口覆盖短中长区间, b_cma 是含截距的 126 日 FF5 回归系数
各列按日做截面 z-score, b_cma 取负方向, 再用 combine_scores 等权合成

此前验证集中, 直接用组合分数与交给 LambdaRank 的表现接近, 都比拆成原始特征好
这只是本次比较的结果, 选定的 63 日标签组合见 smoke_quarterly.py
标签由 future_returns(n_periods=5, gap=1) 转为 31 档相关度
训练到 2022-12-31, 验证到 2024-12-31, embargo 为 6
long_short_5d 是每天五分位多空的 5 日收益, 持有期彼此重叠, 不能直接复利成净值
"""

from __future__ import annotations

import pandas as pd

from quanttoolkit.db.connection import DB, get_duckdb
from quanttoolkit.factors.combine import combine_scores
from quanttoolkit.factors.exposure import cma_exposure
from quanttoolkit.factors.logvol_base import base_logvolfactor
from quanttoolkit.factors.zscore import zscore_by_date
from quanttoolkit.metrics import ic_summary, rank_ic, win_rate
from quanttoolkit.model.dataset import assemble_samples, label_end_dates
from quanttoolkit.model.lambdarank import (
    LambdaRankParams,
    fit_lambdarank,
    predict_scores,
)
from quanttoolkit.model.relevance import relevance_by_date
from quanttoolkit.model.split import split_samples
from quanttoolkit.panel.returns import future_returns
from quanttoolkit.panel.transfer import to_date_ticker_frame
from quanttoolkit.portfolio.sizing import simple_quantile_long_short

WINDOWS = (21, 63, 126, 252)
EXPOSURE = 126
WEIGHTS = {f"logvol_{window}": 1.0 for window in WINDOWS} | {"b_cma": -1.0}


def main() -> None:
    with get_duckdb(path=DB) as connection:
        prices = connection.execute(
            "SELECT date, ticker, adj_close, volume FROM prices"
        ).df()
        ff = connection.execute(
            """
            SELECT date, mkt_rf, smb, hml, rmw, cma, rf
            FROM ff_factors
            WHERE model = 'ff5'
            """
        ).df()

    prices = to_date_ticker_frame(df=prices)
    ff["date"] = pd.to_datetime(ff["date"])
    ff = ff.set_index("date").sort_index()
    factors = pd.DataFrame(
        {f"logvol_{window}": base_logvolfactor(prices, window) for window in WINDOWS}
    )
    factors["b_cma"] = cma_exposure(prices, ff, EXPOSURE)
    score = combine_scores(zscore_by_date(factors), WEIGHTS)
    samples = assemble_samples(
        features=score.to_frame(),
        label=future_returns(df=prices, n_periods=5, gap=1),
    )
    parts = split_samples(
        samples=samples,
        dates=prices.index.get_level_values("date").unique(),
        label_end=label_end_dates(prices.index, n_periods=5, gap=1),
        train_end="2022-12-31",
        valid_end="2024-12-31",
        embargo=6,
    )
    forward = samples.iloc[:, -1]

    def report(name: str, signal: pd.Series) -> None:
        summary = ic_summary(ic=rank_ic(factor=signal, forward_return=forward))
        weights = simple_quantile_long_short(score=signal, n_quantiles=5)
        spread = (weights * forward.reindex(weights.index)).groupby(level="date").sum()
        print(
            f"{name} rank_ic"
            f" mean {summary['ic_mean']:.6f}"
            f" std {summary['ic_std']:.6f}"
            f" ir {summary['ic_ir']:.6f}"
            f" pos {summary['ic_pos']:.6f}"
        )
        print(
            f"{name} long_short_5d"
            f" mean {float(spread.mean()):.6f}"
            f" win {win_rate(returns=spread):.6f}"
        )

    combined = score.reindex(parts["valid"].index)
    train_x = parts["train"].iloc[:, :-1]
    train_y = relevance_by_date(parts["train"].iloc[:, -1])
    valid_x = parts["valid"].iloc[:, :-1]
    booster = fit_lambdarank(
        features=train_x,
        relevance=train_y,
        params=LambdaRankParams(),
    )
    pred = predict_scores(booster=booster, features=valid_x)

    print(f"train {len(train_x)} valid {len(valid_x)}")
    report("score", combined)
    report("model", pred)


if __name__ == "__main__":
    main()
