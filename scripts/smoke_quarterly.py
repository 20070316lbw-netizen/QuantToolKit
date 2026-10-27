"""用选定的 logvol_63 和 b_cma 训练全市场 LambdaRank 排序模型

b_cma 来自过去 63 条记录的股票超额收益对同期 FF5 加截距回归
它是五因子一起回归得到的 CMA 系数, 不等于只对 CMA 做单变量回归
成交量为 0 的补齐日不进入有效回归窗口

此前对照后只保留这两列, 不再加入当天 FF5, 其余贝塔, mom_63,
rel_volume_63, 以及五日对照里的 logvol_21 和 rel_volume_21
标签是 future_returns(n_periods=63, gap=1), embargo 为 64
训练到 2022-12-31, 验证到 2024-12-31

long_short_63d 是每天五分位多空的 63 日收益, 持有期彼此重叠, 不能直接复利成净值
rank_ic_q 则每隔 63 个交易日取一个验证集 Rank IC, 再做汇总
"""

from __future__ import annotations

import pandas as pd

from quanttoolkit.db.connection import DB, get_duckdb
from quanttoolkit.factors.exposure import cma_exposure
from quanttoolkit.factors.logvol_base import base_logvolfactor
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

QUARTER = 63
EMBARGO = QUARTER + 1
KEPT = ["logvol_63", "b_cma"]


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
    samples = assemble_samples(
        features=pd.concat(
            [base_logvolfactor(prices, QUARTER), cma_exposure(prices, ff, QUARTER)],
            axis=1,
        )[KEPT],
        label=future_returns(df=prices, n_periods=QUARTER, gap=1),
    )
    parts = split_samples(
        samples=samples,
        dates=prices.index.get_level_values("date").unique(),
        label_end=label_end_dates(prices.index, n_periods=QUARTER, gap=1),
        train_end="2022-12-31",
        valid_end="2024-12-31",
        embargo=EMBARGO,
    )
    forward = samples.iloc[:, -1]

    train_x = parts["train"].iloc[:, :-1]
    train_y = relevance_by_date(parts["train"].iloc[:, -1])
    valid_x = parts["valid"].iloc[:, :-1]
    booster = fit_lambdarank(
        features=train_x,
        relevance=train_y,
        params=LambdaRankParams(),
    )
    pred = predict_scores(booster=booster, features=valid_x)
    score = rank_ic(factor=pred, forward_return=forward).sort_index()
    summary = ic_summary(ic=score)
    spaced = ic_summary(ic=score.iloc[::QUARTER])
    weights = simple_quantile_long_short(score=pred, n_quantiles=5)
    spread = (weights * forward.reindex(weights.index)).groupby(level="date").sum()
    gain = pd.Series(
        booster.feature_importance(importance_type="gain"),
        index=booster.feature_name(),
    ).sort_values(ascending=False)

    print(f"train {len(train_x)} valid {len(valid_x)}")
    print(
        "rank_ic"
        f" mean {summary['ic_mean']:.6f}"
        f" std {summary['ic_std']:.6f}"
        f" ir {summary['ic_ir']:.6f}"
        f" pos {summary['ic_pos']:.6f}"
    )
    print(
        "rank_ic_q"
        f" mean {spaced['ic_mean']:.6f}"
        f" ir {spaced['ic_ir']:.6f}"
        f" n {len(score.iloc[::QUARTER])}"
    )
    print(
        "long_short_63d"
        f" mean {float(spread.mean()):.6f}"
        f" win {win_rate(returns=spread):.6f}"
    )
    print("gain")
    print(gain.to_string(float_format=lambda value: f"{value:.6f}"))


if __name__ == "__main__":
    main()
