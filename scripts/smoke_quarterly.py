"""选定的全市场 lambdarank 组合

特征是 logvol_63, 以及过去 63 个交易日超额收益对 FF5 回归得到的 b_cma
b_cma 来自含截距的五因子回归, 不是 cma 的单变量斜率
标签是 future_returns(n_periods=63, gap=1), embargo 为 64
对照里更差的列不再进入特征
当天的 FF5, 其余贝塔, mom_63, rel_volume_63, 以及五日的 logvol_21 和 rel_volume_21
成交量为 0 的补齐日不进入回归
训练到 2022-12-31, 验证到 2024-12-31
long_short_63d 是验证集每天五分位多空拿到的 63 日收益, 各天重叠, 不是复利净值
ic_q 把验证集秩相关每 63 个交易日取一个再汇总
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quanttoolkit.db.connection import DB, get_duckdb
from quanttoolkit.factors.logvol_base import base_logvolfactor
from quanttoolkit.metrics import ic_summary, rank_ic, win_rate
from quanttoolkit.model.dataset import assemble_samples
from quanttoolkit.model.lambdarank import (
    LambdaRankParams,
    fit_lambdarank,
    predict_scores,
)
from quanttoolkit.model.relevance import relevance_by_date
from quanttoolkit.model.split import split_dates
from quanttoolkit.panel.returns import future_returns
from quanttoolkit.panel.transfer import to_date_ticker_frame
from quanttoolkit.portfolio.sizing import simple_quantile_long_short

QUARTER = 63
EMBARGO = QUARTER + 1
FACTORS = ["mkt_rf", "smb", "hml", "rmw", "cma"]
KEPT = ["logvol_63", "b_cma"]


def _rolling_exposure(
    excess: np.ndarray, factors: np.ndarray, window: int
) -> np.ndarray:
    rows, width = factors.shape
    design = np.column_stack([np.ones(rows), factors])
    rank = width + 1
    finite = np.isfinite(excess) & np.isfinite(factors).all(axis=1)
    design = np.where(finite[:, None], design, 0.0)
    target = np.where(finite, excess, 0.0)
    moment = np.cumsum(design[:, :, None] * design[:, None, :], axis=0)
    cross = np.cumsum(design * target[:, None], axis=0)
    square = np.cumsum(target**2, axis=0)
    count = np.cumsum(finite.astype(np.int32), axis=0)
    moment[window:] -= moment[:-window]
    cross[window:] -= cross[:-window]
    square[window:] -= square[:-window]
    count[window:] -= count[:-window]

    ready = count == window
    eye = np.eye(rank)
    moment[~ready] = eye
    cross[~ready] = 0.0
    moment[ready] = moment[ready] + eye * 1e-12
    coef = np.linalg.solve(moment, cross[..., None])[..., 0]
    fitted = np.einsum("ij,ij->i", coef, cross)
    resid = np.sqrt(np.clip(square - fitted, 0, None) / (window - rank))
    out = np.column_stack([coef, resid])
    out[~ready] = np.nan
    return out


def _b_cma(prices: pd.DataFrame, ff: pd.DataFrame) -> pd.Series:
    simple = prices["adj_close"].groupby(level="ticker").pct_change()
    excess = simple - ff["rf"].reindex(simple.index, level="date")
    excess = excess.where(prices["volume"] > 0)
    factors = ff[FACTORS].reindex(excess.index, level="date")
    pieces = []
    for index in excess.groupby(level="ticker").groups.values():
        ordered = excess.loc[index].sort_index()
        values = _rolling_exposure(
            ordered.to_numpy(dtype=float),
            factors.loc[ordered.index, FACTORS].to_numpy(dtype=float),
            QUARTER,
        )
        pieces.append(pd.Series(values[:, 5], index=ordered.index, name="b_cma"))
    return pd.concat(pieces).sort_index()


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
            [base_logvolfactor(prices, QUARTER), _b_cma(prices, ff)],
            axis=1,
        )[KEPT],
        label=future_returns(df=prices, n_periods=QUARTER, gap=1),
    )
    relevance = relevance_by_date(samples.iloc[:, -1])
    features = samples.iloc[:, :-1]
    roles = split_dates(
        features.index.get_level_values("date").unique(),
        train_end="2022-12-31",
        valid_end="2024-12-31",
        embargo=EMBARGO,
    )

    def take(role: str) -> tuple[pd.DataFrame, pd.Series]:
        dates = roles.index[roles == role]
        part = features.loc[features.index.get_level_values("date").isin(dates)]
        return part, relevance.loc[part.index]

    train_x, train_y = take("train")
    valid_x, _valid_y = take("valid")
    booster = fit_lambdarank(
        features=train_x,
        relevance=train_y,
        params=LambdaRankParams(),
    )
    pred = predict_scores(booster=booster, features=valid_x)
    forward = samples.iloc[:, -1]
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
