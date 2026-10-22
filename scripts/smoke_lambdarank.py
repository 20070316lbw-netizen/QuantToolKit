"""五日标签上的对照, 选定组合在 smoke_quarterly.py

特征是波动率期限结构和 FF5 投资暴露合成的一个分
logvol_21, logvol_63, logvol_126, logvol_252 覆盖短中长窗口
b_cma 来自含截距的 126 日 FF5 五因子回归, 取负权重, 高投资暴露的未来五日收益更低
五个因子先按日做截面 z-score, 再用 combine_scores 等权合成
组合分直接排序和放进 lambdarank 的验证集表现接近, 都比拆成原始特征好
标签是 future_returns(n_periods=5, gap=1) 收成的 31 档相关度
训练到 2022-12-31, 验证到 2024-12-31, embargo 为 6
long_short_5d 是验证集每天五分位多空拿到的 5 日收益, 各天重叠, 不是复利净值
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quanttoolkit.db.connection import DB, get_duckdb
from quanttoolkit.factors.combine import combine_scores
from quanttoolkit.factors.logvol_base import base_logvolfactor
from quanttoolkit.factors.zscore import zscore_by_date
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

WINDOWS = (21, 63, 126, 252)
EXPOSURE = 126
FF5 = ["mkt_rf", "smb", "hml", "rmw", "cma"]
WEIGHTS = {f"logvol_{window}": 1.0 for window in WINDOWS} | {"b_cma": -1.0}


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


def _b_cma(prices: pd.DataFrame, ff: pd.DataFrame, window: int) -> pd.Series:
    simple = prices["adj_close"].groupby(level="ticker").pct_change()
    excess = simple - ff["rf"].reindex(simple.index, level="date")
    excess = excess.where(prices["volume"] > 0)
    factors = ff[FF5].reindex(excess.index, level="date")
    pieces = []
    for index in excess.groupby(level="ticker").groups.values():
        ordered = excess.loc[index].sort_index()
        values = _rolling_exposure(
            ordered.to_numpy(dtype=float),
            factors.loc[ordered.index, FF5].to_numpy(dtype=float),
            window,
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
    factors = pd.DataFrame(
        {f"logvol_{window}": base_logvolfactor(prices, window) for window in WINDOWS}
    )
    factors["b_cma"] = _b_cma(prices, ff, EXPOSURE)
    score = combine_scores(zscore_by_date(factors), WEIGHTS)
    samples = assemble_samples(
        features=score.to_frame(),
        label=future_returns(df=prices, n_periods=5, gap=1),
    )
    relevance = relevance_by_date(samples.iloc[:, -1])
    features = samples.iloc[:, :-1]
    forward = samples.iloc[:, -1]
    roles = split_dates(
        features.index.get_level_values("date").unique(),
        train_end="2022-12-31",
        valid_end="2024-12-31",
        embargo=6,
    )

    def take(role: str) -> tuple[pd.DataFrame, pd.Series]:
        dates = roles.index[roles == role]
        part = features.loc[features.index.get_level_values("date").isin(dates)]
        return part, relevance.loc[part.index]

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

    valid_dates = roles.index[roles == "valid"]
    combined = score.loc[score.index.get_level_values("date").isin(valid_dates)]
    train_x, train_y = take("train")
    valid_x, _valid_y = take("valid")
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
