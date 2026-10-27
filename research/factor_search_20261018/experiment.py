"""先固定候选, 窗口和方向, 再按协议选择一个因子查看本轮留出区间"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BENCHMARKS = ["mom_63", "mom_231_skip_21"]
CANDIDATES = [
    "intraday_strength_63",
    "overnight_reversal_63",
    "closing_pressure_21",
    "price_volume_divergence_63",
    "low_volatility_63",
    "smooth_momentum_63",
]
PERIODS = {
    "discovery": ("2017-01-01", "2020-12-31"),
    "selection": ("2021-01-01", "2022-12-31"),
    "check": ("2023-01-01", "2024-12-31"),
    "holdout": ("2025-01-01", "2099-12-31"),
}


def _load_prices() -> tuple[dict[str, pd.DataFrame], pd.Series]:
    with duckdb.connect(str(ROOT / "data/quanttoolkit.db"), read_only=True) as conn:
        raw = conn.execute("SELECT * FROM prices ORDER BY date, ticker").df()
        industry = conn.execute("SELECT ticker, sub_industry FROM universe").df()
    raw["date"] = pd.to_datetime(raw["date"])
    prices = {
        field: raw.pivot(index="date", columns="ticker", values=field)
        for field in ["open", "high", "low", "close", "adj_close", "volume"]
    }
    return prices, industry.set_index("ticker")["sub_industry"]


def _make_factors(p: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    adj = p["adj_close"]
    live = p["volume"] > 0
    ret = np.log(adj).diff().where(live & live.shift(1, fill_value=False))
    intra = np.log(p["close"] / p["open"]).where(live)
    adjusted_open = p["open"] * adj / p["close"]
    overnight = np.log(adjusted_open / adj.shift(1)).where(
        live & live.shift(1, fill_value=False)
    )
    clv = (2 * p["close"] - p["high"] - p["low"]) / (p["high"] - p["low"])
    clv = clv.where((p["high"] > p["low"]) & live)
    volume = p["volume"].where(live)
    relative = np.log(volume / volume.rolling(21).mean())
    total = ret.rolling(63).sum()
    result = {
        "mom_63": np.log(adj / adj.shift(63)),
        "mom_231_skip_21": np.log(adj.shift(21) / adj.shift(252)),
        "intraday_strength_63": intra.rolling(63).sum(),
        "overnight_reversal_63": -overnight.rolling(63).sum(),
        "closing_pressure_21": clv.rolling(21).mean(),
        "price_volume_divergence_63": -ret.rolling(63).corr(relative),
        "low_volatility_63": -ret.rolling(63).std(),
        "smooth_momentum_63": total / ret.abs().rolling(63).sum(),
    }
    return {
        name: frame.replace([np.inf, -np.inf], np.nan) for name, frame in result.items()
    }


def _label_and_mask(p, factors, horizon=21):
    adj = p["adj_close"]
    label = adj.shift(-(horizon + 1)) / adj.shift(-1) - 1
    live = p["volume"] > 0
    mask = (
        (live.rolling(64).sum() == 64)
        & live.shift(-1, fill_value=False)
        & live.shift(-(horizon + 1), fill_value=False)
        & adj.shift(252).notna()
        & label.notna()
    )
    for frame in factors.values():
        mask &= frame.notna()
    return label, mask


def _date_mask(calendar, period, horizon=21):
    start, end = PERIODS[period]
    exits = pd.Series(calendar, index=calendar).shift(-(horizon + 1))
    return np.asarray((calendar >= start) & (calendar <= end) & (exits <= end))


def _daily_ic(factor, label, mask):
    x = factor.where(mask).rank(axis=1, method="average")
    y = label.where(mask).rank(axis=1, method="average")
    return x.corrwith(y, axis=1).where(mask.sum(axis=1) >= 100)


def _select(p, factors):
    label, mask = _label_and_mask(p, factors)
    rows = []
    series = {}
    for name, frame in factors.items():
        score = _daily_ic(frame, label, mask)
        series[name] = score
        for period in ["discovery", "selection"]:
            part = score.loc[_date_mask(score.index, period)].dropna()
            rows.append(
                {
                    "factor": name,
                    "period": period,
                    "rank_ic": part.mean(),
                    "days": len(part),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(HERE / "selection.csv", index=False)
    shortlist = table.loc[(table.period == "selection") & table.factor.isin(CANDIDATES)]
    chosen = shortlist.sort_values("rank_ic", ascending=False).iloc[0].factor
    record = {
        "selected": chosen,
        "protocol_sha256": hashlib.sha256(
            (HERE / "protocol.json").read_bytes()
        ).hexdigest(),
        "metrics_seen": ["discovery", "selection"],
    }
    (HERE / "frozen_selection.json").write_text(json.dumps(record, indent=2) + "\n")
    return chosen, table


def _paired_bootstrap(diff, block=63, repetitions=2000):
    values = diff.dropna().to_numpy()
    n = len(values)
    rng = np.random.default_rng(18)
    count = (n + block - 1) // block
    starts = rng.integers(0, n, size=(repetitions, count))
    indices = (starts[..., None] + np.arange(block)) % n
    means = values[indices.reshape(repetitions, -1)[:, :n]].mean(axis=1)
    return np.quantile(means, [0.025, 0.975]).tolist()


def _demean_by_industry(frame, industry):
    labels = industry.reindex(frame.columns)
    means = frame.T.groupby(labels, sort=False).transform("mean").T
    return frame - means


def _evaluate(p, factors, industry, chosen):
    names = BENCHMARKS + [chosen]
    label, mask = _label_and_mask(p, factors)
    scores = pd.DataFrame(
        {name: _daily_ic(factors[name], label, mask) for name in names}
    )
    scores.to_csv(HERE / "daily_rank_ic.csv")
    rows, differences, yearly, quintiles = [], [], [], []
    for period in PERIODS:
        dates = _date_mask(label.index, period)
        for name in names:
            part = scores.loc[dates, name].dropna()
            x, y = factors[name].where(mask).loc[dates], label.where(mask).loc[dates]
            rank = x.rank(axis=1, pct=True, method="average")
            groups = np.ceil(rank * 5).clip(1, 5)
            group_means = pd.DataFrame(
                {i: y.where(groups == i).mean(axis=1) for i in range(1, 6)}
            )
            rows.append(
                {
                    "factor": name,
                    "period": period,
                    "rank_ic": part.mean(),
                    "ic_ir_unannualized": part.mean() / part.std(),
                    "ic_positive": (part > 0).mean(),
                    "days": len(part),
                    "first": str(part.index.min().date()),
                    "last": str(part.index.max().date()),
                    "mean_stocks": mask.loc[part.index].sum(axis=1).mean(),
                    "q5_minus_q1_forward_return": (
                        group_means[5] - group_means[1]
                    ).mean(),
                }
            )
            for group, value in group_means.mean().items():
                quintiles.append(
                    {
                        "factor": name,
                        "period": period,
                        "quintile": group,
                        "mean_forward_return": value,
                    }
                )
        if period in ["check", "holdout"]:
            for baseline in BENCHMARKS:
                diff = (scores[chosen] - scores[baseline]).loc[dates].dropna()
                low, high = _paired_bootstrap(diff)
                differences.append(
                    {
                        "period": period,
                        "baseline": baseline,
                        "mean_ic_difference": diff.mean(),
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
    for year in range(2017, 2027):
        dates = (scores.index.year == year) & (scores.index >= "2017-01-01")
        for name in names:
            part = scores.loc[dates, name].dropna()
            if len(part):
                yearly.append(
                    {
                        "factor": name,
                        "year": year,
                        "rank_ic": part.mean(),
                        "days": len(part),
                    }
                )
    pd.DataFrame(rows).to_csv(HERE / "comparison.csv", index=False)
    pd.DataFrame(differences).to_csv(HERE / "paired_uncertainty.csv", index=False)
    pd.DataFrame(yearly).to_csv(HERE / "yearly.csv", index=False)
    pd.DataFrame(quintiles).to_csv(HERE / "quintiles.csv", index=False)
    robustness = []
    for horizon in [5, 21, 63]:
        y, m = _label_and_mask(p, factors, horizon)
        for adjusted in [False, True]:
            y_used = _demean_by_industry(y.where(m), industry) if adjusted else y
            for name in names:
                x = (
                    _demean_by_industry(factors[name].where(m), industry)
                    if adjusted
                    else factors[name]
                )
                score = _daily_ic(x, y_used, m)
                for period in ["check", "holdout"]:
                    part = score.loc[_date_mask(score.index, period, horizon)].dropna()
                    robustness.append(
                        {
                            "factor": name,
                            "horizon": horizon,
                            "industry_demeaned": adjusted,
                            "period": period,
                            "rank_ic": part.mean(),
                            "days": len(part),
                        }
                    )
    pd.DataFrame(robustness).to_csv(HERE / "robustness.csv", index=False)
    correlations = []
    for baseline in BENCHMARKS:
        score = _daily_ic(factors[chosen], factors[baseline], mask)
        for period in ["check", "holdout"]:
            part = score.loc[_date_mask(score.index, period)].dropna()
            correlations.append(
                {
                    "baseline": baseline,
                    "period": period,
                    "rank_correlation": part.mean(),
                }
            )
    pd.DataFrame(correlations).to_csv(HERE / "factor_correlations.csv", index=False)
    return pd.DataFrame(rows), pd.DataFrame(differences)


def main():
    prices, industry = _load_prices()
    factors = _make_factors(prices)
    chosen, table = _select(prices, factors)
    print(table.to_string(index=False), flush=True)
    print(f"FROZEN: {chosen}", flush=True)
    summary, uncertainty = _evaluate(prices, factors, industry, chosen)
    print(summary.to_string(index=False), flush=True)
    print(uncertainty.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
