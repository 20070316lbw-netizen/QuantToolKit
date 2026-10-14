"""读取合并因子 SQL, 用截面相关和持有期收益评定合成分数"""

from pathlib import Path

import numpy as np
import pandas as pd

from quanttoolkit.db.connection import DB, get_duckdb
from quanttoolkit.db.operation import read_sql
from quanttoolkit.factors.combine import combine_scores
from quanttoolkit.factors.zscore import zscore_by_date
from quanttoolkit.metrics import (
    ic,
    ic_summary,
    max_drawdown,
    rank_ic,
    sharpe,
    total_return,
    win_rate,
)
from quanttoolkit.rebalance import rebalance_dates
from quanttoolkit.sizing import simple_quantile_long_short
from quanttoolkit.transfer import to_date_ticker_frame

SQL_PATH = Path(__file__).with_name("combine_factors.sql")
HOLDING_DAYS = 5
N_QUANTILES = 5
MIN_NAMES = 20
WEIGHTS = {
    "mom_20_1": -0.25,
    "reversal_5": 0.25,
    "rsi_14": -0.25,
    "logvol_20": 0.25,
}


def load_panel(path: Path = DB) -> pd.DataFrame:
    with get_duckdb(path=path) as connection:
        raw = connection.execute(read_sql(SQL_PATH)).df()
    panel = to_date_ticker_frame(df=raw)
    panel["score"] = combine_scores(zscore_by_date(panel[list(WEIGHTS)]), WEIGHTS)
    return panel


def evaluation_sample(panel: pd.DataFrame) -> pd.DataFrame:
    use = panel.dropna(subset=["score", "fwd_ret_5d"]).copy()
    counts = use.groupby(level="date")["score"].transform("size")
    use = use.loc[counts >= MIN_NAMES]
    use["forward_return_5"] = np.expm1(use["fwd_ret_5d"])
    return use


def correlation_table(sample: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    forward = sample["forward_return_5"]
    rows = {}
    for name in columns:
        pearson = ic(factor=sample[name], forward_return=forward)
        spearman = rank_ic(factor=sample[name], forward_return=forward)
        pearson_summary = ic_summary(ic=pearson)
        spearman_summary = ic_summary(ic=spearman)
        rows[name] = {
            "ic_mean": pearson_summary["ic_mean"],
            "ic_ir": pearson_summary["ic_ir"],
            "rank_ic_mean": spearman_summary["ic_mean"],
            "rank_ic_ir": spearman_summary["ic_ir"],
            "dates": int(pearson.notna().sum()),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def period_returns(sample: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    dates = pd.DatetimeIndex(sample.index.get_level_values("date").unique())
    anchors = rebalance_dates(dates, freq=HOLDING_DAYS)
    held = sample.loc[pd.IndexSlice[anchors, :]]
    weights = simple_quantile_long_short(score=held["score"], n_quantiles=N_QUANTILES)
    pnl = weights.mul(held["forward_return_5"].reindex(weights.index))
    long = pnl.where(weights > 0, 0.0).groupby(level="date").sum()
    spread = pnl.groupby(level="date").sum()
    long.name = "top_quintile"
    spread.name = "long_short"
    return long, spread


def return_table(returns: pd.Series) -> pd.Series:
    raw_sharpe = sharpe(returns=returns)
    annualized = raw_sharpe * np.sqrt(252 / HOLDING_DAYS)
    return pd.Series(
        {
            "periods": float(len(returns)),
            "total_return": total_return(returns=returns),
            "sharpe": raw_sharpe,
            "sharpe_annualized": annualized,
            "max_drawdown": max_drawdown(returns=returns),
            "win_rate": win_rate(returns=returns),
        },
        name=returns.name,
    )


def main(path: Path = DB) -> None:
    sample = evaluation_sample(load_panel(path))
    dates = sample.index.get_level_values("date")
    print(
        f"样本 {dates.min():%Y-%m-%d} 至 {dates.max():%Y-%m-%d}, "
        f"{dates.nunique()} 个交易日, "
        f"平均每日 {sample.groupby(level='date').size().mean():.0f} 只"
    )
    print("权重: mom_20_1 -0.25, reversal_5 +0.25, rsi_14 -0.25, logvol_20 +0.25")
    print("标签: 同一收盘价起未来 5 个交易日的简单收益")

    print("\n全部交易日的截面相关, 5 日收益逐日重叠, 信息比率偏高")
    daily = correlation_table(
        sample,
        ["score", "mom_20_1", "reversal_5", "rsi_14", "logvol_20"],
    )
    print(daily.to_string(float_format=lambda value: f"{value: .4f}"))

    anchors = rebalance_dates(
        pd.DatetimeIndex(dates.unique()),
        freq=HOLDING_DAYS,
    )
    spaced = sample.loc[pd.IndexSlice[anchors, :]]
    print("\n每 5 个交易日取一次的截面相关, 持有期不重叠")
    print(
        correlation_table(
            spaced,
            ["score", "mom_20_1", "reversal_5", "rsi_14", "logvol_20"],
        ).to_string(float_format=lambda value: f"{value: .4f}")
    )

    long, spread = period_returns(sample)
    book = pd.concat(
        [return_table(long), return_table(spread)],
        axis=1,
    )
    print("\n每 5 个交易日再平衡, 按五分位构造权重")
    print("top_quintile 是最高组, 组内等权且权重和为 1")
    print("long_short 是最高组权重和为 1, 最低组权重和为 -1")
    print("sharpe 是指标函数的原值, 未年化; sharpe_annualized 按每年 252/5 期折算")
    print(book.to_string(float_format=lambda value: f"{value: .4f}"))


if __name__ == "__main__":
    main()
