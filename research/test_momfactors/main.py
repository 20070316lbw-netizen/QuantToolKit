from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from quanttoolkit.db.connection import DB, get_duckdb
from quanttoolkit.factors.mom_base import base_momfactor
from quanttoolkit.factors.neutral import neutralize_by_industry
from quanttoolkit.panel.returns import future_returns

with get_duckdb(path=DB) as connection:
    universe = connection.execute("SELECT ticker, name, sub_industry FROM universe ORDER BY ticker") .df()
    prices = connection.execute(
        """
        SELECT date, ticker, adj_close, volume
        FROM prices
        ORDER BY date, ticker
        """
        ).df()

industry = universe.set_index("ticker")["sub_industry"]
# print(industry.head())

# # 每个 ticker 应只对应一条行业记录，行业标签也不能缺失

# # 1. 检查 ticker 是否唯一
# print("ticker 是否唯一: ", industry.index.is_unique)

# # 2. 检查行业是否有缺失
# print("行业缺失数量", industry.isna().sum())

# # 3. 检查是否有内容为空字符串
# print("行业空字符串数量", industry.eq("").sum())

prices = prices.set_index(["date", "ticker"]).sort_index()

mom = base_momfactor(prices=prices, window=63, skip=0)
target = future_returns(df = prices, n_periods=20, gap=1)
mom_frame = mom.to_frame()
mom_neutral = neutralize_by_industry(mom_frame, industry)["mom_63"]

panel = pd.concat(
        {
            "mom":mom,
            "mom_neutral": mom_neutral, 
            "target": target
        },
        axis=1,
    )

sample = panel.dropna(subset=["mom", "mom_neutral", "target"])

# print("样本索引是否唯一:", sample.index.is_unique)
# print("各列是否全部为有限数:")
# print(np.isfinite(sample).all())

daily_n = sample.groupby(level='date').size()
# print(daily_n)
# print(daily_n.describe())

one_day = sample.xs("2016-04-05", level="date")
ic_raw = one_day["mom"].corr(one_day["target"], method="pearson")
ic_neutral = one_day["mom_neutral"].corr(one_day["target"], method="pearson")

# print("当天原始动量 IC:", ic_raw)
# print("当天中性化动量 IC:", ic_neutral)

rank_ic_raw = one_day["mom"].corr(one_day["target"], method="spearman")
rank_ic_neutral = one_day["mom_neutral"].corr(one_day["target"], method="spearman")

# print("当天原始动量 Rank IC:", rank_ic_raw)
# print("当天中性化动量 Rank IC:", rank_ic_neutral)


def evaluate_day(day):
    return pd.Series(
        {
            "ic_raw": day["mom"].corr(day["target"], method="pearson"),
            "ic_neutral": day["mom_neutral"].corr(day["target"], method="pearson"),
            "rank_ic_raw": day["mom"].corr(day["target"], method="spearman"),
            "rank_ic_neutral": day["mom_neutral"].corr(day["target"], method="spearman"),
        }
    )

daily_ic = sample.groupby(level="date").apply(evaluate_day)

ic_mean = daily_ic.mean()
ic_std = daily_ic.std(ddof=1)
icir = ic_mean / ic_std

summary = pd.concat(
        {
            "mean": ic_mean,
            "std": ic_std,
            "icir": icir
        },
        axis=1
    )

rank_ic = daily_ic["rank_ic_raw"]
# print("Rank IC 1 期自相关: ", rank_ic.autocorr(lag=1))
# print("Rank IC 20 期自相关: ", rank_ic.autocorr(lag=20))

rank_ic = daily_ic["rank_ic_raw"].sort_index()
x = np.ones((len(rank_ic), 1))
mean_model = sm.OLS(rank_ic, x)
mean_result = mean_model.fit()

# print("回归估计的系数:")
# print(mean_result.params)
# print("直接计算的均值:", rank_ic.mean())

nw_result = mean_model.fit(
    cov_type="HAC",
    cov_kwds={"maxlags": 20},
    use_t=False,
)

# print("平均 Rank IC:")
# print(nw_result.params)
# print("Newey-West 标准误:")
# print(nw_result.bse)

print("平均 Rank IC 的 95% 置信区间")
print(nw_result.conf_int(alpha = 0.05))