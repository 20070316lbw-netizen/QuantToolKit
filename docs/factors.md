# 因子计算与合成

先把行情整理成 `[date, ticker]` MultiIndex, 再交给基础价格因子。
各只证券按自己的交易记录计算, 结果是带名称的 Series。默认用 `adj_close`, 想换价格列时传 `price_col`。
输入格式见 [数据约定](data-structure.md), 组合流程见 [推荐调用顺序](calling-guide.md)。

## 接口速查

| 接口 | 作用 | 输出 |
| --- | --- | --- |
| `base_momfactor` | 窗口累计对数收益 | Series `mom_{window}` 或 `mom_{window}_{skip}` |
| `base_reversalfactor` | 动量取负 | Series `reversal_{window}` 或 `reversal_{window}_{skip}` |
| `base_logvolfactor` | 单期对数收益的样本标准差 | Series `logvol_{window}` |
| `base_rsifactor` | Wilder RSI, 默认窗口 14 | Series `rsi_{window}` |
| `neutralize_by_industry` | 每日每行业去均值 | 同索引、同列的 DataFrame |
| `zscore_by_date` | 每列按日期做截面标准化 | 同索引、同列的 DataFrame |
| `combine_scores` | 已标准化因子的加权合成 | Series `score` |

`window` 表示窗口内的单期收益或价格变动数量, `skip` 表示剔除最近多少期收益,
不包含在 `window` 内。价格因子不填补价格或日期, 历史不足时保留 NaN。

多个因子放在一起时, 顺序是先计算基础因子, 用 `pd.concat(..., axis=1)` 合成面板,
需要的话再去行业均值, 然后按日标准化, 最后加权合成一个分数。
单个因子可以直接作分数使用。

## 动量

[`base_momfactor`](../src/quanttoolkit/factors/mom_base.py) 计算窗口内累计对数收益率。
在记录 t 上, 因子为 `log(P[t-skip] / P[t-skip-window])`, 使用截至 t-skip 的历史价格。
`skip=0` 时名称为 `mom_{window}`, 否则为 `mom_{window}_{skip}`。
首次有效值需要 `window+skip+1` 条价格记录, 窗口内价格缺失时为 NaN。
完整窗口下, 公式可以化成两个端点的比值, 不过代码是先算单期对数收益再求和, 所以整个窗口都要完整。
源码 docstring 含可独立执行的示例。

## 反转

[`base_reversalfactor`](../src/quanttoolkit/factors/reversal_base.py) 将动量取负,
得到 `-log(P[t-skip] / P[t-skip-window])`: 窗口上涨时为负, 下跌时为正。
它与动量使用相同窗口和缺失值规则, 不额外做波动率调整。
名称为 `reversal_{window}`, `skip` 非零时为 `reversal_{window}_{skip}`。

```python
import numpy as np
import pandas as pd

from quanttoolkit.factors.reversal_base import base_reversalfactor
from quanttoolkit.panel.transfer import to_date_ticker_frame

prices = to_date_ticker_frame(
    df=pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=5),
            "ticker": ["A"] * 5,
            "adj_close": [100.0, 110.0, 99.0, 118.8, 118.8],
        }
    )
)
factor = base_reversalfactor(prices, window=2, skip=1)
assert factor.name == "reversal_2_1"
assert factor.iloc[:3].isna().all()
assert np.isclose(factor.iloc[3], -np.log(99.0 / 100.0))
assert np.isclose(factor.iloc[4], -np.log(118.8 / 110.0))
```

## 对数波动率

[`base_logvolfactor`](../src/quanttoolkit/factors/logvol_base.py) 计算窗口内单期对数收益率的样本标准差,
即 `std(log(P[t-window+1] / P[t-window]), ..., log(P[t] / P[t-1]))`, `ddof=1`。
结果名为 `logvol_{window}`, 不年化。窗口内对数收益缺失或记录不足时为 NaN。
窗口至少需要 2 条对数收益, 首次有效值需要 `window+1` 条价格记录。
它与使用简单收益的 [history_vol](volatility.md) 口径不同。

```python
import numpy as np
import pandas as pd

from quanttoolkit.factors.logvol_base import base_logvolfactor
from quanttoolkit.panel.transfer import to_date_ticker_frame

prices = to_date_ticker_frame(
    df=pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=4),
            "ticker": ["A"] * 4,
            "adj_close": [1.0, 2.0, 4.0, 8.0],
        }
    )
)
factor = base_logvolfactor(prices, window=2)
assert factor.name == "logvol_2"
assert factor.iloc[:2].isna().all()
assert np.isclose(factor.iloc[2], 0.0)
assert np.isclose(factor.iloc[3], 0.0)
```

## RSI

[`base_rsifactor`](../src/quanttoolkit/factors/rsi_base.py) 计算 Wilder RSI,
窗口默认 14, 结果名为 `rsi_{window}`。价格变动为 `P[t] - P[t-1]`,
上涨部分和下跌绝对值分别计算平均。前 `window` 个连续变动用简单平均初始化,
之后按 `avg = (上一期 avg * (window - 1) + 当期涨跌) / window` 平滑。

`RSI = 100 * 平均上涨 / (平均上涨 + 平均下跌)`。
只有上涨时为 100, 只有下跌时为 0, 两者都为 0 时为 NaN。
首次有效值需要 `window+1` 条价格记录。价格缺失会打断平滑,
之后重新累计 `window` 个连续变动; 成交量为 0 的补齐日仍参与计算。

```python
import pandas as pd

from quanttoolkit.factors.rsi_base import base_rsifactor
from quanttoolkit.panel.transfer import to_date_ticker_frame

prices = to_date_ticker_frame(
    df=pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=4),
            "ticker": ["A"] * 4,
            "adj_close": [1.0, 2.0, 1.0, 2.0],
        }
    )
)
factor = base_rsifactor(prices, window=2)
assert factor.name == "rsi_2"
assert factor.iloc[:2].isna().all()
assert factor.iloc[2] == 50.0
assert factor.iloc[3] == 75.0
```

## 截面标准化

[`zscore_by_date`](../src/quanttoolkit/factors/zscore.py) 将因子面板的每列按日期做截面标准化:
`z = (原始值 - 当天均值) / 当天样本标准差`, `ddof=1`。
缺失值不进入均值和标准差; 当天该列有效值少于 2 个或标准差为 0 时, 结果为 NaN。
比较的是同一天的股票, 均值和标准差也只用当天的数据, 不从整个历史区间估计, 不修改输入。

```python
import pandas as pd

from quanttoolkit.factors.zscore import zscore_by_date

index = pd.MultiIndex.from_product(
    [pd.to_datetime(["2024-01-01"]), ["A", "B", "C"]],
    names=["date", "ticker"],
)
factors = pd.DataFrame({"f1": [1.0, 2.0, 3.0]}, index=index)
assert zscore_by_date(factors)["f1"].tolist() == [-1.0, 0.0, 1.0]
```

## 行业中性

[`neutralize_by_industry`](../src/quanttoolkit/factors/neutral.py) 对每列因子在每个日期内减去同行业均值。
行业标签是以 `ticker` 为索引的 Series, 按证券代码对齐, 不要求带日期。
缺少行业标签的证券结果为 NaN; 行业内只有一个有效值时, 该值减去自身得到 0。
结果不修改输入, 可继续交给 `zscore_by_date` 和 `combine_scores`。
本接口使用传入的静态行业标签, 不自动处理历史行业变更。

```python
import pandas as pd

from quanttoolkit.factors.combine import combine_scores
from quanttoolkit.factors.neutral import neutralize_by_industry
from quanttoolkit.factors.zscore import zscore_by_date

index = pd.MultiIndex.from_product(
    [pd.to_datetime(["2024-01-02"]), ["A", "B", "C"]],
    names=["date", "ticker"],
)
factors = pd.DataFrame({"f1": [1.0, 3.0, 10.0]}, index=index)
industry = pd.Series(["Tech", "Tech", "Bank"], index=["A", "B", "C"])
neutral = neutralize_by_industry(factors, industry)
assert neutral["f1"].tolist() == [-1.0, 1.0, 0.0]
score = combine_scores(zscore_by_date(neutral))
assert score.name == "score"
assert score.tolist() == [-1.0, 1.0, 0.0]
```

## 加权合成

[`combine_scores`](../src/quanttoolkit/factors/combine.py) 将已标准化的因子列合成 `score` Series:
分子是权重乘因子后求和, 分母是该行参与因子的权重绝对值之和。
权重不必加总为 1, 负数权重反转因子方向; 未传权重时, 每列权重为 1。
显式传入权重字典时, 未列出的因子权重为 0。

某行缺失的因子不参与该行合成, 分母也随之调整; 没有可用权重时结果为 NaN。
而 [assemble_samples](dataset.md) 要等每个特征和标签都有值才保留这一行, 两处缺失值规则要分开看。
函数不修改输入, 合成后可接 [IC 评估](metrics.md) 或 [调仓仓位](sizing.md)。

```python
import pandas as pd

from quanttoolkit.factors.combine import combine_scores

index = pd.MultiIndex.from_product(
    [pd.to_datetime(["2024-01-01"]), ["A", "B", "C"]],
    names=["date", "ticker"],
)
z = pd.DataFrame(
    {"f1": [-1.0, 0.0, 1.0], "f2": [-0.71, float("nan"), 0.71]},
    index=index,
)
score = combine_scores(z, {"f1": 0.25, "f2": -0.25})
assert score.name == "score"
assert round(float(score.iloc[0]), 3) == -0.145
assert score.iloc[1] == 0.0
assert round(float(score.iloc[2]), 3) == 0.145
```

## FF5 CMA 暴露

`quanttoolkit.factors.exposure.cma_exposure(prices, ff, window)` 返回名为 `b_cma` 的 Series。
`prices` 使用已排序的 `[date, ticker]` 索引, 包含 `adj_close` 和 `volume`;
`ff` 按日期索引, 包含 `mkt_rf, smb, hml, rmw, cma, rf`, 收益均使用小数单位。
函数以股票简单收益减 `rf` 为因变量, 对同期 FF5 加截距做滚动回归, 只返回 CMA 系数。
窗口按每只证券自己的记录计数, 必须完整有效; 价格或因子缺失, 或成交量非正数时该记录不参与有效窗口。
回归矩阵的对角线上加 `1e-12`, 用来减轻求解时的数值不稳定, 不年化, 不修改输入。
两个 smoke 脚本共用此函数, 分别使用 126 和 63 条记录的窗口。
