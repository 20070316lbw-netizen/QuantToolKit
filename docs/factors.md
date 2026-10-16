# 因子基础函数

基础函数接受 `[date, ticker]` MultiIndex 行情, 在每只证券内按交易记录计算, 返回命名的 Series
`window` 是窗口内的单期收益数量, `skip` 是剔除最近收益的记录数, 不包含在 window 内

## 动量

[`base_momfactor`](../src/quanttoolkit/factors/mom_base.py) 计算窗口内累计对数收益率
在记录 t 上, 因子为 `log(P[t-skip] / P[t-skip-window])`, 使用截至 t-skip 的历史价格
`skip=0` 时名称为 `mom_{window}`, 否则为 `mom_{window}_{skip}`
源码 docstring 含可独立执行的示例, 测试会校验其结果

## 反转

[`base_reversalfactor`](../src/quanttoolkit/factors/reversal_base.py) 复用 `base_momfactor`, 取累计对数收益率的负值

```python
import numpy as np
import pandas as pd

from quanttoolkit.factors.reversal_base import base_reversalfactor
from quanttoolkit.transfer import to_date_ticker_frame

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

在记录 t 上, 因子为 `-log(P[t-skip] / P[t-skip-window])`, 上涨得到负值, 下跌得到正值
计算会检查窗口是否具有完整收益记录, 窗口内价格缺失或记录不足时保留 NaN
不填补价格, 不年化, 不做波动率调整, 不修改输入, 不额外校验字段或参数
`skip=0` 时名称为 `reversal_{window}`, 否则为 `reversal_{window}_{skip}`
首次有效值需要 `window+skip+1` 条价格记录, 日历缺口不会补齐

## 对数波动率

[`base_logvolfactor`](../src/quanttoolkit/factors/logvol_base.py) 计算窗口内单期对数收益率的样本标准差
在记录 t 上, 因子为 `std(log(P[t-window+1] / P[t-window]), ..., log(P[t] / P[t-1]))`, `ddof=1`
名称为 `logvol_{window}`
不年化, 不填补价格, 不修改输入
窗口内对数收益缺失或记录不足时为 NaN
样本标准差至少需要 2 条对数收益, 首次有效值需要 `window+1` 条价格记录

```python
import numpy as np
import pandas as pd

from quanttoolkit.factors.logvol_base import base_logvolfactor
from quanttoolkit.transfer import to_date_ticker_frame

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

[`base_rsifactor`](../src/quanttoolkit/factors/rsi_base.py) 计算 Wilder RSI
`window` 是连续价格变动的个数, 默认 14, 名称为 `rsi_{window}`
变动为 `P[t] - P[t-1]`, 上涨记正, 下跌记正的绝对值
前 `window` 个连续变动用简单平均作为平均上涨和平均下跌
之后 `avg = (上一期 avg * (window - 1) + 当期涨跌) / window`
`RSI = 100 * 平均上涨 / (平均上涨 + 平均下跌)`
只有上涨时为 100, 只有下跌时为 0, 两者都为 0 时为 NaN
默认价格列是 `adj_close`
不填补价格, 不修改输入, 也不跳过成交量为 0 的补齐日
价格缺失会打断平滑, 之后重新累计 `window` 个连续变动
首次有效值需要 `window+1` 条价格记录

```python
import pandas as pd

from quanttoolkit.factors.rsi_base import base_rsifactor
from quanttoolkit.transfer import to_date_ticker_frame

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

## 基本面

[`filing_panel`](../src/quanttoolkit/fundamentals.py) 把时点科目和期间科目收成每个 `(ticker, filed)` 一行
时点科目取该披露日及之前 `period_end` 最晚的一条, 只修订更早的截止日时不退回比较期
单季跨度包含 80 到 100 天, 全年跨度包含 350 到 380 天
全年截止日还没有单季时, 第四季等于全年减去该财年内的前三季
`net_income_ttm` 是当时已经披露的最近四个单季之和
`asset_growth` 是当前资产除以大约一年前的资产再减 1, 间隔取 300 到 430 天里最接近 365 天的那份
算不出增长率的资产披露日, 增长率缺失, 不沿用上一份
[`read_filing_panel`](../src/quanttoolkit/fundamentals.py) 从已入库的三张表读出这个面板

[`align_filings`](../src/quanttoolkit/fundamentals.py) 按 `filed` 把面板对齐到行情的每个交易日, 披露当天可用
[`book_to_market`](../src/quanttoolkit/factors/fundamental.py) 是权益除以当天 `close` 乘以股数
[`earnings_yield`](../src/quanttoolkit/factors/fundamental.py) 是滚动四季净利润除以同一市值
市值不用 `adj_close`
[`asset_growth`](../src/quanttoolkit/factors/fundamental.py) 把申报日上的资产增长率放到每个交易日
三个因子都返回 `[date, ticker]` Series, 可以交给 `zscore_by_date`

```python
import pandas as pd

from quanttoolkit.factors.fundamental import book_to_market

index = pd.MultiIndex.from_product(
    [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["A"]],
    names=["date", "ticker"],
)
prices = pd.DataFrame({"close": [5.0, 10.0]}, index=index)
filings = pd.DataFrame(
    {
        "ticker": ["A"],
        "filed": pd.to_datetime(["2024-01-02"]),
        "equity": [50.0],
        "shares": [2.0],
    }
)
assert book_to_market(prices, filings).tolist() == [5.0, 2.5]
```

## 截面标准化

[`zscore_by_date`](../src/quanttoolkit/factors/zscore.py) 把因子面板的每一列变成当天截面上的 z-score
`z = (原始值 - 当天均值) / 当天样本标准差`, 标准差 `ddof=1`
缺失值不进入当天的均值和标准差
当天该列有效值少于 2 个, 或标准差为 0 时, 结果为 NaN
不修改输入

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

## 加权合成

[`combine_scores`](../src/quanttoolkit/factors/combine.py) 把已经标准化的因子列合成名为 `score` 的 Series
某一行缺失的因子不参与该行
分子是权重乘因子后求和, 分母是参与因子的权重绝对值之和
权重不必事先加总为 1, 负数权重用来反转因子方向
未传入权重时每一列权重为 1
不修改输入

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
