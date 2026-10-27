# 分位仓位

实现见 [sizing.py](../src/quanttoolkit/portfolio/sizing.py)。
先准备 `[date, ticker]` 分数 Series, 用 [调仓日历](rebalance.md) 筛出要调仓的日期,
再交给仓位函数。每一天各自分组, 函数不会替我们决定哪天调仓。

## 两种仓位规则

| 接口 | 分组方式 | 中间组 |
| --- | --- | --- |
| `simple_quantile_long_short` | 当天全截面按分数分位 | 保留为 0 权重 |
| `vol_quantile_long_short` | 先按波动率分层, 再在各层内按分数分位 | 不写入结果 |

两者均返回名为 `weight` 的 Series, 多头等权且权重和为 1, 空头等权且权重和为 -1。
多空相加的净权重为 0, 但两边的绝对权重加起来是 2, 接到回测时要按这个资金口径处理。
函数不修改输入, 也不计算实际持仓、成交费用或组合收益。

## 全截面分位多空

`simple_quantile_long_short` 用 `pd.qcut` 将当天分数切成 `n_quantiles` 组,
最高组做多, 最低组做空。重复分位边界会被丢掉, 极端组可能宽于 `1/n_quantiles`。
缺失分数先剔除, 不记为 0。

当天有效股票少于 `n_quantiles`, 或实际分组少于 2 组时, 整天不出现在结果中。
`n_quantiles` 小于 2 时抛出 `ValueError`。

```python
import pandas as pd

from quanttoolkit.portfolio.sizing import simple_quantile_long_short

index = pd.MultiIndex.from_product(
    [pd.to_datetime(["2024-01-01"]), ["A", "B", "C", "D", "E", "F"]],
    names=["date", "ticker"],
)
score = pd.Series([5.0, 3.0, 1.0, -1.0, -3.0, -5.0], index=index)
weights = simple_quantile_long_short(score=score, n_quantiles=3)
assert weights.name == "weight"
assert weights.tolist() == [0.5, 0.5, 0.0, 0.0, -0.5, -0.5]
assert weights[weights > 0].sum() == 1
assert weights[weights < 0].sum() == -1
```

## 波动率分层多空

`vol_quantile_long_short` 先用 `pd.qcut` 按波动率分成 `n_vol_groups` 层,
再在每层内按分数分成 `n_quantiles` 组, 选出最高组和最低组。
先把各层选出的多头合在一起, 再给每只股票相同权重, 空头也一样, 所以每层拿到的资金不一定相同。
风险输入可用 [history_vol](volatility.md), 应使用信号生成时已经知道的历史值。

分数与波动率先按索引取交集, 剔除任一侧缺失的股票。
当天有效股票少于 `n_vol_groups` 时跳过整天; 波动率有足够记录但切不开时, 整天当作一层。
重复边界会被丢掉, 实际层数和组数可能少于请求值。
某层有效股票少于 `n_quantiles` 或实际分组不足 2 组时, 跳过该层。
当天没有多头或没有空头时, 整天不出现。`n_quantiles` 小于 2 时抛出 `ValueError`。

```python
import pandas as pd

from quanttoolkit.portfolio.sizing import vol_quantile_long_short

tickers = ["A", "B", "C", "D", "E", "F", "G", "H"]
index = pd.MultiIndex.from_product(
    [pd.to_datetime(["2024-01-01"]), tickers],
    names=["date", "ticker"],
)
score = pd.Series([9, 7, 5, 3, 20, 16, 12, 4], index=index)
vol = pd.Series([1, 1, 1, 1, 9, 9, 9, 9], index=index)
weights = vol_quantile_long_short(score=score, vol=vol, n_vol_groups=2, n_quantiles=2)
assert weights.name == "weight"
assert sorted(weights[weights > 0].index.get_level_values("ticker")) == [
    "A",
    "B",
    "E",
    "F",
]
assert sorted(weights[weights < 0].index.get_level_values("ticker")) == [
    "C",
    "D",
    "G",
    "H",
]
assert weights[weights > 0].sum() == 1
assert weights[weights < 0].sum() == -1
assert (weights == 0).sum() == 0
```

“结果中没有某天或某只股票”表示没有生成该项权重, 与显式返回 0 不同。
把目标权重接到实际持仓时, 还要自己决定缺项表示什么, 何时平仓, 以及旧仓位是否继续保留。
