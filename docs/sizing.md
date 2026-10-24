# 分位仓位

实现见 [sizing.py](../src/quanttoolkit/portfolio/sizing.py)
输入是调仓日上的 `[date, ticker]` 分数, 每个日期单独分组

`simple_quantile_long_short` 用 `pd.qcut` 把当天分数分成 `n_quantiles` 组
最高组等权做多, 权重和为 1
最低组等权做空, 权重和为 -1
中间组为 0
重复的分位边界会被丢掉, 极端组因此可能宽于 `1/n_quantiles`
缺失分数不进入分组, 也不记为 0
某一天有效股票少于 `n_quantiles`, 或实际分组少于 2 组时, 该天不出现在结果里
`n_quantiles` 小于 2 时抛出 `ValueError`
不修改输入
结果是名为 `weight` 的 Series

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

`vol_quantile_long_short` 先按波动率把当天股票分成 `n_vol_groups` 层
每一层再用 `pd.qcut` 按分数分成 `n_quantiles` 组
最高组做多, 最低组做空
各层选出的多头合并后等权, 权重和为 1
空头合并后等权, 权重和为 -1
中间组不写入结果
重复的分位边界会被丢掉, 层数或极端组因此可能比请求的更少
波动率切不开时, 整天当作一层
分数或波动率缺失的股票不进入分组
某一层有效股票少于 `n_quantiles`, 或该层实际分组少于 2 组时, 跳过这一层
当天没有多头或没有空头时, 这一天不出现
`n_quantiles` 小于 2 时抛出 `ValueError`
不修改输入
结果是名为 `weight` 的 Series

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
