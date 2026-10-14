# 分位仓位

实现见 [sizing.py](../src/quanttoolkit/sizing.py)
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

from quanttoolkit.sizing import simple_quantile_long_short

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
