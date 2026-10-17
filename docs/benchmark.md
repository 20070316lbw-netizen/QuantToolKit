# 等权买入并持有

实现见 [benchmark.py](../src/quanttoolkit/portfolio/benchmark.py)

`equal_weight_hold` 用排序后第一个交易日上有价格的证券构造基准
这些证券各买入相同金额, 之后不调仓, 份额保持不变
之后才进入面板的证券不买入
组合净值是 `price / 首日价格` 的等权平均
返回值是这条净值的单期简单收益率, 名为 `equal_weight_hold`
`0.1` 表示 10%
首日收益为 NaN
已持有证券当天价格缺失时, 不填补价格, 当天和下一期收益为 NaN
不修改输入

```python
import pandas as pd

from quanttoolkit.portfolio.benchmark import equal_weight_hold

index = pd.MultiIndex.from_product(
    [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["A", "B"]],
    names=["date", "ticker"],
)
prices = pd.DataFrame({"adj_close": [1.0, 1.0, 2.0, 4.0]}, index=index)
held = equal_weight_hold(prices=prices)
assert held.name == "equal_weight_hold"
assert held.isna().iloc[0]
assert held.iloc[1] == 2.0
```
