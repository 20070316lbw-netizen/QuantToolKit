# 等权买入并持有

实现见 [benchmark.py](../src/quanttoolkit/portfolio/benchmark.py)。
`equal_weight_hold` 用来做一个买入持有基准: 首日给每只有价格的证券分配相同金额,
之后一直持有最初买到的份额。默认使用 `adj_close`, 想换价格列时传 `price_col`。

## 持仓与收益口径

证券范围取排序后第一个交易日上价格非缺失的证券, 后续才出现的证券不买入。
首日买入金额相同, 之后哪只股票涨得多, 它占组合的比重就会变大, 不会每天调回等权。
组合净值为这些证券 `price / 首日价格` 的等权平均。

返回值是组合净值的单期简单收益率 Series, 以日期为索引, 名为 `equal_weight_hold`。
`0.1` 表示 10%, 首日因没有前期净值而为 NaN。
已持有证券某日价格缺失时, 当天和下一期收益均为 NaN; 函数不填价格或修改输入。

## 可独立运行的示例

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

## 与策略对照

先截出要比较的行情区间, 再计算基准, 这样首日买哪些股票, 从哪一天开始算净值, 才和评估区间对应。
策略与基准应对齐时间区间和收益频率。交给 [收益指标](metrics.md) 前,
可显式移除没有前期净值的首日; 中间缺失收益应先查原因。
本基准不模拟交易费用, 股票池由输入面板决定。
