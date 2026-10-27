# 调仓日历

实现见 [rebalance.py](../src/quanttoolkit/portfolio/rebalance.py)。
先从大家共用的交易日历里选出调仓日, 再把每一天对应到最近一次已经发生的调仓。
输入为从行情索引 `date` 层提取的 `DatetimeIndex`, 按排序去重后的日历记录计数,
不按自然日或每只证券自己的记录计数。

## 接口与区间

| 接口 | 返回值 | 规则 |
| --- | --- | --- |
| `rebalance_dates(dates, freq)` | `DatetimeIndex` | 选择第 0、freq、2*freq 条日期 |
| `rebalance_dates_belong(dates, freq)` | `rebalance_date Series` | 每个日期映射到最近已经发生的调仓日 |

区间为 `[本次调仓日, 下次调仓日)`, 调仓日当天就属于新的区间, 最后一个区间覆盖剩余日期。
不补齐交易日。两种输出都保留日期索引名称、时区和精度; 空日历返回同类型空结果。
`freq` 按正整数传入, 日历应为不含 `NaT` 的 `DatetimeIndex`。
接口直接排序、去重并选择日期, 不额外校验输入。

调仓节奏以传入日历的首日为起点, `freq=5` 表示每隔 5 条交易日记录,
不表示固定每周某一天。先截取评估区间还是先生成调仓日, 可能得到不同的调仓节奏。

## 可独立运行的示例

```python
import pandas as pd

from quanttoolkit.portfolio.rebalance import rebalance_dates, rebalance_dates_belong

dates = pd.DatetimeIndex(
    ["2024-01-08", "2024-01-02", "2024-01-03", "2024-01-02"], name="date"
)
selected = rebalance_dates(dates, freq=2)
belongs = rebalance_dates_belong(dates, freq=2)
assert selected.strftime("%Y-%m-%d").tolist() == ["2024-01-02", "2024-01-08"]
assert belongs.dt.strftime("%Y-%m-%d").tolist() == [
    "2024-01-02",
    "2024-01-02",
    "2024-01-08",
]
```

构造组合时, 用 `rebalance_dates` 筛选调仓日分数, 再调用 [分位仓位](sizing.md)。
`rebalance_dates_belong` 只告诉我们每一天属于哪次调仓, 不会替信号延迟或模拟成交;
使用收盘信号时, 由使用方确定目标仓位何时生效。完整流程见 [推荐调用顺序](calling-guide.md)。
