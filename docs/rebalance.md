# 调仓日历

实现见 [rebalance.py](../src/quanttoolkit/rebalance.py)。输入为共同交易日历 `DatetimeIndex`,
从行情索引的 `date` 层提取, 按排序去重后的日历记录计数, 不按每只证券自己的记录计数。

| 接口 | 返回值 | 规则 |
| --- | --- | --- |
| `rebalance_dates(dates, freq)` | `DatetimeIndex` | 选择第 0、freq、2*freq 条日期 |
| `rebalance_dates_belong(dates, freq)` | `rebalance_date Series` | 每个日期映射到最近已经发生的调仓日 |

区间为 `[本次调仓日, 下次调仓日)`, 调仓日自身开启新区间, 最后一个区间覆盖剩余日期。
不补齐交易日。两种输出都保留日期索引名称、时区和精度; 空日历返回同类型空结果。
`freq` 为正整数, 接受 NumPy 整数, 不接受布尔值; 无效频率或 `NaT` 抛出 `ValueError`,
非 `DatetimeIndex` 输入抛出 `TypeError`。空日历也必须传有效频率。

```python
import pandas as pd

from quanttoolkit.rebalance import rebalance_dates, rebalance_dates_belong

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
