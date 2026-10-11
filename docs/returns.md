# 收益率

实现见 [returns.py](../src/quanttoolkit/returns.py), 输入格式见 [数据约定](data-structure.md)。
全部接口仅接受关键词参数, 返回排序后的 `[date, ticker]` 命名 `Series`。
`0.1` 表示 10% 收益。默认价格列为 `adj_close`, 可通过 `price_col` 选择其他列。

## 接口与区间

`t` 是当前证券排序后的记录位置, `P[t]` 是所选列的价格。

| 接口 | 默认参数 | 计算公式 | Series 名称 |
| --- | --- | --- | --- |
| `historical_return` | `n_periods=5` | `P[t] / P[t-n_periods] - 1` | `historical_return_{n_periods}` |
| `future_returns` | `n_periods=5, gap=1` | `P[t+gap+n_periods] / P[t+gap] - 1` | `forward_return_{n_periods}_gap_{gap}` |
| `log_returns` | 无期数参数 | `log(P[t]) - log(P[t-1])` | `log_return` |

`n_periods` 必须为正整数, `gap` 必须为非负整数; 接受 NumPy 整数, 不接受布尔值。
参数类型不符抛出 `TypeError`, 数值越界抛出 `ValueError`, 空输入也会校验参数。

`gap` 表示从基准记录到入场价格的距离。例如 `gap=1, n_periods=2` 在 `t+1`
入场, 在 `t+3` 结束, 对应两个单期收益 `r[t+2]` 与 `r[t+3]`。
`gap=0` 从当天价格开始。所有期数均按每只证券的交易记录计数, 不按自然日计数。

历史收益包含当天价格, 对数收益包含当天结束的价格区间。
未来收益使用尚未发生的价格, 用于事后标签; 在基准日不能作为已知特征。
这些计算不模拟成交、费用或组合净值。

## 缺失值

历史和未来持有期收益只需要起止端点价格; 中间价格缺失不会使端点收益失效。
对数收益需要相邻两个价格。历史/未来记录不足, 或所需端点缺失时返回 `NaN`。
不填补价格、不删除缺失记录、不补齐交易日, 空输入返回命名的空 `float64 Series`。

## 可独立运行的示例

```python
import numpy as np
import pandas as pd

from quanttoolkit.returns import future_returns, historical_return, log_returns
from quanttoolkit.transfer import to_date_ticker_frame

raw = pd.DataFrame(
    {
        "date": pd.bdate_range("2024-01-02", periods=5),
        "ticker": ["A"] * 5,
        "adj_close": [100.0, 110.0, 99.0, 118.8, 118.8],
    }
)
prices = to_date_ticker_frame(df=raw)
historical = historical_return(df=prices, n_periods=2)
future = future_returns(df=prices, n_periods=2, gap=1)
logs = log_returns(df=prices)

assert np.isclose(historical.iloc[2], -0.01)
assert np.isclose(future.iloc[0], 0.08)
assert np.isclose(logs.iloc[1], np.log(1.1))
assert future.iloc[-3:].isna().all()
result = pd.concat([historical, future, logs], axis=1)
print(result)
```

使用 `.to_frame()` 将单个结果转为 DataFrame, 用 `pd.concat(..., axis=1)` 合并多个结果。
结果已经是 Series, 不需要再取同名列或调用 `.stack()`。

包含收益率、波动率及调仓日历的离线示例:

```bash
uv run python scripts/example_price_metrics.py
```
