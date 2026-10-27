# 收益率

实现见 [returns.py](../src/quanttoolkit/panel/returns.py), 输入格式见 [数据约定](data-structure.md)。
调用时写出参数名, 返回的是按 `[date, ticker]` 排序并带有名称的 `Series`。
简单收益 `0.1` 表示 10%; 对数收益使用自然对数。
默认价格列为 `adj_close`, 可通过 `price_col` 选择其他列。
通常把历史收益放进特征, 把未来收益单独作为标签, 再接 [样本组装](dataset.md) 或 [因子评估](metrics.md)。

## 接口与区间

`t` 是当前证券排序后的记录位置, `P[t]` 是所选列的价格。

| 接口 | 默认参数 | 计算公式 | Series 名称 |
| --- | --- | --- | --- |
| `historical_return` | `n_periods=5` | `P[t] / P[t-n_periods] - 1` | `historical_return_{n_periods}` |
| `future_returns` | `n_periods=5, gap=1` | `P[t+gap+n_periods] / P[t+gap] - 1` | `forward_return_{n_periods}_gap_{gap}` |
| `log_returns` | 无期数参数 | `log(P[t]) - log(P[t-1])` | `log_return` |

`n_periods` 按正整数传入, `gap` 按非负整数传入, 不额外校验参数。

`gap` 说明从基准记录往后数几条, 才到入场价格。例如 `gap=1, n_periods=2` 在 `t+1`
入场, 在 `t+3` 结束, 对应两个单期收益 `r[t+2]` 与 `r[t+3]`。
`gap=0` 从当天价格开始。所有期数均按每只证券的交易记录计数, 不按自然日计数。

历史收益包含当天价格, 对数收益包含当天结束的价格区间。
这两项只用到当天及更早的价格, 可以当作模型特征。
未来收益使用尚未发生的价格, 是监督模型的标签, 例如 LightGBM 要预测的持有期收益。
它可以与当天动量、波动率按索引对齐, 但必须单独作为标签列, 不能混入特征列。
标签在日期 t 用到的最后价格是 `P[t+gap+n_periods]`。
用 [split_dates](split.md) 切成 `train`, `valid`, `test` 时, 通常将 `embargo` 设为 `gap + n_periods`。
这要求标签记录与共同日历对应; 若证券缺行, 还需检查标签实际结束日期是否跨段。
样本对齐见 [assemble_samples](dataset.md)。
这里算的是两个价格之间的收益, 实际成交, 费用和组合净值还要在使用项目里接上。

## 缺失值

历史和未来持有期收益都只取起点和终点的价格; 中间价格缺失不会使端点收益失效。
对数收益需要相邻两个价格。历史/未来记录不足, 或所需端点缺失时返回 `NaN`。
不填补价格、不删除缺失记录、不补齐交易日, 空输入返回命名的空 `Series`。

## 可独立运行的示例

```python
import numpy as np
import pandas as pd

from quanttoolkit.panel.returns import future_returns, historical_return, log_returns
from quanttoolkit.panel.transfer import to_date_ticker_frame

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

用 `pd.concat(..., axis=1)` 合并多个结果, 或用 `.to_frame()` 将单个结果转为 DataFrame。
用作特征时, 应在所含价格已知后生成信号, 例如收盘价特征在收盘后才可用。

包含收益率、波动率及调仓日历的离线示例:

```bash
uv run python scripts/example_price_metrics.py
```
