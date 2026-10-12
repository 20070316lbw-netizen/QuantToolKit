# 波动率

实现见 [volatility.py](../src/quanttoolkit/volatility.py), 输入格式见 [数据约定](data-structure.md)。
接口仅接受关键词参数, 返回排序后的 `[date, ticker]` 命名 `Series`。
默认价格列是 `close`; 要使用复权价格, 显式传 `price_col="adj_close"`。

## 接口与区间

设当前证券的单期简单收益为 `r[t] = P[t] / P[t-1] - 1`。

| 接口 | 默认参数 | 计算区间 | Series 名称 |
| --- | --- | --- | --- |
| `history_vol` | `window=21` | `std(r[t-window+1], ..., r[t])` | `volatility_{window}` |
| `forward_volatility` | `window=21, gap=1` | `std(r[t+gap+1], ..., r[t+gap+window])` | `forward_volatility_{window}_gap_{gap}` |

均使用样本标准差 `ddof=1`, 单位为小数波动率, 不做年化。
`window` 为收益率数量, 至少为 2; 计算一个窗口需要 `window+1` 个连续有效价格。
`gap` 为从基准记录到区间起点价格的距离, 为非负整数。
参数按整数传入, 不额外校验参数。

`gap` 与 [未来收益率](returns.md#接口与区间) 使用同一含义。
`gap=1, window=2` 使用 `P[t+1]` 至 `P[t+3]`, 标准差计算 `r[t+2]` 和 `r[t+3]`;
不包含从 `P[t]` 到 `P[t+1]` 的收益。
`gap=0` 则从当天价格开始, 使用 `r[t+1]` 和 `r[t+2]`。

历史波动率包含当天结束的单期收益, 首次有效值出现在每只证券的第 `window+1` 条价格记录。
未来波动率使用未来数据, 只能作为事后标签; 当 `gap+window` 条未来记录不足时为 `NaN`。
窗口按每只证券的交易记录计数, 不补齐自然日或共同交易日历。

## 缺失值

窗口内任何价格缺失都会使相关单期收益缺失, 整个窗口返回 `NaN`。
不前向填补、不忽略缺失收益来缩短窗口。完整有效窗口恢复后会重新得到结果。
基准日价格不在未来持有区间内时, 该价格缺失不会影响该标签。
常数收益窗口的波动率为 0, 空输入返回命名的空 `Series`。

## 可独立运行的示例

```python
import numpy as np
import pandas as pd

from quanttoolkit.transfer import to_date_ticker_frame
from quanttoolkit.volatility import forward_volatility, history_vol

prices = to_date_ticker_frame(
    df=pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=5),
            "ticker": ["A"] * 5,
            "adj_close": [100.0, 110.0, 99.0, 118.8, 118.8],
        }
    )
)
historical = history_vol(df=prices, window=2, price_col="adj_close")
future = forward_volatility(df=prices, window=2, gap=1, price_col="adj_close")

# 历史窗口在第三条记录使用 0.1 和 -0.1。
assert np.isclose(historical.iloc[2], 0.2 / np.sqrt(2))
# 基准日未来窗口使用 -0.1 和 0.2。
assert np.isclose(future.iloc[0], 0.3 / np.sqrt(2))
assert future.iloc[-3:].isna().all()
print(pd.concat([historical, future], axis=1))
```

样本标准差口径与 [pandas Rolling.std](https://pandas.pydata.org/docs/reference/api/pandas.api.typing.Rolling.std.html) 一致。
