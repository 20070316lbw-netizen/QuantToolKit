# 因子基础函数

基础函数接受 `[date, ticker]` MultiIndex 行情, 在每只证券内按交易记录计算, 返回命名的 Series
`window` 是窗口内的单期收益数量, `skip` 是剔除最近收益的记录数, 不包含在 window 内

## 动量

[`base_momfactor`](../src/quanttoolkit/factors/mom_base.py) 计算窗口内累计对数收益率
在记录 t 上, 因子为 `log(P[t-skip] / P[t-skip-window])`, 使用截至 t-skip 的历史价格
`skip=0` 时名称为 `mom_{window}`, 否则为 `mom_{window}_{skip}`
源码 docstring 含可独立执行的示例, 测试会校验其结果

## 反转

[`base_reversalfactor`](../src/quanttoolkit/factors/reversal_base.py) 复用 `base_momfactor`, 取累计对数收益率的负值

```python
import numpy as np
import pandas as pd

from quanttoolkit.factors.reversal_base import base_reversalfactor
from quanttoolkit.transfer import to_date_ticker_frame

prices = to_date_ticker_frame(
    df=pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=5),
            "ticker": ["A"] * 5,
            "adj_close": [100.0, 110.0, 99.0, 118.8, 118.8],
        }
    )
)
factor = base_reversalfactor(prices, window=2, skip=1)
assert factor.name == "reversal_2_1"
assert factor.iloc[:3].isna().all()
assert np.isclose(factor.iloc[3], -np.log(99.0 / 100.0))
assert np.isclose(factor.iloc[4], -np.log(118.8 / 110.0))
```

在记录 t 上, 因子为 `-log(P[t-skip] / P[t-skip-window])`, 上涨得到负值, 下跌得到正值
计算会检查窗口是否具有完整收益记录, 窗口内价格缺失或记录不足时保留 NaN
不填补价格, 不年化, 不做波动率调整, 不修改输入, 不额外校验字段或参数
`skip=0` 时名称为 `reversal_{window}`, 否则为 `reversal_{window}_{skip}`
首次有效值需要 `window+skip+1` 条价格记录, 日历缺口不会补齐

## 对数波动率

[`base_logvolfactor`](../src/quanttoolkit/factors/logvol_base.py) 计算窗口内单期对数收益率的样本标准差
在记录 t 上, 因子为 `std(log(P[t-window+1] / P[t-window]), ..., log(P[t] / P[t-1]))`, `ddof=1`
名称为 `logvol_{window}`
不年化, 不填补价格, 不修改输入
窗口内对数收益缺失或记录不足时为 NaN
样本标准差至少需要 2 条对数收益, 首次有效值需要 `window+1` 条价格记录

```python
import numpy as np
import pandas as pd

from quanttoolkit.factors.logvol_base import base_logvolfactor
from quanttoolkit.transfer import to_date_ticker_frame

prices = to_date_ticker_frame(
    df=pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=4),
            "ticker": ["A"] * 4,
            "adj_close": [1.0, 2.0, 4.0, 8.0],
        }
    )
)
factor = base_logvolfactor(prices, window=2)
assert factor.name == "logvol_2"
assert factor.iloc[:2].isna().all()
assert np.isclose(factor.iloc[2], 0.0)
assert np.isclose(factor.iloc[3], 0.0)
```
