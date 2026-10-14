# 评定指标

实现见 [metrics.py](../src/quanttoolkit/metrics.py)。
因子指标接受两列按 `[date, ticker]` 对齐的 Series, 收益指标接受一条简单收益率 Series。
接口仅接受关键词参数。因子相关按日期升序返回命名 Series, 收益指标返回一个浮点数。
`0.1` 表示 10% 收益或 0.1 的相关系数。这些计算不模拟成交、费用或组合约束。

## 接口

| 接口 | 输入 | 计算 | 结果 |
| --- | --- | --- | --- |
| `ic` | 因子, 未来收益 | 每个日期的 Pearson 相关 | Series `ic` |
| `rank_ic` | 因子, 未来收益 | 有效样本平均秩的 Pearson 相关 | Series `rank_ic` |
| `ic_summary` | `ic` 或 `rank_ic` | 均值, `ddof=1` 标准差, 信息比率 | Series `ic_summary` |
| `total_return` | 简单收益 | `prod(1+r)-1` | 小数收益率 |
| `sharpe` | 简单收益 | `mean(r)/std(r, ddof=1)` | 不年化夏普 |
| `max_drawdown` | 简单收益 | 净值从 1 起算的最小 `净值/峰值-1` | 0 或负数 |
| `win_rate` | 简单收益 | 收益严格大于 0 的比例 | 小数比例 |

截面相关只使用两边索引都存在的 `(date, ticker)`。
同一日期内成对去掉缺失值; 有效证券少于 2 只, 或因子、收益没有波动时, 该日期为 `NaN`。
两边没有交集的日期不出现。`rank_ic` 的并列取值使用平均秩。

`ic_summary` 忽略无法计算的日期。有效日期少于 2 个时, 标准差和信息比率为 `NaN`。
标准差为 0 且均值非 0 时, 信息比率为 `inf`; 均值和标准差都为 0 时为 `NaN`。
信息比率不年化。

收益指标先按索引升序排列, 再计算。调用方传入已经排好的时间索引。
任一收益为 `NaN` 或输入为空时, 四个收益指标都返回 `NaN`, 不会删掉缺失收益后继续算。
夏普的无风险利率按 0, 不年化; 常数非零收益的夏普为 `inf`, 常数零收益为 `NaN`。
最大回撤把第一笔收益之前的峰值记为 1, 所以首笔亏损也会形成回撤。
胜率不把 0 收益计为获胜。

## 可独立运行的示例

```python
import numpy as np
import pandas as pd

from quanttoolkit.metrics import (
    ic,
    ic_summary,
    max_drawdown,
    rank_ic,
    sharpe,
    total_return,
    win_rate,
)

index = pd.MultiIndex.from_product(
    [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["A", "B", "C"]],
    names=["date", "ticker"],
)
factor = pd.Series([1.0, 2.0, 3.0, 1.0, 2.0, 4.0], index=index)
forward = pd.Series([0.1, 0.2, 0.3, 0.0, 0.1, 0.1], index=index)
score = pd.concat(
    [
        ic(factor=factor, forward_return=forward),
        rank_ic(factor=factor, forward_return=forward),
    ],
    axis=1,
)
assert score.loc[pd.Timestamp("2024-01-02")].tolist() == [1.0, 1.0]
assert np.isclose(score.loc[pd.Timestamp("2024-01-03"), "ic"], 2 / np.sqrt(7))
assert np.isclose(score.loc[pd.Timestamp("2024-01-03"), "rank_ic"], np.sqrt(0.75))

summary = ic_summary(ic=pd.Series([1.0, -1.0]))
assert summary.name == "ic_summary"
assert summary["ic_mean"] == 0
assert np.isclose(summary["ic_std"], np.sqrt(2))
assert summary["ic_ir"] == 0

returns = pd.Series([0.5, -0.5], index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
assert total_return(returns=returns) == -0.25
assert max_drawdown(returns=returns) == -0.5
assert win_rate(returns=pd.Series([0.1, -0.2, 0.0, 0.3])) == 0.5
assert np.isclose(sharpe(returns=pd.Series([0.1, 0.3])), np.sqrt(2))
```
