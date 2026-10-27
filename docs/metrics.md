# 评估指标

实现见 [metrics.py](../src/quanttoolkit/metrics.py)。
先看我们要评估什么: 想知道当天分数和之后的收益是否一致, 用因子指标;
已经算出组合逐期收益, 再用收益指标。
因子指标接受两条按 `[date, ticker]` 对齐的 Series, 收益指标接受一条按日期索引的简单收益率 Series。
接口仅接受关键词参数。每日相关和汇总返回命名 Series, 收益指标返回浮点数。
收益 `0.1` 表示 10%; 相关系数没有收益率单位, 不按百分收益解读。

## 接口速查

| 接口 | 输入 | 计算 | 结果 |
| --- | --- | --- | --- |
| `ic` | 因子, 未来收益 | 每个日期的 Pearson 相关 | Series `ic` |
| `rank_ic` | 因子, 未来收益 | Spearman 相关, 并列取平均秩 | Series `rank_ic` |
| `ic_summary` | `ic` 或 `rank_ic` | 均值, `ddof=1` 标准差, 信息比率, 大于 0 的比例 | Series `ic_summary` |
| `total_return` | 简单收益 | `prod(1+r)-1` | 小数收益率 |
| `sharpe` | 简单收益 | `mean(r)/std(r, ddof=1)` | 不年化夏普 |
| `max_drawdown` | 简单收益 | 净值从 1 起算的最小 `净值/峰值-1` | 0 或负数 |
| `win_rate` | 简单收益 | 收益严格大于 0 的比例 | 小数比例 |

## 截面相关与汇总

常用顺序是 `rank_ic(factor=score, forward_return=label)` → `ic_summary(ic=daily_ic)`。
截面相关只使用两边索引都存在的 `(date, ticker)`。
同一日期内成对去掉缺失值; 有效证券少于 2 只, 或因子、收益没有波动时, 该日期为 `NaN`。
两边没有交集的日期不出现。`rank_ic` 的并列取值使用平均秩。

`ic_summary` 忽略无法计算的日期。有效日期少于 2 个时, 标准差和信息比率为 `NaN`。
`ic_pos` 是有效日期里信息系数严格大于 0 的比例。没有有效日期时为 `NaN`。
标准差为 0 且均值非 0 时, 信息比率为 `inf`; 均值和标准差都为 0 时为 `NaN`。
`ic_ir = ic_mean / ic_std`, 不年化, 描述每日 IC 的均值相对于波动的大小。
要判断均值是否显著, 还需另做检验。多期未来收益标签若逐日生成, 区间通常重叠,
不能把每天的 IC 直接视为相互独立的观测。

## 收益路径与缺失值

收益指标先按索引升序排列, 再计算。传入时要有能按时间排序的索引,
并保持收益频率一致。输入应为有限的简单收益率, 无需预先排序。
任一收益为 `NaN` 或输入为空时, 四个收益指标都返回 `NaN`, 不会删掉缺失收益后继续算。
夏普的无风险利率按 0, 不年化; 常数非零收益的夏普为 `inf`, 常数零收益为 `NaN`。
算最大回撤时, 第一笔收益之前的净值和峰值都是 1, 所以第一笔就亏损也会记为回撤。
胜率不把 0 收益计为获胜。

因子分数、目标权重和对数收益不能直接作为这里的组合简单收益。
逐日生成的重叠多期收益也不能直接复利或计算净值回撤。
[买入持有基准](benchmark.md) 的首日收益为 NaN, 可显式移除这个没有前期净值的首日;
中间缺失收益应先排查, 避免直接删除后改变收益路径。

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
assert summary["ic_pos"] == 0.5

returns = pd.Series([0.5, -0.5], index=pd.to_datetime(["2024-01-02", "2024-01-03"]))
assert total_return(returns=returns) == -0.25
assert max_drawdown(returns=returns) == -0.5
assert win_rate(returns=pd.Series([0.1, -0.2, 0.0, 0.3])) == 0.5
assert np.isclose(sharpe(returns=pd.Series([0.1, 0.3])), np.sqrt(2))
```
