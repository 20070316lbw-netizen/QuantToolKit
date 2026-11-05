# 研究和回测图表

调用 `plot_report` 就能按流程生成一张共享时间轴的报告。给 `output` 传文件路径即可保存为 PNG 或 PDF, 目录需要已存在。

| 流程 | 固定图表 | 有对应产物时附加 |
| --- | --- | --- |
| `vectorized` 向量化回测 | 净值和回撤, 输入 `portfolio_returns` | 调仓日的总敞口和净敞口, 输入 `target_weights` |
| `event_driven` 事件驱动回测 | 净值和回撤, 输入 `portfolio_returns` | 逐笔成交滑点, 输入 `slippage_bps` |
| `factor_research` 因子研究 | 每日 IC 或 Rank IC, 输入 `ic` | 远期多空收益差, 输入 `forward_spread` |

报告会拒绝缺失的必需产物, 也会拒绝传给当前流程的其他产物。例如向量化报告接受目标权重, 逐笔成交滑点属于事件驱动报告。
当前两种回测由调用方提供结果。外部组合收益先用 `prepare_portfolio_returns` 接入, 收益的频率和含义由调用方保证。

## 数据怎么传

- `portfolio_returns` 是按期记录的组合简单收益率 Series, 索引是唯一的 `DatetimeIndex`, 值为小数且没有缺失。各期连续且不重叠。若使用 `equal_weight_hold` 的结果, 先显式切掉表示没有前期净值的第一行后再接入。净值从 1 起算, 回撤的初始峰值也是 1
- `target_weights` 是 `[date, ticker]` MultiIndex Series, 权重为小数, 空头为负。可以直接接 `simple_quantile_long_short` 的输出。敞口只展示目标权重, 总敞口是绝对值之和, 净敞口是带符号的和
- `slippage_bps` 是按成交时间索引的 Series, 一笔成交一个值, 同一时刻允许多笔。传已经算好的滑点, 1 bps 为 0.01%, 正值表示买入更贵或卖出更便宜。传入带日期索引的空数值 Series 时显示 `No fills`
- `ic` 和 `forward_spread` 都是日期索引唯一的 Series, 允许缺失并在图上留空。远期多空收益差可能有重叠持有期, 用独立时序图展示各次观测值

## 生成整张报告

下面用手工样本生成向量化回测报告, 保存为 `vectorized.png`。

```python
import pandas as pd

from quanttoolkit.figure import plot_report
from quanttoolkit.metrics import prepare_portfolio_returns

dates = pd.bdate_range("2024-01-02", periods=4, name="date")
returns = pd.Series([0.01, -0.02, 0.015, 0.005], index=dates)
index = pd.MultiIndex.from_product([dates[::2], ["A", "B"]], names=["date", "ticker"])
weights = pd.Series([0.6, -0.4, 0.5, -0.5], index=index)

figure = plot_report(
    "vectorized",
    portfolio_returns=prepare_portfolio_returns(returns),
    target_weights=weights,
    output="vectorized.png",
)
assert len(figure.axes) == 3
```

事件驱动回测用 `plot_report("event_driven", portfolio_returns=prepare_portfolio_returns(returns), slippage_bps=slippage, output="event.png")`。
因子研究用 `plot_report("factor_research", ic=daily_ic, forward_spread=spread, output="factor.png")`。

每张图也能单独调用。例如 `ax = plot_drawdown(prepare_portfolio_returns(returns))` 返回 Matplotlib 的 `Axes`, 可以用 `ax.figure.savefig("drawdown.png")` 保存。
已有坐标轴时, 传入 `ax=` 即可拼进自己的布局。
