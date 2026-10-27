# 训练样本

实现见 [dataset.py](../src/quanttoolkit/model/dataset.py)。
`assemble_samples` 把两类信息拼在一起: 特征是基准日已经知道的东西, 标签是之后发生的结果。
先算 [因子特征](factors.md) 和 [未来收益标签](returns.md), 再按索引对齐, 就得到离线训练用的样本表。

## 输入与输出

| 对象 | 格式与约定 |
| --- | --- |
| `features` | `[date, ticker]` DataFrame, 每列一个特征 |
| `label` | 同类索引的 Series, 必须有 `name`, 且不能与特征列同名 |
| 返回值 | 按索引升序的 DataFrame, 特征列在前, 标签列在最后 |

两边的索引名要相同, 都使用 `[date, ticker]`, 索引键不能重复。
标签通常来自 `future_returns`, 列名为 `forward_return_{n_periods}_gap_{gap}`。
只留下两边都存在, 而且每个特征和标签都有值的行, 不修改输入或填补缺失值。
没有完整样本时, 返回保留列名和索引名的空 DataFrame。

## 与日期切分衔接

先用完整行情日历调用 [`split_dates`](split.md), 再按日期从组装结果中取出 `train`, `valid`, `test`。
日历不能只取删去缺失样本后剩余的日期。未来收益标签通常需要 `embargo=gap+n_periods`,
证券缺行时使用 `label_end_dates(prices.index, n_periods=..., gap=...)` 生成实际标签结束日期,
再交给 [`split_samples`](split.md#直接切分样本) 排除跨段行。

训练时先把最后一列取出来作标签, 剩下的列才作为特征交给模型, 避免把答案也放进输入里。
实际预测时只需准备相同列名和顺序的特征, 无需等待未来标签。
完整示例见 [模型调用顺序](calling-guide.md#加模型时-从因子面板另开一条分支)。

## 可独立运行的示例

```python
import pandas as pd

from quanttoolkit.model.dataset import assemble_samples

index = pd.MultiIndex.from_product(
    [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["A", "B"]],
    names=["date", "ticker"],
)
features = pd.DataFrame({"mom_2": [0.1, 0.2, float("nan"), 0.4]}, index=index)
label = pd.Series(
    [0.01, float("nan"), 0.03, 0.04],
    index=index,
    name="forward_return_1_gap_1",
)
samples = assemble_samples(features=features, label=label)
assert samples.columns.tolist() == ["mom_2", "forward_return_1_gap_1"]
assert samples["mom_2"].tolist() == [0.1, 0.4]
assert samples["forward_return_1_gap_1"].tolist() == [0.01, 0.04]
```
