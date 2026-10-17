# 训练样本

实现见 [dataset.py](../src/quanttoolkit/dataset.py)

`assemble_samples` 把特征面板和一条未来收益标签对齐成监督学习样本
索引是 `[date, ticker]`
特征列在前, 标签列在最后, 列名沿用标签 Series 的 `name`
标签用 [`future_returns`](returns.md), 名字是 `forward_return_{n_periods}_gap_{gap}`

标签必须有 `name`, 且不能与特征列同名
两侧索引名必须都是 `[date, ticker]`, `(date, ticker)` 不能重复
只保留每个特征列和标签都有值的行
只出现在特征或标签一侧的 `(date, ticker)` 不进入结果
不填补缺失值, 不修改输入
没有完整样本时, 返回保留列名和索引名的空 DataFrame

除最后一列以外的列是 LightGBM 的特征, 最后一列是要预测的持有期收益
切分用 [`split_dates`](split.md), `embargo` 取标签的 `gap + n_periods`
先组装, 再按 `train`, `valid`, `test` 取行

```python
import pandas as pd

from quanttoolkit.dataset import assemble_samples

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
