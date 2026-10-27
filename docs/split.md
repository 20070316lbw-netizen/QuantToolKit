# 日期切分

实现见 [split.py](../src/quanttoolkit/model/split.py)。
`split_dates` 先给共同交易日历分段, 再从训练和验证段末尾各留出一段不用的日期, 避免标签看到下一段的价格。
返回的是以日期为索引, 名为 `split` 的 Series。需要直接取出样本表时, 用下方的 `split_samples`。

## 日期边界

输入为 `DatetimeIndex`, 内部先排序去重, 按共同日历记录计数, 不按自然日或单只证券记录计数。

| 区间 | 初始边界 | 用途 |
| --- | --- | --- |
| `train` | `date <= train_end` | 拟合模型 |
| `valid` | `train_end < date <= valid_end` | 选参数、选轮数或比较方案 |
| `test` | `date > valid_end` | 方案固定后的评估 |

`embargo` 从 `train` 和 `valid` 各自末尾丢掉指定数量的交易日, 这些日期标为缺失, 不进入任一段。
`valid` 和 `test` 的第一天保持不变, `test` 末尾不丢。
某段天数不足 `embargo` 时, 该段全部标为缺失; `embargo=0` 时按边界相邻切分。

## embargo 要空出多少天

对 `future_returns(n_periods=n, gap=g)` 标签, 通常使用 `embargo=g+n`。
当每只证券的记录与共同日历对应时, 训练标签的结束价格早于验证段首日,
验证标签的结束价格早于测试段首日。

函数只接收日期, 不读取标签的实际结束时间。证券若在共同日历中缺行,
`g+n` 条证券记录可能跨越更多共同交易日, 这时还要看标签实际结束在哪一天, 再去掉跨段样本。
输入日历应来自完整行情, 不要从已经删除缺失样本的训练表反推。

函数不修改输入或补齐日期, 输出保留日期索引名、时区和精度。
带时区日历遇到无时区边界时, 按日历时区解释边界。
空日历返回同类日期索引的空 Series。

## 可独立运行的示例

```python
import pandas as pd

from quanttoolkit.model.split import split_dates

dates = pd.bdate_range("2024-01-01", periods=8, name="date")
result = split_dates(dates, train_end="2024-01-04", valid_end="2024-01-09", embargo=1)
assert result.name == "split"
assert result.fillna("embargo").tolist() == [
    "train",
    "train",
    "train",
    "embargo",
    "valid",
    "valid",
    "embargo",
    "test",
]
```

取样本行时, 用分段结果的日期筛选 `[date, ticker]` 面板。
样本的结构见 [训练样本](dataset.md), 完整调用见 [推荐调用顺序](calling-guide.md)。

## 直接切分样本

`split_samples(samples=..., dates=..., label_end=..., train_end=..., valid_end=..., embargo=...)`
返回键为 `train`, `valid`, `test` 的样本表字典。它先按 `split_dates` 留出隔离期,
再检查标签最后看到哪一天, 去掉跨段的行: 训练标签结束日期不得晚于 `train_end`, 验证不得晚于 `valid_end`。
因此证券缺行时也会剔除跨段样本。各段允许为空, 训练函数会拒绝空训练集。

`label_end` 来自 `label_end_dates` 在完整行情索引上的结果, 参数与 `future_returns` 相同。
它按每只证券的记录位置取 `t+gap+n_periods` 的日期, 未来记录不足时为 `NaT`。
日期不在共同日历, 落在隔离期, 或标签结束日期未知的行, 都不进入任何一段。
测试段不额外限制结束日期; 需要固定测试截止日时, 先将完整行情截到该日再生成标签和结束日期。
调用示例见 [离线训练脚本](../scripts/example_model_workflow.py)。
