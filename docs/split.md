# 日期切分

实现见 [split.py](../src/quanttoolkit/model/split.py)

`split_dates` 把共同交易日历分成 `train`, `valid`, `test`
比较的是排序去重后的日历, 不按自然日, 也不按单只证券自己的记录

`date <= train_end` 记为 `train`
`train_end < date <= valid_end` 记为 `valid`
`date > valid_end` 记为 `test`

`embargo` 是 `train` 和 `valid` 末尾各自丢掉的交易日数, 这些日期改为缺失
`valid` 和 `test` 的第一天保持不变, `test` 的末尾不丢
丢掉的日期位于最终的上一段最后一天和下一段第一天之间, 两段都不用
`embargo` 取未来收益标签的 `gap + n_periods`
训练日标签用到的最后价格因此仍早于 `valid` 的第一天
某一段的天数少于 `embargo` 时, 这一段全部变为缺失
`embargo=0` 时按边界相邻切分
不修改输入, 不补齐交易日
空日历返回名为 `split` 的空 Series, 并保留索引名, 时区和精度

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
