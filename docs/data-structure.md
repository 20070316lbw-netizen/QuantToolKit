# 计算输入与输出

收益率和波动率接口接受 `pandas.DataFrame`, 索引必须是恰好两层、依次命名为
`[date, ticker]` 的 `MultiIndex`。`date` 层必须为 `DatetimeIndex`, 可以带时区。
索引键不能缺失, ticker 字符串不能为空或仅含空白, `(date, ticker)` 不能重复。

所选价格列必须存在且列名唯一。非缺失价格必须是有限、严格大于零的实数;
不自动转换数字字符串, 不接受布尔值、复数、无穷或非正数。允许 `NaN`、`pd.NA`
和 pandas 可空数值类型。无效索引或价格抛出 `ValueError`, 非 DataFrame 输入抛出 `TypeError`。

函数内部按 `[date, ticker]` 升序排序, 在每只证券内独立计算。不会修改输入,
不会删除缺失价格记录、填补价格或补齐日期。每只证券的交易记录可以不等长, 日期可以不一致。
输出为保留全部输入键、名称明确的 `float64 Series`, 计算不可用的位置保留 `NaN`。
空输入也必须具有正确索引和价格列结构, 返回空 Series。

数据源与数据库入库接口使用普通长表, `date`, `ticker` 是列。使用
[`to_date_ticker_frame`](../src/quanttoolkit/transfer.py) 转为计算格式:

```python
import pandas as pd

from quanttoolkit.transfer import to_date_ticker_frame

raw = pd.DataFrame(
    {
        "date": ["2024-01-03", "2024-01-02"],
        "ticker": ["A", "A"],
        "adj_close": [110.0, 100.0],
    }
)
# 转换器会改写输入的 date 列; 要保留原始表, 传入副本。
prices = to_date_ticker_frame(df=raw.copy())
assert prices.index.names == ["date", "ticker"]
assert prices.index.is_monotonic_increasing
```

转换器仅转换日期、设置索引和排序, 完整输入校验由计算接口执行。
历史特征与未来标签的精确区间见 [收益率](returns.md) 和 [波动率](volatility.md)。
