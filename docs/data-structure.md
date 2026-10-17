# 计算输入与输出

收益率和波动率接口接受 `pandas.DataFrame`, 索引必须是恰好两层、依次命名为
`[date, ticker]` 的 `MultiIndex`。`date` 层必须为 `DatetimeIndex`, 可以带时区。
索引键不能缺失, ticker 字符串不能为空或仅含空白, `(date, ticker)` 不能重复。

所选价格列应为数值类型, 非缺失价格应为有限正数, 缺失价格使用 `NaN`。
计算接口直接取列计算, 不额外校验索引、价格或参数, 输入由调用方保证符合约定。

函数内部按 `[date, ticker]` 升序排序, 在每只证券内独立计算。不会修改输入,
不会删除缺失价格记录、填补价格或补齐日期。每只证券的交易记录可以不等长, 日期可以不一致。
入库脚本会先调用 [`fill_price_gaps`](../src/quanttoolkit/panel/prices.py), 把每只证券首尾之间缺失的交易日写入 `prices`, 价格沿用前一条记录, volume 为 0。计算函数读到的是补齐后的面板, 自身仍然不补日期。
输出为保留全部输入键、名称明确的 `Series`, 计算不可用的位置保留 `NaN`。
空输入也必须具有正确索引和价格列结构, 返回空 Series。

时序数据源输出通过 [`to_date_ticker_frame`](../src/quanttoolkit/panel/transfer.py) 转为计算格式。
Yahoo 行情可直接用于计算, 入库时用 `.reset_index()` 转为 `prices` 表对应的长表。
读取数据库或其他长表后, 使用同一转换函数:

```python
import pandas as pd

from quanttoolkit.panel.transfer import to_date_ticker_frame

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

转换器仅转换日期、设置索引和排序, 输入是否符合约定由调用方保证。
历史特征与未来标签的精确区间见 [收益率](returns.md) 和 [波动率](volatility.md)。
