# 数据约定

先分清表要交给谁: 行情和因子计算通常用 `[date, ticker]` 面板, 写入数据库前再把索引还原成普通列。
完整的组合调用见 [推荐调用顺序](calling-guide.md)。

## 先区分三种表

| 数据 | 索引 | 示例 |
| --- | --- | --- |
| 行情、因子和证券分数 | 两层 MultiIndex `[date, ticker]` | `get_prices` 输出, 因子面板, 仓位 |
| 全市场日频序列 | 日期索引 `date` | `read_french_factors` 输出, 每日 IC, 基准收益 |
| 静态资料或数据库长表 | 普通索引, 标识字段作为列 | `universe`, SQL 查询后的 `.df()`, 入库前的行情 |

静态股票池无需添加日期索引。French 因子是一日一行的市场序列, 无需添加 ticker 层。

## 行情计算的输入

收益率、波动率和基础价格因子接受 `pandas.DataFrame`。
索引有两层, 按顺序命名为 `[date, ticker]`; `date` 层为 `DatetimeIndex`, 可以带时区。
索引键不能缺失, ticker 字符串不能为空或仅含空白, `(date, ticker)` 不能重复。

所选价格列应为数值类型, 非缺失价格应为有限正数, 缺失价格使用 `NaN`。
价格应使用一致的复权口径。收益率和基础价格因子默认选择 `adj_close`,
波动率接口默认选择 `close`, 混用时应显式指定 `price_col`。
这些函数会直接取价格列计算, 不再检查索引, 价格和参数, 所以调用前要自己准备好符合约定的数据。

计算按 `[date, ticker]` 升序排序, 在每只证券内独立进行, 不修改输入。
只算一项时返回带名称的 Series, 保留全部输入键; 历史不足或所需价格缺失时保留 `NaN`。
空输入也要具有正确的索引和列结构, 才能返回对应的空结果。
仓位构造和样本组装会按各自规则筛选记录, 具体见 [仓位](sizing.md) 和 [训练样本](dataset.md)。

## 长表与面板之间转换

Yahoo 的 `get_prices` 已返回标准行情面板, 可以直接计算。
SQL 查询或其他长表读入后, 使用 [`to_date_ticker_frame`](../src/quanttoolkit/panel/transfer.py)
转换日期、设置索引并排序:

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
# 转换器会改写输入的 date 列, 要保留原始表时传入副本
prices = to_date_ticker_frame(df=raw.copy())
assert prices.index.names == ["date", "ticker"]
assert prices.index.is_monotonic_increasing

# 写入数据库前, 把索引还原为普通列
stored = prices.reset_index()
assert stored.columns.tolist() == ["date", "ticker", "adj_close"]
```

这里只负责转日期, 设索引和排序, 不检查价格, 也不补日期或删缺失记录。
单个计算结果用 `.to_frame()` 转为 DataFrame, 多个结果用 `pd.concat(..., axis=1)` 合并;
结果已经是 Series 时, 无需再取同名列或调用 `.stack()`。

## 交易记录与缺失日期

收益与因子窗口按每只证券自己的交易记录计数, 不按自然日计数。
调仓与日期切分使用共同日历。各证券记录可以不等长, 日期也可以不一致。

价格计算自身不补日期或填价格。仓库的行情入库脚本会先调用
[`fill_price_gaps`](../src/quanttoolkit/panel/prices.py),
按输入里出现过的日期补齐每只证券首尾记录之间的缺行。
新增行沿用此前可用价格, volume 为 0; 已有行保留原值, 不修补已有行中的 NaN。
首条记录之前和末条记录之后不补, 所有证券都缺失的日期也不会凭空生成。

因此, 直接使用下载行情与使用补齐后入库的行情, 可能得到不同的滚动窗口结果。
选定补齐方式后, 训练, 评估和实际使用都沿用同一套规则。
历史特征与未来标签的精确区间见 [收益率](returns.md) 和 [波动率](volatility.md)。
