# 函数注册表

注册表按公开接口记录所属阶段、适用流程、业务定义、各参数的要求与允许上游, 以及返回内容。接口仍按原来的模块导入和调用, 可以自由组合 pandas 操作和外部研究代码。

```python
from quanttoolkit.registry import get_contract, list_contracts

spec = get_contract("panel.returns.future_returns")
assert spec["stage"] == "标签计算"
assert spec["inputs"]["df"]["source_policy"] == "required"
assert "sources.yahoo.get_prices" in spec["inputs"]["df"]["upstream"]

charts = list_contracts(stage="报告", workflow="event_driven")
assert "figure.plots.plot_execution_slippage" in charts
```

名称使用 `quanttoolkit` 下的模块和函数名, 类方法加类名。每条记录的 `definition` 解释计算口径, `inputs` 按参数记录 `requirements` 和 `upstream`, `output` 描述返回内容。
`workflows` 列出已明确归类的 `vectorized`、`event_driven` 或 `factor_research`。通用数据和模型接口可以只登记阶段。
查询返回副本, 修改查询结果不会改动运行时规则。

## 哪些上游会在运行时检查

| 来源规则 | 含义 | 示例 |
| --- | --- | --- |
| `required` | 当前传入的对象必须由允许的项目内入口返回 | 收益率和波动率的行情, 样本切分的标签结束日期, 净值图的组合收益 |
| `if_known` | 数据来自已登记接口时检查来源, 允许调用方直接传外部数据 | IC 的分数和未来收益, 组合绩效指标, 因子函数的行情 |
| `described` | 列出可接的项目内方法, 无额外来源限制 | 外部 LightGBM 模型或特征表 |

格式校验与来源约束分别设置。例如价格计算会检查索引键和本次选中的价格列, 未选中的列可以留给其他计算。少量函数继续使用自己的业务校验, 如模型是否已经拟合。
来源按当前 Python 进程里的对象记录。复制、切片、拼接或写库后重读会得到新对象, 严格入口需要重新接入。格式或来源不符时, 错误会给出函数、参数和允许的上游。

- 已有 `[date, ticker]` 索引的外部行情用 `prepare_price_panel` 接入, 含日期和证券普通列的长表用 `to_date_ticker_frame` 转换
- `split_samples` 的结束日期用 `label_end_dates` 生成, 或用 `prepare_label_end_dates` 接入外部日期。标签与结束日期都由库内生成时, 样本组装和切分会核对已记录的标签期限。已保存的样本重读后可以按面板格式切分
- 净值和回撤图接收 `prepare_portfolio_returns` 登记的外部组合单期收益。回测引擎由调用方选择, 调用方负责保证各期收益连续且不重叠。`prepare_forward_spread` 接入允许重叠的远期多空价差, 可以用于因子研究和观测胜率
- [报告清单](figure.md) 决定每个流程的必需图与附加图, 各单图按自己的接口契约检查输入

## 新增接口

在 [接口目录](../src/quanttoolkit/_catalog.py) 增加条目, 输入参数名和函数签名一致, 上游使用目录里已有的接口名。
需要来源检查、共用格式检查或记录产物时, 在函数入口加 `@_contract`。新增报告图时在图表条目登记流程, 并把图加入对应的报告清单。
