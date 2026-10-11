# 项目数据结构

QuantToolsKit 内函数接口满足:

- 数据源流出最后形态强制流经 `to_date_ticker_frame` 函数, 传回 `['date', 'ticker']` MultiIndex DataFrame
- 因子计算, 仓位构造, 收益率, 波动率, 策略比如等权买入持仓不变的 `benchmark` 的单结果返回 `pd.Series`; 多结果保留 DataFrame

# 提交信息

所有 commit 信息统一使用 `类型: 中文描述` 格式, 例如 `feat: 新增动量因子`
类型可使用 feat、fix、docs、test、refactor、chore 等英文标识；类型标记后的描述统一使用中文


# 注释、参数说明与调用示范

新增或修改公开计算函数时, 使用中文 docstring, 注释中不要出现中文句号, 说明计算口径, 并包含：

- `Args`：输入 DataFrame 的索引、必需列、参数默认值、单位及限制
- `Returns`：输出类型、索引、Series.name 或 DataFrame 列名、单位、缺失值和样本不足的行为
- `Raises`：主要参数与数据校验错误
- `Example`：自包含、可执行、具有正确预期结果的调用示例


# 源码字符规范

编辑器可能提示 U+FF0C 全角符号与常见源码字符 U+002C ASCII 符号混淆
项目源码、注释、docstring、示范与文档统一使用 ASCII 逗号, 分号, 括号等 `,`, `;`, `()`;
正文逗号后加一个空格, 保留中文文字。提及全角逗号, 分号或者括号时使用 `U+FF0C`
或转义形式 `\uFF0C`, 避免再次引入触发警告的字符
不要通过关闭 Unicode 字符提示来掩盖问题。涉及外部原始数据、解析匹配或
测试输入时, 如必须表示全角逗号, 使用转义并注明其业务用途

