# 模型

实现见 [model](../src/quanttoolkit/model)。
先把历史特征和未来标签 [拼成样本](dataset.md), 再按 [日期切分](split.md)
取出训练, 验证和测试集。训练集用来拟合, 验证集用来选方案, 方案固定后再看测试集。
从数据开始的完整调用见 [推荐调用顺序](calling-guide.md)。

## 接口速查

| 工具 | 输入 | 使用方式 |
| --- | --- | --- |
| `RegressionTree` | 二维特征数组 `X`, 一维数值标签 `y` | `fit` → `predict`, 可用 `apply` 查看叶子编号 |
| `SimpleGBDT` | 同上, 可另传验证集 | `fit` → `predict`, 可按验证误差挑选轮数 |
| `LightGBMParams` | 回归参数 | `fit_lightgbm` → `predict_scores`, 支持验证早停 |
| `LambdaRankParams` | 排序参数 | 配合 `relevance_by_date` → `fit_lambdarank` → `predict_scores` |
| `TrainLog` | 模型名、参数字典、指标字典 | 保存或读取 JSON 训练记录 |

预测时仍按训练时的列数和顺序传入特征。数组模型返回数组, 需要自己把结果装回原样本索引的 Series;
`predict_scores` 会检查训练特征的列名和顺序, 直接返回与输入特征同索引的 `pred` Series。
完整流程见 [离线示例](../scripts/example_model_workflow.py), 在仓库根目录运行
`uv run python scripts/example_model_workflow.py`。

## 模型参数

`ModelParams` 提供共用随机种子 `seed`, 默认 7, 各 LightGBM 参数类继承并补充自己的字段。
各参数类用 dataclass 保存字段, 也能写成 JSON; `to_dict()` 返回字段字典,
`save(path)` 和对应参数类的 `load(path)` 写入和读回 JSON。路径使用 `Path`, 父目录需要已存在。

`LightGBMParams` 用于回归, 标签通常是 `assemble_samples` 最后一列的未来收益。
`num_boost_round` 和 `early_stopping_rounds` 属于训练循环, `booster_params()` 不包含它们,
`fit_lightgbm` 会设置训练轮数, 有验证集时按验证指标早停。它会将 `seed` 复制到 LightGBM 的用于特征抽样, 行抽样和数据处理的三个随机种子,
其余未写出的参数沿用 LightGBM 默认值。无验证集或 `early_stopping_rounds=0` 时按指定轮数训练。

`LambdaRankParams` 用于每日截面排序, `booster_params()` 只包含
`objective="lambdarank"` 和随机种子。学习率、叶子数、评价指标和 `label_gain` 沿用
LightGBM 默认值, 因而相关度标签需落在 `0` 到 `30`。默认训练 100 轮, 不做早停。

参数用法和相关度转换示例:

```python
import pandas as pd

from quanttoolkit.model.lambdarank import LambdaRankParams
from quanttoolkit.model.lightgbm import LightGBMParams
from quanttoolkit.model.relevance import relevance_by_date

params = LightGBMParams(num_leaves=15, seed=3)
booster = params.booster_params()
assert params.num_boost_round == 200
assert "num_boost_round" not in booster
assert booster["num_leaves"] == 15
assert booster["data_random_seed"] == 3

rank = LambdaRankParams(seed=3)
rank_booster = rank.booster_params()
assert rank.num_boost_round == 100
assert rank_booster["objective"] == "lambdarank"
assert "learning_rate" not in rank_booster
assert "num_leaves" not in rank_booster
assert rank_booster["data_random_seed"] == 3

index = pd.MultiIndex.from_product(
    [pd.to_datetime(["2024-01-02"]), ["A", "B", "C", "D"]],
    names=["date", "ticker"],
)
label = pd.Series([0.1, 0.4, 0.2, 0.3], index=index)
assert relevance_by_date(label, n_grades=4).tolist() == [0, 3, 1, 2]
```

## 回归训练

`fit_lightgbm` 从 `quanttoolkit.model.lightgbm` 导入, 接收训练特征 `features` 和收益标签 `label`。
可选的 `valid_features` 和 `valid_label` 必须同时提供, 特征列名和顺序须与训练集一致。
函数按特征索引对齐标签, 特征 NaN 交给 LightGBM 处理, 标签须为有限数值。
默认最多 200 轮, 有验证集时连续 20 轮未改善就早停; 无验证集或
`early_stopping_rounds=0` 时按指定轮数训练。返回 LightGBM Booster,
预测可以直接调用 `booster.predict(features, validate_features=True)`。

训练前用 [`split_samples`](split.md#直接切分样本) 切分样本, 验证集负责早停和选型,
固定方案后才评估测试集。实际预测只需要相同列名和顺序的历史特征, 无需未来标签。

## 排序训练与预测

`relevance_by_date(label, n_grades=31)` 在每个日期内按未来收益排序并映射为整数档,
收益越高档位越高。并列收益按输入行顺序拆开, 不取平均秩; 缺失标签保持缺失。
使用默认 `label_gain` 时, `n_grades` 不应超过 31。

`fit_lambdarank(features=..., relevance=..., params=...)` 接收 `[date, ticker]`
特征表和对齐的相关度标签, 内部按索引排序、按日期分组, 返回 LightGBM Booster。
`predict_scores(booster=..., features=...)` 保留在 `quanttoolkit.model.lambdarank`,
排序模型返回连续排序分数, 回归模型返回与标签同单位的预测值。
预测分数可直接接 [IC 评估](metrics.md) 和 [仓位构造](sizing.md)。

`group_sizes(index)` 数出每天有多少行, 告诉 LightGBM 哪些股票放在一起比较, 所以同一天的行要连在一起;
调用 `fit_lambdarank` 时无需手动准备它。

## 手写回归树

`RegressionTree` 拟合平方误差。输入是二维数值数组 `X` 和一维标签 `y`,
样本数相同, 取值必须有限; 每次 `fit` 都重新建树。

建树时, `X[:, feature] <= threshold` 的样本走左边, 其余走右边。候选阈值取这一列去重后的值,
去掉最大值以保证右侧不空。增益严格更大时才更新最优分裂,
相同时保留更早的列和更小的阈值。
`max_depth` 是从根节点开始允许的分裂次数, `1` 表示只分裂一次。
样本少于 `min_samples_split`, 或两侧无法各自留下 `min_samples_leaf` 条样本时, 当前节点成为叶子。

叶子权重为 `-sum(g) / (sum(h) + lambda_l2)`。
普通回归中梯度 `g=-y`, 海森 `h=1`, 因而 `lambda_l2=0` 时预测值就是叶内标签均值;
正则项大于 0 时, 按 `sum(y) / (n + lambda_l2)` 向 0 收缩。

`predict` 返回叶子权重。`apply` 返回叶子编号, 范围为 `[0, n_leaves)`, 按先左后右顺序编号。
编号只对应当前这棵树, 重新拟合后不保证对应同一区域。

```python
import numpy as np

from quanttoolkit.model import RegressionTree

X = np.array([[0.0], [1.0], [2.0], [3.0]])
y = np.array([1.0, 1.0, 3.0, 3.0])

tree = RegressionTree(max_depth=1).fit(X, y)
assert tree.n_leaves == 2
assert tree.predict(X).tolist() == [1.0, 1.0, 3.0, 3.0]
assert tree.apply(X).tolist() == [0, 0, 1, 1]

shrunk = RegressionTree(max_depth=1, lambda_l2=1).fit(X, y)
assert np.allclose(shrunk.predict(X), [2 / 3, 2 / 3, 2.0, 2.0])
```

## 简化 GBDT

`SimpleGBDT` 从全 0 预测开始, 逐轮用 `RegressionTree` 拟合平方误差的梯度。
梯度为当前预测减去标签, 海森为 1, 每棵新树的输出乘以 `lr` 后累加。
因此 `n_trees=1, lr=1` 时与一棵 `RegressionTree` 等价。
默认 `n_trees=100`, `max_depth=3`, `lr=0.1`。

`X_val` 和 `y_val` 必须同时给出, 特征列数和顺序与训练集一致。
有验证集时, 每增加一棵树计算一次验证 MSE, 严格变小才算改善。
训练结束后, 找到验证 MSE 最低的那一轮, 只留下截至这一轮的树, 全 0 的初始预测不参与比较。

| 属性或参数 | 含义 |
| --- | --- |
| `best_iteration` | 保留的树数, 从 1 开始; 无验证集时为 `None` |
| `best_val_loss` | 保留模型的验证 MSE; 无验证集时为 `None` |
| `early_stopping_rounds=None` | 跑完指定轮数, 有验证集时再保留到最佳轮数 |
| `early_stopping_rounds` 为正整数 | 必须提供验证集, 连续指定轮数未改善后停止并保留到最佳轮数 |

无验证集时保留全部树。再次 `fit` 会清空之前的树和验证记录。

```python
import numpy as np

from quanttoolkit.model import SimpleGBDT

X = np.array([[0.0], [1.0], [2.0], [3.0]])
y = np.array([1.0, 1.0, 3.0, 3.0])

booster = SimpleGBDT(n_trees=1, max_depth=1, lr=1.0).fit(X, y)
assert booster.predict(X).tolist() == [1.0, 1.0, 3.0, 3.0]
assert booster.best_iteration is None

half = SimpleGBDT(n_trees=1, max_depth=1, lr=0.5).fit(X, y)
assert half.predict(X).tolist() == [0.5, 0.5, 1.5, 1.5]

X_stop = np.arange(4.0).reshape(-1, 1)
stopped = SimpleGBDT(n_trees=20, max_depth=1, lr=0.5)
stopped.fit(
    X_stop,
    np.ones(4),
    X_val=X_stop,
    y_val=np.zeros(4),
    early_stopping_rounds=2,
)
assert stopped.best_iteration == 1
assert len(stopped.trees) == 1
assert stopped.best_val_loss == 0.25
assert stopped.predict(X_stop).tolist() == [0.5, 0.5, 0.5, 0.5]

stopped.fit(X_stop, np.ones(4))
assert len(stopped.trees) == 20
assert stopped.best_iteration is None
assert stopped.best_val_loss is None
```

## 保存训练记录

`TrainLog(model=..., params=..., metrics=...)` 记录一次训练的模型名、参数和指标。
参数可用 `ModelParams.to_dict()` 生成, 训练完成前 `metrics` 可以为空字典。
`save(Path(...))` 和 `TrainLog.load(Path(...))` 读写 JSON, 父目录需要已存在。
模型权重直接用 LightGBM 的 `booster.save_model(path)` 保存,
通过 `lightgbm.Booster(model_file=path)` 重载, 无需额外保存接口。

评估复用 [metrics](metrics.md) 的 `rank_ic` 和 `ic_summary`, 回归误差可用
`sklearn.metrics.mean_squared_error`。排序分数不具有收益率单位, 不与收益标签计算 MSE。
[离线示例](../scripts/example_model_workflow.py) 展示了评估、日志写入和权重重载,
产物位于新的 `logs/model_example/run_*/` 目录。示例使用固定种子合成行情,
这些指标只用来检查流程能否跑通, 不能说明因子在市场上有效; 重叠标签的 IC 汇总也不代表独立样本推断。
