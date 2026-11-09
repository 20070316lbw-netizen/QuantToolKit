# MLP 截面排序

实现集中在 [model/mlp](../src/quanttoolkit/model/mlp), 对外导出 `MLPParams`,
`fit_mlp_ranker` 和 `MLPRanker`。共用损失放在 [model/loss](../src/quanttoolkit/model/loss),
训练时从 `quanttoolkit.model.loss` 导入 `pairwise_rank_loss`。

输入沿用项目的 `[date, ticker]` 特征表和未来收益标签。
网络为每只股票输出一个连续分数, 训练时让同一天未来收益较高的股票获得更高分数。
输出没有收益率单位, 只用于同日排序, 不能当作预测收益计算 MSE。

## 目录与职责

| 文件 | 内容 |
| --- | --- |
| [mlp/params.py](../src/quanttoolkit/model/mlp/params.py) | 网络结构和训练参数, 继承 `ModelParams` 的 JSON 读写 |
| [mlp/network.py](../src/quanttoolkit/model/mlp/network.py) | 普通 `nn.Module`, 二维特征 Tensor → 一维分数 Tensor |
| [loss/ranking.py](../src/quanttoolkit/model/loss/ranking.py) | 不依赖 MLP 的 RankNet 成对排序损失 |
| [mlp/train.py](../src/quanttoolkit/model/mlp/train.py) | 标签对齐, 训练集缩放, 按日训练, 验证和早停 |
| [mlp/model.py](../src/quanttoolkit/model/mlp/model.py) | 预测, 特征顺序检查, 模型保存和加载 |

后续新增模型时, 可以参考参数 → 网络 → 训练 → 预测 → 保存重载这条流程。
损失只接收 Tensor, 日期分组由训练流程负责, 因而其他网络也能复用。
本次没有修改其他模型的目录或训练方式。

## 标签和排序损失

直接传入未来收益 Series, 不需要经过 `relevance_by_date` 分档。
对同日的每对非并列标签, 记收益较高者为 i, 较低者为 j, 损失为:

```text
loss(i, j) = softplus(-(score_i - score_j))
           = log(1 + exp(-(score_i - score_j)))
```

每对只计算一次, 并列收益跳过。每天先平均有效股票对的损失, 日期之间再等权平均,
避免证券较多的日期因股票对更多而获得更大权重。
这使用 [RankNet](https://www.microsoft.com/en-us/research/wp-content/uploads/2005/08/icml_ranking.pdf)
的成对排序思路, 不直接优化 Spearman Rank IC, 也不按收益差或前几名位置加权。

```python
import torch

from quanttoolkit.model.loss import pairwise_rank_loss

label = torch.tensor([0.01, 0.03])
correct = pairwise_rank_loss(torch.tensor([0.0, 1.0]), label)
reversed_order = pairwise_rank_loss(torch.tensor([1.0, 0.0]), label)
assert correct < reversed_order
assert pairwise_rank_loss(torch.zeros(2), torch.ones(2)) is None
```

`pairwise_rank_loss` 接收一个日期的 `(N,)` 分数和标签 Tensor。
无有效比较对时返回 `None`, 由调用方跳过。所有数值须有限, 不能把多个日期的股票混在一起比较。
使用全部股票对, 单日计算和存储量为 `O(N^2)`。

## 特征处理和训练

先用 [assemble_samples](dataset.md) 对齐历史特征与未来标签,
再用 [split_samples](split.md) 按日期切分并排除标签跨段样本。
`gap`, `n_periods`, 完整日历和标签结束日期仍使用原来的约定。
MLP 训练入口不自行切分数据, 测试集只在方案固定后评估。

训练入口按特征索引对齐标签, 内部排序后按日期组织完整截面。
特征和标签必须有限, 空特征或缺失标签会报错, 不自动补值。
训练集每列拟合均值和总体标准差 `ddof=0`, 常量列的缩放分母设为 1。
验证和预测始终使用训练统计量, 与逐日截面 `zscore_by_date` 是不同的处理。

一个完整日期是一个 batch, 每个日期更新一次网络, 每轮打乱训练日期。
内部先将当天收益转换为保留并列的整数秩, 只使用大小关系,
避免相近的原始 float64 收益转成 float32 后被误判为并列。
特征和网络使用 float32, 优化器为 AdamW。

不足两只股票或标签全相等的日期跳过, 数量写入 `training_summary`:
`train_days`, `train_skipped_days`, 有验证集时再加 `valid_days`, `valid_skipped_days`。
训练段或所提供的验证段没有任何有效日期时会报错。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `hidden_dims` | `[64, 32]` | 隐藏层宽度, 每层依次为 Linear, ReLU, Dropout, 空列表为线性打分 |
| `dropout` | `0.1` | 隐藏层丢弃比例 |
| `learning_rate` | `0.001` | AdamW 学习率 |
| `weight_decay` | `0.0001` | AdamW 权重衰减 |
| `max_epochs` | `100` | 最多完整遍历训练日期的轮数 |
| `early_stopping_rounds` | `10` | 验证损失连续不改善的轮数, 0 关闭提前停止 |
| `seed` | `7` | 网络初始化, dropout 和日期打乱的种子 |
| `device` | `"cpu"` | 训练设备, 可显式选择本机支持的 `mps` 或 `cuda` |

相同 CPU 环境和配置可复现, 不保证不同设备之间得到完全相同的结果。
当前示例和自动测试使用 CPU。

## 验证与训练记录

每轮结束使用 `eval` 模式计算验证损失, 关闭 dropout 和梯度计算。
验证损失严格下降才算改善, 有验证集时最终始终恢复最佳轮权重。
`early_stopping_rounds=0` 时仍选择最佳轮, 只是跑满 `max_epochs` 后再恢复。
没有验证集时使用最后一轮权重, `best_epoch` 和 `best_valid_loss` 都为 `None`。

`history` 每轮记录 `epoch` 和 `train_loss`, 有验证集时再加 `valid_loss`。
`train_loss` 是训练模式下各日期更新前的损失平均, 含 dropout 的影响;
`valid_loss` 使用轮末固定权重和预测模式, 两列计算时点和模式不同。
训练或验证损失非有限时抛出 `FloatingPointError`。
`best_epoch` 从 1 开始, `history` 会保留最佳轮之后实际运行过的记录。

评估复用 `rank_ic` 和 `ic_summary`, 分数也能接现有仓位函数。
重叠持有期标签的 IC 汇总不代表独立样本推断。

## 可运行的小例子

下面用人工设定的顺序检查训练和重载, 不涉及真实行情或未来收益生成。
从行情出发的完整流程见 [离线示例](../scripts/example_mlp_workflow.py)。

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from quanttoolkit.metrics import rank_ic
from quanttoolkit.model.mlp import MLPParams, MLPRanker, fit_mlp_ranker
from quanttoolkit.model.params import TrainLog

dates = pd.bdate_range("2024-01-01", periods=8)
index = pd.MultiIndex.from_product(
    [dates, ["A", "B", "C", "D"]], names=["date", "ticker"]
)
signal = np.tile([-1.5, -0.5, 0.5, 1.5], len(dates))
features = pd.DataFrame({"signal": signal, "constant": 2.0}, index=index)
label = pd.Series(signal * 0.01, index=index, name="synthetic_label")
train = features.index.get_level_values("date") <= dates[3]
valid = features.index.get_level_values("date").isin(dates[4:6])
test = features.index.get_level_values("date") > dates[5]
params = MLPParams(hidden_dims=[8], dropout=0.0, learning_rate=0.05, max_epochs=20)
model = fit_mlp_ranker(
    features=features.loc[train],
    label=label.loc[train],
    valid_features=features.loc[valid],
    valid_label=label.loc[valid],
    params=params,
)
scores = model.predict(features.loc[test])
assert scores.name == "pred"
assert scores.index.equals(features.loc[test].index)
assert rank_ic(factor=scores, forward_return=label.loc[test]).eq(1.0).all()

with TemporaryDirectory() as folder:
    path = Path(folder)
    model.save(path / "model.pt")
    loaded = MLPRanker.load(path / "model.pt")
    pd.testing.assert_series_equal(scores, loaded.predict(features.loc[test]))
    params.save(path / "params.json")
    assert MLPParams.load(path / "params.json") == params
    TrainLog(
        model="mlp_ranker",
        params=model.params.to_dict(),
        metrics={
            "best_epoch": model.best_epoch,
            "best_valid_loss": model.best_valid_loss,
        },
    ).save(path / "run.json")
```

## 保存与完整示例

`model.save(Path(...))` 同时保存 CPU `state_dict`, 参数, 特征列及顺序,
缩放统计量, `history`, 最佳轮和日期数量。父目录须存在。
文件保存的是预测模型, 不包含优化器和随机数状态, 不支持从中断位置续训。
使用 PyTorch 的 [state_dict 保存方式](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html),
加载时使用 `weights_only=True`。

`MLPRanker.load(path)` 默认恢复到 CPU, 可通过 `device="mps"` 或 `device="cuda"`
显式指定预测设备。实际网络设备与原训练配置分别保留, 加载不会改写 `params.device`。
预测检查特征列名和顺序, 沿用训练缩放, 返回与输入相同行顺序的 `pred` Series。
缺失值和无穷值会报错, 同列的空特征表返回空 Series。

在仓库根目录运行:

```bash
uv run python scripts/example_mlp_workflow.py
```

结果保存到新的 `logs/mlp_example/run_*/`:

- `model.pt`: 可独立重载的预测模型
- `run.json`: `TrainLog`, 记录参数, 标签期限, 切分边界, 验证和测试 IC 汇总
- `history.json`: 可直接检查的逐轮训练与验证损失

示例使用固定种子的合成随机行情, 没有设定可预测规律。
输出指标仅用于检查流程, 不能证明 MLP 或输入因子在市场上有效。
