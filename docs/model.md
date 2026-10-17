# 模型参数

实现见 [model](../src/quanttoolkit/model)

模型参数使用 dataclass, 这是 `AGENT.md` 里允许的例外
字段只用 `str`, `int`, `float`, `bool`, `list`, `dict`, 以便写成 JSON
`ModelParams` 只有共用的 `seed`
每种模型继承它, 再写自己的字段
`save` 和 `load` 读写 JSON 文件

`LightGBMParams` 是回归任务的参数
标签是 `assemble_samples` 最后一列的未来持有期收益
`num_boost_round` 和 `early_stopping_rounds` 属于训练循环
`booster_params()` 去掉这两项, 并把 `seed` 抄到 LightGBM 的三个随机种子
没写出的 LightGBM 参数沿用其自身默认值

`LambdaRankParams` 是全市场排序用的 lambdarank 参数
传给 `lgb.train` 的字典只有 `objective="lambdarank"` 和种子
学习率, 叶子数, 评价指标, `label_gain` 都不传, 使用 LightGBM 自己的默认值
相关度标签因此必须落在 `0` 到 `30`
`num_boost_round` 是 `100`, 不做早停

`TrainLog` 记录一次训练的模型名, 参数字典和指标
训练完成前 `metrics` 可以是空字典

```python
from quanttoolkit.model.lambdarank import LambdaRankParams
from quanttoolkit.model.lightgbm import LightGBMParams

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
```
