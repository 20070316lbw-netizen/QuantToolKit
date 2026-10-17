"""LightGBM 回归参数"""

from __future__ import annotations

from dataclasses import dataclass

from quanttoolkit.model.params import ModelParams


@dataclass
class LightGBMParams(ModelParams):
    """回归任务的 LightGBM 参数

    标签是 assemble_samples 最后一列的未来持有期收益, 不做分档
    num_boost_round 和 early_stopping_rounds 属于训练循环, 不传给 lgb.train
    booster_params 会去掉这两项, 并把 seed 抄到 LightGBM 的三个随机种子
    其余没写出的 LightGBM 参数沿用其自身默认值

    Example:
        >>> params = LightGBMParams(num_leaves=15, seed=3)
        >>> booster = params.booster_params()
        >>> params.num_boost_round
        200
        >>> "num_boost_round" in booster
        False
        >>> booster["num_leaves"]
        15
        >>> booster["data_random_seed"]
        3
    """

    objective: str = "regression"
    metric: str = "l2"
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_data_in_leaf: int = 50
    feature_fraction: float = 1.0
    verbosity: int = -1
    num_boost_round: int = 200
    early_stopping_rounds: int = 20

    def booster_params(self) -> dict:
        """返回传给 lgb.train 的参数字典

        Returns:
            不含 num_boost_round 和 early_stopping_rounds 的 dict
            seed, feature_fraction_seed, bagging_seed, data_random_seed 都等于 seed
        """
        params = self.to_dict()
        params.pop("num_boost_round")
        params.pop("early_stopping_rounds")
        params["feature_fraction_seed"] = self.seed
        params["bagging_seed"] = self.seed
        params["data_random_seed"] = self.seed
        return params
