"""LightGBM 默认 lambdarank 参数"""

from __future__ import annotations

from dataclasses import dataclass

from quanttoolkit.model.params import ModelParams


@dataclass
class LambdaRankParams(ModelParams):
    """全市场排序用的 LightGBM lambdarank 参数

    传给 lgb.train 的字典只包含目标和种子
    学习率, 叶子数, 评价指标, label_gain 等都不传入, 使用 LightGBM 自己的默认值
    因此相关度标签必须落在默认 label_gain 的 0 到 30
    num_boost_round 使用 lgb.train 的默认轮数 100, 不放进 booster 参数
    不做早停

    Example:
        >>> params = LambdaRankParams(seed=3)
        >>> params.num_boost_round
        100
        >>> params.booster_params()
        {'objective': 'lambdarank', 'seed': 3, 'feature_fraction_seed': 3, 'bagging_seed': 3, 'data_random_seed': 3}
    """

    num_boost_round: int = 100

    def booster_params(self) -> dict:
        """返回传给 lgb.train 的参数字典

        Returns:
            只含 objective 和四个种子的 dict
            不含学习率, 叶子数, 评价指标和 num_boost_round
        """
        return {
            "objective": "lambdarank",
            "seed": self.seed,
            "feature_fraction_seed": self.seed,
            "bagging_seed": self.seed,
            "data_random_seed": self.seed,
        }
