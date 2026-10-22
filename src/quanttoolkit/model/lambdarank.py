"""LightGBM 默认 lambdarank 参数, 以及按日期分组的训练和预测"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

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


def group_sizes(index: pd.MultiIndex) -> np.ndarray:
    """数出每个日期连续行的个数, 供 lambdarank 的 group 使用

    Args:
        index: 已按 date 排序的 [date, ticker] MultiIndex
            同一天的行必须连在一起

    Returns:
        一维整数数组, 顺序与日期出现顺序一致, 总和等于行数
        空索引返回空数组

    Example:
        >>> index = pd.MultiIndex.from_product(
        ...     [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["A", "B"]],
        ...     names=["date", "ticker"],
        ... )
        >>> group_sizes(index).tolist()
        [2, 2]
    """
    dates = index.get_level_values("date")
    if len(dates) == 0:
        return np.array([], dtype=np.int64)
    changed = np.flatnonzero(dates[1:] != dates[:-1]) + 1
    bounds = np.concatenate(([0], changed, [len(dates)]))
    return np.diff(bounds)


def fit_lambdarank(
    *,
    features: pd.DataFrame,
    relevance: pd.Series,
    params: LambdaRankParams | None = None,
) -> lgb.Booster:
    """按日期分组训练默认 lambdarank

    内部把特征和相关度按索引排序
    相关度应是 relevance_by_date 的结果, 取值为 0 到 30 的整数
    不做早停
    调用 lgb.train 时额外把 verbosity 设为 -1, 不把训练日志写进返回的参数

    Args:
        features: [date, ticker] 特征表, 列名是 LightGBM 的特征名
        relevance: 与 features 对齐的整数相关度, 不能有缺失
        params: 默认 LambdaRankParams()

    Returns:
        训练好的 lightgbm.Booster

    Example:
        >>> index = pd.MultiIndex.from_product(
        ...     [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["A", "B", "C"]],
        ...     names=["date", "ticker"],
        ... )
        >>> features = pd.DataFrame(
        ...     {"mom_21": [0.1, 0.2, 0.0, 0.2, 0.0, 0.1]}, index=index
        ... )
        >>> relevance = pd.Series([0, 2, 1, 2, 0, 1], index=index)
        >>> booster = fit_lambdarank(
        ...     features=features,
        ...     relevance=relevance,
        ...     params=LambdaRankParams(num_boost_round=1, seed=1),
        ... )
        >>> booster.num_trees()
        1
    """
    if params is None:
        params = LambdaRankParams()
    ordered = features.sort_index()
    label = relevance.reindex(ordered.index).astype("int32").to_numpy()
    dataset = lgb.Dataset(
        ordered,
        label=label,
        group=group_sizes(ordered.index),
        free_raw_data=False,
    )
    train_params = params.booster_params()
    train_params["verbosity"] = -1
    return lgb.train(
        train_params,
        dataset,
        num_boost_round=params.num_boost_round,
    )


def predict_scores(*, booster: lgb.Booster, features: pd.DataFrame) -> pd.Series:
    """按特征表的行顺序预测排序分数

    Args:
        booster: fit_lambdarank 返回的模型
        features: [date, ticker] 特征表, 列与训练时相同

    Returns:
        与 features 索引相同的 Series, name 为 pred
        值是 LightGBM 的排序分数, 不是整数档

    Example:
        >>> index = pd.MultiIndex.from_product(
        ...     [pd.to_datetime(["2024-01-02"]), ["A", "B", "C"]],
        ...     names=["date", "ticker"],
        ... )
        >>> features = pd.DataFrame({"mom_21": [0.1, 0.2, 0.0]}, index=index)
        >>> relevance = pd.Series([0, 2, 1], index=index)
        >>> booster = fit_lambdarank(
        ...     features=features,
        ...     relevance=relevance,
        ...     params=LambdaRankParams(num_boost_round=1, seed=1),
        ... )
        >>> scores = predict_scores(booster=booster, features=features)
        >>> scores.name
        'pred'
        >>> bool(scores.notna().all())
        True
    """
    scores = booster.predict(features)
    return pd.Series(scores, index=features.index, name="pred")
