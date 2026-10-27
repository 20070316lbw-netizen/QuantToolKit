"""让 LightGBM 在同一天的股票之间学习排序

LambdaRankParams 保存训练配置, 不处理数据或开始训练
排序标签使用整数相关度, 数值越大, 越希望这只股票排在前面

group_sizes 数出每天有多少行, 告诉 LightGBM 谁和谁比较排名
同一天的股票放在一组, 毕竟这里关心的是当天选谁, 跨日期比较没有这个意义

先用 fit_lambdarank 传入特征表, 相关度标签和参数
训练完成后, 再用 predict_scores 对新样本打分
"""

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
        """把排序目标和随机种子整理成 lgb.train 使用的字典

        objective 设为 lambdarank, 让 LightGBM 按排序目标训练
        训练轮数不放进这个字典, 调用时另传 num_boost_round=params.num_boost_round

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
    """数出每天有多少条连续记录, 告诉 LambdaRank 谁和谁比较排名

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

    # 相邻两行日期不同时, 后一行就是新组的起点
    changed = np.flatnonzero(dates[1:] != dates[:-1]) + 1

    # 再补上整张表的起点和终点
    bounds = np.concatenate(([0], changed, [len(dates)]))

    # 两个相邻边界之间有多少行, 这一组就有多少只股票
    return np.diff(bounds)


def fit_lambdarank(
    *,
    features: pd.DataFrame,
    relevance: pd.Series,
    params: LambdaRankParams | None = None,
) -> lgb.Booster:
    """把同一天的股票放在一组, 训练 LambdaRank 排序模型

    内部把特征和相关度按索引排序
    相关度应是 relevance_by_date 的结果, 取值为 0 到 30 的整数
    不做早停
    调用 lgb.train 时额外把 verbosity 设为 -1, 不把这个日志设置写进参数类

    Args:
        features: [date, ticker] 特征表, 列名是 LightGBM 的特征名
        relevance: 与 features 对齐的整数相关度, 不能有缺失
        params: 默认 LambdaRankParams()

    Returns:
        训练好的 lightgbm.Booster

    Raises:
        ValueError: 特征为空或相关度不是对齐且无缺失的 0 到 30 整数

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

    # 先让标签按特征索引对齐, 再转成数组, 避免把答案配给另一只股票
    ordered = features.sort_index()
    label = relevance.reindex(ordered.index).to_numpy(dtype=float)

    if (
        ordered.empty
        or not np.isfinite(label).all()
        or not ((label >= 0) & (label <= 30) & (label == np.floor(label))).all()
    ):
        raise ValueError("相关度必须是对齐且无缺失的 0 到 30 整数, 特征不能为空")

    # 把特征, 对齐后的标签和每天的比较范围一起交给 LightGBM
    dataset = lgb.Dataset(
        ordered,
        label=label.astype("int32"),
        group=group_sizes(ordered.index),
        free_raw_data=False,
    )
    train_params = params.booster_params()

    # 关闭常规训练日志
    train_params["verbosity"] = -1

    return lgb.train(
        train_params,
        dataset,
        num_boost_round=params.num_boost_round,
    )


def predict_scores(*, booster: lgb.Booster, features: pd.DataFrame) -> pd.Series:
    """按特征表的行顺序预测 LightGBM 分数

    Args:
        booster: fit_lightgbm 或 fit_lambdarank 返回的模型
        features: [date, ticker] 特征表, 列与训练时相同

    Returns:
        与 features 索引相同的 Series, name 为 pred
        回归模型为标签同单位的预测值, 排序模型为连续排序分数

    Raises:
        lightgbm.basic.LightGBMError: 特征列名或顺序与训练时不同

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

    # group 用来确定训练时谁和谁比较, 预测时不用再传
    scores = booster.predict(features, validate_features=True)

    # LightGBM 返回数组, 这里把分数装回输入的 [date, ticker] 索引
    # 后面就能按日期排名, 选股和构造仓位, 行顺序仍与输入一致
    return pd.Series(scores, index=features.index, name="pred")
