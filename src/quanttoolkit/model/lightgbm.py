"""准备 LightGBM 回归参数, 再用未来收益标签训练模型

LightGBMParams 只保存配置, fit_lightgbm 负责把数据交给 LightGBM
有验证集时可以早停, 预测结果仍可接回项目里的评估和仓位函数
"""

from __future__ import annotations

from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd

from quanttoolkit.model.params import ModelParams


@dataclass
class LightGBMParams(ModelParams):
    """回归任务的 LightGBM 参数

    标签是 assemble_samples 最后一列的未来持有期收益, 不做分档
    num_boost_round 和 early_stopping_rounds 属于训练循环, 不放入 booster 参数
    booster_params 会去掉这两项, 并把 seed 复制到 LightGBM 的三个专项随机种子
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


def fit_lightgbm(
    *,
    features: pd.DataFrame,
    label: pd.Series,
    valid_features: pd.DataFrame | None = None,
    valid_label: pd.Series | None = None,
    params: LightGBMParams | None = None,
) -> lgb.Booster:
    """训练 LightGBM 回归, 有验证集时按验证指标早停

    Args:
        features: 非空 [date, ticker] 特征表, 列名为字符串
        label: 同索引的有限数值 Series, 内部按 features 索引对齐
            通常为未来简单收益率, 不做分档
        valid_features: 验证特征, 默认 None, 列名和顺序须与训练一致
        valid_label: 验证标签, 默认 None, 必须与 valid_features 同时提供
        params: 默认 LightGBMParams(), 默认最多 200 轮
            有验证集且 early_stopping_rounds > 0 时早停, 默认连续 20 轮未改善就停止
            无验证集或 early_stopping_rounds=0 时训练指定轮数

    Returns:
        lightgbm.Booster, 不修改输入
        best_iteration 为验证集选定轮数, 无早停时为 0
        特征可含 NaN, 由 LightGBM 处理, 标签不得缺失

    Raises:
        ValueError: 验证输入不成对, 特征列不一致, 样本为空或标签非有限
        lightgbm.basic.LightGBMError: LightGBM 参数或数据不合法

    Example:
        >>> index = pd.MultiIndex.from_product(
        ...     [pd.to_datetime(["2024-01-02"]), ["A", "B"]],
        ...     names=["date", "ticker"],
        ... )
        >>> features = pd.DataFrame({"x": [0.0, 1.0]}, index=index)
        >>> booster = fit_lightgbm(
        ...     features=features, label=pd.Series([0.1, 0.2], index=index),
        ...     params=LightGBMParams(num_boost_round=1),
        ... )
        >>> booster.feature_name()
        ['x']
    """
    if (valid_features is None) != (valid_label is None):
        raise ValueError("valid_features 和 valid_label 必须同时提供")
    if params is None:
        params = LightGBMParams()
    train = _dataset(features, label)
    valid_sets = None
    callbacks = []
    if valid_features is not None:
        if not valid_features.columns.equals(features.columns):
            raise ValueError("训练集和验证集的特征列名及顺序必须相同")
        valid_sets = [_dataset(valid_features, valid_label, reference=train)]
        if params.early_stopping_rounds > 0:
            callbacks.append(
                lgb.early_stopping(params.early_stopping_rounds, verbose=False)
            )
    return lgb.train(
        params.booster_params(),
        train,
        num_boost_round=params.num_boost_round,
        valid_sets=valid_sets,
        valid_names=["valid"] if valid_sets else None,
        callbacks=callbacks,
    )


def _dataset(features, label, reference=None):
    aligned = label.reindex(features.index)
    if features.empty or not np.isfinite(aligned.to_numpy(dtype=float)).all():
        raise ValueError("样本不能为空, 标签必须按特征索引对齐且为有限数值")
    return lgb.Dataset(features, label=aligned, reference=reference)
