"""按完整日期组织样本, 训练 MLP 截面排序"""

from copy import deepcopy

import numpy as np
import pandas as pd
import torch

from quanttoolkit.model.loss import pairwise_rank_loss
from quanttoolkit.model.mlp.model import MLPRanker
from quanttoolkit.model.mlp.network import MLP
from quanttoolkit.model.mlp.params import MLPParams
from quanttoolkit.registry import _contract, _panel_frame


@_contract
def fit_mlp_ranker(
    *,
    features: pd.DataFrame,
    label: pd.Series,
    valid_features: pd.DataFrame | None = None,
    valid_label: pd.Series | None = None,
    params: MLPParams | None = None,
) -> MLPRanker:
    """用同日股票的未来收益顺序训练 MLP

    Args:
        features: 非空 [date, ticker] DataFrame, 特征列唯一且为有限数值
            特征须在基准日可得, 标签不能混入特征, 不自动填补缺失
        label: 同索引的有限未来收益 Series, 按特征索引对齐
            只使用同日大小关系, 并列收益不强行区分
        valid_features: 验证特征, 默认 None, 列名和顺序与训练一致
        valid_label: 验证标签, 默认 None, 与 valid_features 成对提供
            日期隔离和标签跨段处理由调用方使用 split_samples 完成
        params: 默认 MLPParams(), CPU, 最多 100 轮, 早停耐心值为 10
            每日完整截面为一个 batch, 每轮打乱训练日期
            每天的有效股票对等权, 日期之间等权, 不跨日期比较

    Returns:
        MLPRanker, 不修改输入, 内部按日期排序后训练
        均值和总体标准差仅由训练特征拟合, 常量列的缩放分母为 1
        不足 2 只股票或标签全相等的日期跳过, 数量记入 training_summary
        history 的 train_loss 为训练模式下各次更新前的每日损失平均
        valid_loss 为轮末 eval 模式下的每日损失平均
        验证损失严格下降才改善, 有验证集时始终恢复最佳轮权重
        early_stopping_rounds=0 时跑满轮数再恢复最佳轮
        无验证集时保留最后一轮, best_epoch 和 best_valid_loss 为 None

    Raises:
        ValueError: 验证输入不成对, 特征为空或非有限, 标签不能对齐
            特征列不一致, 某段没有有效比较日期, 或轮数和耐心值无效
        FloatingPointError: 训练或验证损失变为非有限数值
        RuntimeError: 网络配置无效或指定设备不可用

    Example:
        >>> index = pd.MultiIndex.from_product(
        ...     [pd.to_datetime(["2024-01-02"]), ["A", "B"]],
        ...     names=["date", "ticker"],
        ... )
        >>> x = pd.DataFrame({"x": [0.0, 1.0]}, index=index)
        >>> model = fit_mlp_ranker(
        ...     features=x, label=pd.Series([0.0, 0.1], index=index),
        ...     params=MLPParams(hidden_dims=[], max_epochs=2),
        ... )
        >>> model.training_summary["train_days"]
        1
        >>> len(model.history)
        2
    """
    if (valid_features is None) != (valid_label is None):
        raise ValueError("valid_features 和 valid_label 必须同时提供")
    params = MLPParams(**(params or MLPParams()).to_dict())
    if params.max_epochs < 1 or params.early_stopping_rounds < 0:
        raise ValueError("max_epochs 必须为正整数, early_stopping_rounds 必须非负")
    ordered, aligned = _prepare(features, label)
    if valid_features is not None:
        valid_features, valid_label = _prepare(valid_features, valid_label)
        if not valid_features.columns.equals(ordered.columns):
            raise ValueError("训练集和验证集的特征列名及顺序必须相同")

    values = ordered.to_numpy(dtype=float)
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale[scale == 0] = 1.0
    torch.manual_seed(params.seed)
    network = MLP(len(ordered.columns), params.hidden_dims, params.dropout).to(
        device=params.device, dtype=torch.float32
    )
    model = MLPRanker(
        network=network,
        params=params,
        feature_columns=ordered.columns.tolist(),
        feature_mean=mean,
        feature_scale=scale,
    )
    train, skipped = _daily_batches(model, ordered, aligned)
    model.training_summary.update(train_days=len(train), train_skipped_days=skipped)
    valid = []
    if valid_features is not None:
        valid, skipped = _daily_batches(model, valid_features, valid_label)
        model.training_summary.update(valid_days=len(valid), valid_skipped_days=skipped)
    optimizer = torch.optim.AdamW(
        network.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay
    )
    rng = np.random.default_rng(params.seed)
    best_state = None
    stale = 0
    for epoch in range(1, params.max_epochs + 1):
        network.train()
        total = 0.0
        for day in rng.permutation(len(train)):
            x, y = train[day]
            optimizer.zero_grad(set_to_none=True)
            loss = pairwise_rank_loss(network(x), y)
            loss.backward()
            optimizer.step()
            total += loss.detach().item()
        row = {"epoch": epoch, "train_loss": total / len(train)}
        if valid:
            network.eval()
            with torch.inference_mode():
                row["valid_loss"] = sum(
                    pairwise_rank_loss(network(x), y).item() for x, y in valid
                ) / len(valid)
        if not all(np.isfinite(value) for value in row.values()):
            raise FloatingPointError("MLP 损失变为非有限数值, 请检查特征和学习率")
        model.history.append(row)
        if valid:
            if (
                model.best_valid_loss is None
                or row["valid_loss"] < model.best_valid_loss
            ):
                model.best_valid_loss = row["valid_loss"]
                model.best_epoch = epoch
                best_state = deepcopy(network.state_dict())
                stale = 0
            else:
                stale += 1
            if (
                params.early_stopping_rounds > 0
                and stale >= params.early_stopping_rounds
            ):
                break
    if best_state is not None:
        network.load_state_dict(best_state)
    network.eval()
    return model


def _prepare(features, label):
    _panel_frame(features, "MLP.features")
    if features.empty or not features.columns.is_unique:
        raise ValueError("MLP 特征不能为空, 列名必须唯一")
    ordered = features.sort_index()
    aligned = label.reindex(ordered.index).to_numpy(dtype=float)
    if not np.isfinite(aligned).all():
        raise ValueError("MLP 标签必须按特征索引对齐且为有限数值")
    if not np.isfinite(ordered.to_numpy(dtype=float)).all():
        raise ValueError("MLP 特征必须为有限数值, 请先处理缺失值")
    return ordered, aligned


def _daily_batches(model, features, label):
    values = model._transform(features)
    batches = []
    skipped = 0
    for positions in features.groupby(level="date", sort=True).indices.values():
        # 用整数秩保留原始收益的大小与并列关系, 避免 float32 把接近的收益合并
        _, ranks = np.unique(label[positions], return_inverse=True)
        if len(positions) < 2 or ranks.max() == 0:
            skipped += 1
            continue
        rows = torch.as_tensor(positions, dtype=torch.long, device=values.device)
        y = torch.as_tensor(ranks, dtype=torch.long, device=values.device)
        batches.append((values[rows], y))
    if not batches:
        raise ValueError("训练或验证段没有可比较的日期, 每天至少需要两个不同标签")
    return batches, skipped
