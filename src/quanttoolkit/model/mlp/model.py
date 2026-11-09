"""把训练后的网络和预测所需的特征处理一起保存"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from quanttoolkit.model.mlp.network import MLP
from quanttoolkit.model.mlp.params import MLPParams
from quanttoolkit.registry import _contract


class MLPRanker:
    """fit_mlp_ranker 返回的模型, 保存网络, 特征缩放和训练记录

    通常通过训练入口或 load 创建
    params 保存训练配置, network 所在设备决定实际预测设备
    feature_mean 和 feature_scale 仅由训练集拟合, 标准差使用 ddof=0
    history 记录逐轮损失, best_epoch 从 1 开始, 无验证集时为 None
    training_summary 记录各段有效日期和跳过日期的数量
    """

    def __init__(
        self,
        *,
        network: MLP,
        params: MLPParams,
        feature_columns: list[str],
        feature_mean: np.ndarray,
        feature_scale: np.ndarray,
    ) -> None:
        self.network = network
        self.params = params
        self.feature_columns = list(feature_columns)
        self.feature_mean = feature_mean.copy()
        self.feature_scale = feature_scale.copy()
        self.history: list[dict] = []
        self.best_epoch: int | None = None
        self.best_valid_loss: float | None = None
        self.training_summary: dict = {}

    def _transform(self, features: pd.DataFrame) -> torch.Tensor:
        if features.columns.tolist() != self.feature_columns:
            raise ValueError("特征列名和顺序必须与训练时相同")
        values = features.to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("MLP 特征必须为有限数值, 请先处理缺失值")
        scaled = ((values - self.feature_mean) / self.feature_scale).astype(np.float32)
        device = next(self.network.parameters()).device
        return torch.from_numpy(scaled).to(device)

    @_contract
    def predict(self, features: pd.DataFrame) -> pd.Series:
        """使用训练时的缩放统计量对新样本打分

        Args:
            features: [date, ticker] DataFrame, 各键唯一
                特征列名及顺序与训练时一致, 数值有限, 无需标签

        Returns:
            与输入索引和行顺序相同的 Series, name 为 pred
            无收益率单位, 分数越高表示越希望排在前面
            关闭 dropout 和梯度计算, 空输入返回空 Series, 不修改输入

        Raises:
            TypeError: 输入不是面板 DataFrame
            ValueError: 面板索引无效, 特征列不一致或数值非有限

        Example:
            >>> from quanttoolkit.model.mlp import MLPParams, fit_mlp_ranker
            >>> index = pd.MultiIndex.from_product(
            ...     [pd.to_datetime(["2024-01-02"]), ["A", "B"]],
            ...     names=["date", "ticker"],
            ... )
            >>> x = pd.DataFrame({"x": [0.0, 1.0]}, index=index)
            >>> model = fit_mlp_ranker(
            ...     features=x, label=pd.Series([0.0, 0.1], index=index),
            ...     params=MLPParams(hidden_dims=[], max_epochs=1),
            ... )
            >>> model.predict(x).name
            'pred'
        """
        values = self._transform(features)
        self.network.eval()
        with torch.inference_mode():
            scores = self.network(values).cpu().numpy()
        return pd.Series(scores, index=features.index, name="pred")

    def save(self, path: Path) -> None:
        """保存供预测重载的权重, 配置, 缩放和训练记录

        Args:
            path: 模型文件路径, 建议使用 .pt, 父目录须存在

        Returns:
            None, 写入 CPU state_dict 和普通 Python 数据
            不包含优化器和随机数状态, 不用于恢复中断训练

        Raises:
            OSError: 路径不可写或父目录不存在
        """
        torch.save(
            {
                "state_dict": {
                    name: value.detach().cpu()
                    for name, value in self.network.state_dict().items()
                },
                "params": self.params.to_dict(),
                "feature_columns": self.feature_columns,
                "feature_mean": self.feature_mean.tolist(),
                "feature_scale": self.feature_scale.tolist(),
                "history": self.history,
                "best_epoch": self.best_epoch,
                "best_valid_loss": self.best_valid_loss,
                "training_summary": self.training_summary,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path, *, device: str = "cpu") -> MLPRanker:
        """恢复预测模型, 默认在 CPU 加载

        Args:
            path: save 生成的模型文件
            device: 实际预测设备, 默认 cpu, 不改写原来的训练配置

        Returns:
            MLPRanker, 网络处于 eval 模式, 保留缩放和训练记录

        Raises:
            OSError: 文件不存在或不可读
            RuntimeError: 权重无效或指定设备不可用

        Example:
            >>> from tempfile import TemporaryDirectory
            >>> from quanttoolkit.model.mlp import MLPParams, fit_mlp_ranker
            >>> index = pd.MultiIndex.from_product(
            ...     [pd.to_datetime(["2024-01-02"]), ["A", "B"]],
            ...     names=["date", "ticker"],
            ... )
            >>> x = pd.DataFrame({"x": [0.0, 1.0]}, index=index)
            >>> model = fit_mlp_ranker(
            ...     features=x, label=pd.Series([0.0, 0.1], index=index),
            ...     params=MLPParams(hidden_dims=[], max_epochs=1),
            ... )
            >>> with TemporaryDirectory() as folder:
            ...     path = Path(folder) / "model.pt"
            ...     model.save(path)
            ...     loaded = MLPRanker.load(path)
            ...     np.testing.assert_array_equal(model.predict(x), loaded.predict(x))
        """
        saved = torch.load(path, map_location="cpu", weights_only=True)
        params = MLPParams(**saved["params"])
        network = MLP(
            len(saved["feature_columns"]), params.hidden_dims, params.dropout
        ).to(device=device, dtype=torch.float32)
        network.load_state_dict(saved["state_dict"])
        network.eval()
        model = cls(
            network=network,
            params=params,
            feature_columns=saved["feature_columns"],
            feature_mean=np.array(saved["feature_mean"]),
            feature_scale=np.array(saved["feature_scale"]),
        )
        model.history = saved["history"]
        model.best_epoch = saved["best_epoch"]
        model.best_valid_loss = saved["best_valid_loss"]
        model.training_summary = saved["training_summary"]
        return model
