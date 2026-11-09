"""与具体网络结构无关的 Pearson IC 损失"""

import math

import torch
from torch import nn


class PearsonCorrLoss(nn.Module):
    """通过最小化 1 - Pearson IC 提高预测值与标签的线性相关性

    Args:
        eps: 加到两个中心化向量的 L2 范数乘积上的正数, 默认 1e-8

    Raises:
        ValueError: eps 不是有限正数

    Example:
        >>> import torch
        >>> from quanttoolkit.model.loss import PearsonCorrLoss
        >>> y = torch.tensor([-1.0, 0.0, 1.0])
        >>> loss = PearsonCorrLoss()
        >>> loss(y[:, None], y).item()
        0.0
        >>> loss(-y, y).item()
        2.0
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("eps 必须是有限正数")
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """计算单个截面的 Pearson IC 损失

        Args:
            pred: 一个日期的有限预测值, 形状为 (N,) 或 (N, 1) 的浮点 Tensor
            target: 同日且与 pred 逐行对齐的有限浮点标签, 形状为 (N,) 或 (N, 1)
                与 pred 在同一设备, 调用方负责按日期分组并处理缺失值

        Returns:
            与 pred 同设备的无量纲标量 Tensor, 可反向传播
            float16 和 bfloat16 输入以至少 float32 精度计算
            样本少于两个或任一输入为常量时, 损失为 1 且梯度为零

        Raises:
            ValueError: 输入形状不符合要求或样本数不同
            RuntimeError: 输入设备不匹配

        Example:
            >>> pred = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
            >>> loss = PearsonCorrLoss()(pred, torch.tensor([1.0, 3.0, 2.0]))
            >>> round(loss.item(), 6)
            0.5
            >>> loss.backward()
            >>> bool(torch.isfinite(pred.grad).all())
            True
        """
        if pred.ndim == 2:
            pred = pred.squeeze(-1)
        if target.ndim == 2:
            target = target.squeeze(-1)
        if pred.ndim != 1 or pred.shape != target.shape:
            raise ValueError("pred 和 target 必须是样本数相同的 (N,) 或 (N, 1) Tensor")

        dtype = torch.promote_types(pred.dtype, target.dtype)
        if dtype in (torch.float16, torch.bfloat16):
            dtype = torch.float32
        pred = pred.to(dtype)
        target = target.to(dtype)
        # 先减去首个值, 确保常量向量不会因求均值的舍入误差产生残差
        pred = pred - pred[:1].detach()
        target = target - target[:1].detach()
        pred = pred - pred.mean()
        target = target - target.mean()

        # vector_norm 在零向量处的反向传播返回零梯度
        scale = torch.linalg.vector_norm(pred) * torch.linalg.vector_norm(target)
        covariance = (pred * target).sum()
        # 相关系数无定义时跳过梯度, 避免除以 eps 放大梯度
        covariance = torch.where(scale == 0, 0, covariance)
        corr = covariance / (scale + self.eps)
        return 1 - corr.clamp(-1, 1)
