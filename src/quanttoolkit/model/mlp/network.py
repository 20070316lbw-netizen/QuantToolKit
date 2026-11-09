"""只定义二维特征到一维分数的网络, 不处理日期和训练流程"""

import torch
from torch import nn


class MLP(nn.Module):
    """各隐藏层依次使用 Linear, ReLU 和 Dropout, 输出层只做线性映射

    Args:
        input_dim: 正整数, 每条样本的特征数
        hidden_dims: 各隐藏层的正整数宽度, 空列表表示直接线性打分
        dropout: 隐藏层丢弃比例, 默认 0.1, 范围为 [0, 1]

    Example:
        >>> MLP(input_dim=2, hidden_dims=[4], dropout=0.0)(torch.zeros(3, 2)).shape
        torch.Size([3])
    """

    def __init__(
        self, input_dim: int, hidden_dims: list[int], dropout: float = 0.1
    ) -> None:
        super().__init__()
        layers = []
        width = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(width, hidden), nn.ReLU(), nn.Dropout(dropout)])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.layers = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """返回每条样本的连续分数, 不在输出端施加 sigmoid

        Args:
            features: 形状为 (N, input_dim) 的有限浮点 Tensor
                dtype 和 device 须与网络相同

        Returns:
            形状为 (N,) 的 Tensor, 分数越高表示越希望排在前面
            没有收益率单位, N=0 时返回空 Tensor

        Raises:
            RuntimeError: 特征形状, dtype 或 device 与网络不匹配

        Example:
            >>> net = MLP(input_dim=1, hidden_dims=[], dropout=0.0)
            >>> net(torch.ones(1, 1)).shape
            torch.Size([1])
        """
        return self.layers(features).squeeze(-1)
