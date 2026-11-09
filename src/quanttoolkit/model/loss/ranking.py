"""与具体网络结构无关的 RankNet 成对排序损失"""

import torch
from torch.nn import functional as F


def pairwise_rank_loss(
    scores: torch.Tensor, label: torch.Tensor
) -> torch.Tensor | None:
    """对标签较高者应有更高分数的股票对求平均损失

    Args:
        scores: 一个日期的有限连续分数, 形状为 (N,) 的浮点 Tensor
        label: 同一日期且与 scores 对齐的有限标签, 形状为 (N,)
            可用未来收益或保留并列关系的整数秩, 与 scores 在同一设备

    Returns:
        标量 Tensor, 对所有 label[i] > label[j] 的组合平均
        每对损失为 softplus(-(scores[i] - scores[j])), 每对只计算一次
        并列标签不参与比较, N < 2 或标签全部相等时返回 None
        调用方负责按日期分组和跳过 None, 计算与存储量为 O(N^2)

    Raises:
        RuntimeError: Tensor 的形状或设备不匹配

    Example:
        >>> y = torch.tensor([0.1, 0.2])
        >>> good = pairwise_rank_loss(torch.tensor([0.0, 1.0]), y)
        >>> bad = pairwise_rank_loss(torch.tensor([1.0, 0.0]), y)
        >>> bool(good < bad)
        True
        >>> pairwise_rank_loss(torch.zeros(2), torch.ones(2)) is None
        True
    """
    higher = label[:, None] > label[None, :]
    if not higher.any():
        return None
    differences = scores[:, None] - scores[None, :]
    return F.softplus(-differences[higher]).mean()
