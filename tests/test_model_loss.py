import doctest

import pytest
import torch
from torch.nn import functional as F

from quanttoolkit.model.loss import (
    PearsonCorrLoss,
    pairwise_rank_loss,
    pearsoncorrloss,
    ranking,
)


def test_pairwise_loss_direction_ties_and_gradient():
    scores = torch.tensor([0.0, 0.0, 1.0], requires_grad=True)
    label = torch.tensor([0.1, 0.1, 0.2])
    value = pairwise_rank_loss(scores, label)
    torch.testing.assert_close(value, F.softplus(torch.tensor(-1.0)))
    value.backward()
    assert scores.grad[:2].gt(0).all()
    assert scores.grad[2] < 0
    assert value < pairwise_rank_loss(-scores, label)


def test_loss_handles_no_pairs_and_extreme_score_differences():
    assert pairwise_rank_loss(torch.ones(1), torch.ones(1)) is None
    assert pairwise_rank_loss(torch.ones(2), torch.zeros(2)) is None
    assert pairwise_rank_loss(torch.empty(0), torch.empty(0)) is None
    assert torch.isfinite(
        pairwise_rank_loss(torch.tensor([1000.0, -1000.0]), torch.tensor([0, 1]))
    )


def test_loss_ignores_row_order_and_common_score_offset():
    scores = torch.tensor([0.0, 1.0, -1.0])
    labels = torch.tensor([0.1, 0.3, 0.2])
    order = [2, 0, 1]
    expected = pairwise_rank_loss(scores, labels)
    torch.testing.assert_close(
        pairwise_rank_loss(scores[order], labels[order]), expected
    )
    torch.testing.assert_close(pairwise_rank_loss(scores + 10, labels), expected)


def test_ranking_loss_docstrings():
    result = doctest.testmod(ranking)
    assert result.failed == 0 and result.attempted > 0


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_pearson_loss_matches_correlation_for_strided_input(dtype):
    pred = torch.tensor([[0.0, 0.0], [2.0, 0.0], [1.0, 0.0]], dtype=dtype)
    pred = pred[:, :1].detach().requires_grad_()
    target = torch.tensor([1.0, 2.0, 3.0], dtype=dtype)
    assert not pred.is_contiguous()
    loss = PearsonCorrLoss()(pred, target)
    expected = (
        1 - torch.corrcoef(torch.stack([pred[:, 0], target]).to(loss.dtype))[0, 1]
    )
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.isfinite(pred.grad).all()


@pytest.mark.parametrize(
    ("pred", "target"),
    [([], []), ([1.0], [2.0]), ([0.0, 0.0], [1.0, 2.0]), ([1.0, 2.0], [0.0, 0.0])],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_pearson_loss_has_zero_gradients_for_degenerate_inputs(pred, target, dtype):
    scores = torch.tensor(pred, dtype=dtype, requires_grad=True)
    loss = PearsonCorrLoss()(scores, torch.tensor(target, dtype=dtype))
    assert loss.item() == 1.0
    loss.backward()
    torch.testing.assert_close(scores.grad, torch.zeros_like(scores))


def test_pearson_loss_rejects_mismatched_samples():
    with pytest.raises(ValueError, match="样本数相同"):
        PearsonCorrLoss()(torch.ones(1), torch.ones(3))


def test_pearson_loss_docstrings():
    result = doctest.testmod(pearsoncorrloss)
    assert result.failed == 0 and result.attempted > 0
