from typing import Optional

import torch
from torch.nn.modules.loss import _WeightedLoss


class MultiLabelNCELoss(_WeightedLoss):
    __constants__ = ["reduction"]

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        size_average=None,
        reduction: Optional[str] = "mean",
    ) -> None:
        super(MultiLabelNCELoss, self).__init__(weight, size_average, None, reduction)

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, ignore_index: int = -100
    ) -> torch.Tensor:
        gold_scores = input.masked_fill(~(target.bool()), 0)
        gold_scores_sum = gold_scores.sum(-1)  # B x C
        neg_logits = input.masked_fill(target.bool(), float("-inf"))  # B x C x L
        neg_log_sum_exp = torch.logsumexp(neg_logits, -1, keepdim=True)  # B x C x 1
        norm_term = (
            torch.logaddexp(input, neg_log_sum_exp)
            .masked_fill(~(target.bool()), 0)
            .sum(-1)
        )
        gold_log_probs = gold_scores_sum - norm_term
        loss = -gold_log_probs.sum()
        if self.reduction == "mean":
            loss /= input.size(0)
        return loss
