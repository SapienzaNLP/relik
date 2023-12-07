from typing import List

import torch
import transformers
from torch.optim import AdamW


class AdamWWithWarmupOptimizer:
    def __init__(
        self,
        lr: float,
        warmup_steps: int,
        total_steps: int,
        weight_decay: float,
        no_decay_params: List[str],
    ):
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.weight_decay = weight_decay
        self.no_decay_params = no_decay_params

    def group_params(self, module: torch.nn.Module) -> list:
        if self.no_decay_params is not None:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in module.named_parameters()
                        if not any(nd in n for nd in self.no_decay_params)
                    ],
                    "weight_decay": self.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in module.named_parameters()
                        if any(nd in n for nd in self.no_decay_params)
                    ],
                    "weight_decay": 0.0,
                },
            ]

        else:
            optimizer_grouped_parameters = [
                {"params": module.parameters(), "weight_decay": self.weight_decay}
            ]

        return optimizer_grouped_parameters

    def __call__(self, module: torch.nn.Module):
        optimizer_grouped_parameters = self.group_params(module)
        optimizer = AdamW(
            optimizer_grouped_parameters, lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer, self.warmup_steps, self.total_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
