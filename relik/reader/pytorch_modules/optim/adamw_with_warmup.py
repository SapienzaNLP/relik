from typing import List

import torch
import transformers
from torch.optim import AdamW


class AdamWWithWarmupOptimizer:
    def __init__(
        self,
        lr: float | List[float],
        warmup_steps: int,
        total_steps: int,
        weight_decay: float,
        no_decay_params: List[str],
        other_lr_params: List[str] = None,
    ):
        self.lr = lr[0] if isinstance(lr, list) else lr
        self.lr2 = lr[1] if isinstance(lr, list) else lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.weight_decay = weight_decay
        self.no_decay_params = no_decay_params
        self.other_lr_params = other_lr_params or []  # Ensure it's a list

    def group_params(self, module: torch.nn.Module) -> list:
        decay_params = set()
        no_decay_params = set()
        other_lr_params = set()
        # Populate parameter sets
        for n, p in module.named_parameters():
            if any(nd in n for nd in self.no_decay_params):
                no_decay_params.add(p)
            elif any(olr in n for olr in self.other_lr_params):
                other_lr_params.add(p)
            else:
                decay_params.add(p)
        # Group parameters
        optimizer_grouped_parameters = []
        if decay_params:
            optimizer_grouped_parameters.append({"params": list(decay_params), "weight_decay": self.weight_decay})
        if no_decay_params:
            optimizer_grouped_parameters.append({"params": list(no_decay_params), "weight_decay": 0.0})
        if other_lr_params:
            optimizer_grouped_parameters.append({"params": list(other_lr_params), "lr": self.lr2, "weight_decay": self.weight_decay})

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
