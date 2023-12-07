import torch
from torch.optim.lr_scheduler import LRScheduler


class LinearSchedulerWithWarmup(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_warmup_steps: int,
        num_training_steps: int,
        last_epoch: int = -1,
        verbose: bool = False,
        **kwargs,
    ):
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        def scheduler_fn(current_step):
            if current_step < self.num_warmup_steps:
                return current_step / max(1, self.num_warmup_steps)
            return max(
                0.0,
                float(self.num_training_steps - current_step)
                / float(max(1, self.num_training_steps - self.num_warmup_steps)),
            )

        return [base_lr * scheduler_fn(self.last_epoch) for base_lr in self.base_lrs]


class LinearScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        num_training_steps: int,
        last_epoch: int = -1,
        verbose: bool = False,
        **kwargs,
    ):
        self.num_training_steps = num_training_steps
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        def scheduler_fn(current_step):
            # if current_step < self.num_warmup_steps:
            #     return current_step / max(1, self.num_warmup_steps)
            return max(
                0.0,
                float(self.num_training_steps - current_step)
                / float(max(1, self.num_training_steps)),
            )

        return [base_lr * scheduler_fn(self.last_epoch) for base_lr in self.base_lrs]
