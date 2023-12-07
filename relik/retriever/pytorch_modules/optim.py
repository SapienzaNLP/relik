import math

import torch
from torch.optim import Optimizer


class RAdamW(Optimizer):
    r"""Implements RAdamW algorithm.

    RAdam from `On the Variance of the Adaptive Learning Rate and Beyond
    <https://arxiv.org/abs/1908.03265v1>`_

    * `Adam: A Method for Stochastic Optimization
      <https://arxiv.org/abs/1412.6980>`_
    * `Decoupled Weight Decay Regularization
      <https://arxiv.org/abs/1711.05101>`_
    * `On the Convergence of Adam and Beyond
      <https://openreview.net/forum?id=ryQu7f-RZ>`_
    * `On the Variance of the Adaptive Learning Rate and Beyond
      <https://arxiv.org/abs/1908.03265v1>`_

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
    """

    def __init__(
        self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(RAdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                # Perform optimization step
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                lr = group["lr"]
                if "rho_inf" not in group:
                    group["rho_inf"] = 2 / (1 - beta2) - 1
                rho_inf = group["rho_inf"]

                state["step"] += 1
                t = state["step"]

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                rho_t = rho_inf - ((2 * t * (beta2**t)) / (1 - beta2**t))

                # Perform stepweight decay
                p.data.mul_(1 - lr * group["weight_decay"])

                if rho_t >= 5:
                    var = exp_avg_sq.sqrt().add_(eps)
                    r = math.sqrt(
                        (1 - beta2**t)
                        * ((rho_t - 4) * (rho_t - 2) * rho_inf)
                        / ((rho_inf - 4) * (rho_inf - 2) * rho_t)
                    )

                    p.data.addcdiv_(exp_avg, var, value=-lr * r / (1 - beta1**t))
                else:
                    p.data.add_(exp_avg, alpha=-lr / (1 - beta1**t))

        return loss
