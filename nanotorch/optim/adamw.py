"""
AdamW optimizer.

This module provides the AdamW optimizer with decoupled weight decay,
similar to PyTorch's optim.AdamW.
"""

from typing import Iterator, Tuple, Dict, Any
import numpy as np
import nanotorch.tensor as T
from nanotorch.optim.optimizer import Optimizer


class AdamW(Optimizer):
    """AdamW optimizer with decoupled weight decay.

    Implements AdamW algorithm from "Decoupled Weight Decay Regularization"
    (Loshchilov & Hutter, 2017). This is a modification of Adam where
    weight decay is applied directly to parameters rather than added to
    the gradient.

    Args:
        params: Iterator of parameters to optimize.
        lr: Learning rate (default: 0.001).
        betas: Coefficients for computing running averages of gradient and its square
               (default: (0.9, 0.999)).
        eps: Term added to denominator to improve numerical stability (default: 1e-8).
        weight_decay: Weight decay (L2 penalty) (default: 1e-2).
        amsgrad: Whether to use the AMSGrad variant of Adam (default: False).

    Example:
        >>> optimizer = AdamW(model.parameters(), lr=0.001)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params: Iterator[T.Tensor],
        lr: float = 0.001,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        amsgrad: bool = False,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "amsgrad": amsgrad,
        }
        super().__init__(params, defaults)

        self.state: Dict[T.Tensor, Dict[str, Any]] = {}

    def step(self) -> None:
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            amsgrad = group["amsgrad"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad

                # Apply decoupled weight decay directly to parameters
                if weight_decay != 0:
                    param.data = param.data * (1 - lr * weight_decay)

                # Get or initialize state
                param_state = self.state.get(param)
                if param_state is None:
                    param_state = {
                        "step": 0,
                        "exp_avg": T.Tensor.zeros(param.shape, requires_grad=False),
                        "exp_avg_sq": T.Tensor.zeros(param.shape, requires_grad=False),
                    }
                    if amsgrad:
                        param_state["max_exp_avg_sq"] = T.Tensor.zeros(
                            param.shape, requires_grad=False
                        )
                    self.state[param] = param_state

                # Increment step
                param_state["step"] += 1
                step = param_state["step"]
                exp_avg = param_state["exp_avg"]
                exp_avg_sq = param_state["exp_avg_sq"]

                # Update biased first moment estimate
                exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                param_state["exp_avg"] = exp_avg

                # Update biased second raw moment estimate
                exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * (grad * grad)
                param_state["exp_avg_sq"] = exp_avg_sq

                # Bias correction
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                if amsgrad:
                    # Maintains the maximum of all second moment running averages
                    max_exp_avg_sq = param_state["max_exp_avg_sq"]
                    max_exp_avg_sq = T.Tensor(
                        np.maximum(max_exp_avg_sq.data, exp_avg_sq.data),
                        requires_grad=False,
                    )
                    param_state["max_exp_avg_sq"] = max_exp_avg_sq

                    # Use max for denominator
                    denom = (max_exp_avg_sq.data / bias_correction2) ** 0.5 + eps
                else:
                    denom = (exp_avg_sq.data / bias_correction2) ** 0.5 + eps

                # Compute step size
                step_size = lr / bias_correction1

                # Update parameter with gradient step
                param.data = param.data - step_size * (exp_avg.data / denom)

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + " ("
        for i, group in enumerate(self.param_groups):
            format_string += f"\nParameter Group {i}"
            for key in ["lr", "betas", "eps", "weight_decay", "amsgrad"]:
                format_string += f"\n    {key}: {group[key]}"
            format_string += f"\n    params: {len(group['params'])} tensors"
        format_string += "\n)"
        return format_string