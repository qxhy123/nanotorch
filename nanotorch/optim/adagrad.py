"""
Adagrad optimizer.

This module provides the Adagrad optimizer similar to PyTorch's optim.Adagrad.
"""

from typing import Iterator, Dict, Any
import numpy as np

import nanotorch.tensor as T
from nanotorch.optim.optimizer import Optimizer


class Adagrad(Optimizer):
    """Adagrad optimizer.

    Implements Adagrad algorithm from "Adaptive Subgradient Methods for Online Learning
    and Stochastic Optimization" by Duchi et al.

    Args:
        params: Iterator of parameters to optimize.
        lr: Learning rate (default: 0.01).
        lr_decay: Learning rate decay (default: 0).
        weight_decay: Weight decay (L2 penalty) (default: 0).
        initial_accumulator_value: Initial value for accumulator (default: 0).
        eps: Term added to denominator to improve numerical stability (default: 1e-10).

    Example:
        >>> optimizer = Adagrad(model.parameters(), lr=0.01)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params: Iterator[T.Tensor],
        lr: float = 0.01,
        lr_decay: float = 0,
        weight_decay: float = 0,
        initial_accumulator_value: float = 0,
        eps: float = 1e-10,
    ) -> None:
        """Initialize Adagrad optimizer."""
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if lr_decay < 0:
            raise ValueError(f"Invalid lr_decay value: {lr_decay}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if initial_accumulator_value < 0:
            raise ValueError(
                f"Invalid initial_accumulator_value: {initial_accumulator_value}"
            )
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        defaults = {
            "lr": lr,
            "lr_decay": lr_decay,
            "weight_decay": weight_decay,
            "initial_accumulator_value": initial_accumulator_value,
            "eps": eps,
        }
        super().__init__(params, defaults)

        # Initialize state buffers
        self.state: Dict[T.Tensor, Dict[str, Any]] = {}

        # Initialize accumulators with initial_accumulator_value
        for group in self.param_groups:
            for param in group["params"]:
                param_state = self.state.get(param)
                if param_state is None:
                    param_state = {
                        "step": 0,
                        "sum": T.Tensor(
                            np.full(
                                param.shape, initial_accumulator_value, dtype=np.float32
                            ),
                            requires_grad=False,
                        ),
                    }
                    self.state[param] = param_state

    def step(self) -> None:
        """Perform a single optimization step."""
        for group in self.param_groups:
            lr = group["lr"]
            lr_decay = group["lr_decay"]
            weight_decay = group["weight_decay"]
            eps = group["eps"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad

                # Apply weight decay (L2 regularization)
                if weight_decay != 0:
                    grad = grad + weight_decay * param

                param_state = self.state[param]
                param_state["step"] += 1
                step = param_state["step"]
                sum_ = param_state["sum"]

                # Accumulate squared gradients
                sum_ = sum_ + grad * grad
                param_state["sum"] = sum_

                # Compute learning rate with decay
                clr = lr / (1 + (step - 1) * lr_decay)

                # Update parameter
                param.data = param.data - clr * grad.data / (sum_.data**0.5 + eps)

    def __repr__(self) -> str:
        """String representation of the Adagrad optimizer."""
        format_string = self.__class__.__name__ + " ("
        for i, group in enumerate(self.param_groups):
            format_string += f"\nParameter Group {i}"
            for key in [
                "lr",
                "lr_decay",
                "weight_decay",
                "initial_accumulator_value",
                "eps",
            ]:
                format_string += f"\n    {key}: {group[key]}"
            format_string += f"\n    params: {len(group['params'])} tensors"
        format_string += "\n)"
        return format_string
