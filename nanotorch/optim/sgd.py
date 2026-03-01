"""
Stochastic Gradient Descent (SGD) optimizer.

This module provides the SGD optimizer similar to PyTorch's optim.SGD.
"""

from typing import Iterator, Dict, Any
import nanotorch.tensor as T
from nanotorch.optim.optimizer import Optimizer


class SGD(Optimizer):
    """Stochastic Gradient Descent (SGD) optimizer.

    Implements standard SGD with optional momentum and weight decay.

    Args:
        params: Iterator of parameters to optimize.
        lr: Learning rate (required).
        momentum: Momentum factor (default: 0).
        dampening: Dampening for momentum (default: 0).
        weight_decay: Weight decay (L2 penalty) (default: 0).
        nesterov: Enables Nesterov momentum (default: False).

    Example:
        >>> optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params: Iterator[T.Tensor],
        lr: float,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
    ) -> None:
        """Initialize SGD optimizer."""
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        defaults = {
            "lr": lr,
            "momentum": momentum,
            "dampening": dampening,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
        }
        super().__init__(params, defaults)

        # Initialize velocity buffers for momentum
        self.state: Dict[T.Tensor, Dict[str, Any]] = {}

    def step(self) -> None:
        """Perform a single optimization step."""
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad

                # Apply weight decay (L2 regularization)
                if weight_decay != 0:
                    grad = grad + weight_decay * param

                # Apply momentum
                if momentum != 0:
                    param_state = self.state.get(param)
                    if param_state is None:
                        # Initialize velocity buffer
                        param_state = {
                            "velocity": T.Tensor.zeros(param.shape, requires_grad=False)
                        }
                        self.state[param] = param_state

                    velocity = param_state["velocity"]

                    # Update velocity: v = momentum * v + (1 - dampening) * grad
                    velocity = momentum * velocity + (1 - dampening) * grad
                    param_state["velocity"] = velocity

                    if nesterov:
                        # Nesterov momentum: use lookahead velocity
                        grad = grad + momentum * velocity
                    else:
                        grad = velocity

                # Update parameter: param = param - lr * grad
                param.data = param.data - lr * grad.data

    def __repr__(self) -> str:
        """String representation of the SGD optimizer."""
        format_string = self.__class__.__name__ + " ("
        for i, group in enumerate(self.param_groups):
            format_string += f"\nParameter Group {i}"
            for key in ["lr", "momentum", "dampening", "weight_decay", "nesterov"]:
                format_string += f"\n    {key}: {group[key]}"
            format_string += f"\n    params: {len(group['params'])} tensors"
        format_string += "\n)"
        return format_string
