"""
RMSprop optimizer.

This module provides the RMSprop optimizer similar to PyTorch's optim.RMSprop.
"""

from typing import Iterator, Dict, Any
import numpy as np
import nanotorch.tensor as T
from nanotorch.optim.optimizer import Optimizer


class RMSprop(Optimizer):
    """RMSprop optimizer.

    Implements RMSprop algorithm from "Lecture 6e - rmsprop: Divide the gradient by a
    running average of its recent magnitude" by Geoffrey Hinton.

    Args:
        params: Iterator of parameters to optimize.
        lr: Learning rate (default: 0.01).
        alpha: Smoothing constant (default: 0.99).
        eps: Term added to denominator to improve numerical stability (default: 1e-8).
        weight_decay: Weight decay (L2 penalty) (default: 0).
        momentum: Momentum factor (default: 0).
        centered: If True, compute the centered RMSProp (default: False).

    Example:
        >>> optimizer = RMSprop(model.parameters(), lr=0.01)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
    """

    def __init__(
        self,
        params: Iterator[T.Tensor],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0,
        momentum: float = 0,
        centered: bool = False,
    ) -> None:
        """Initialize RMSprop optimizer."""
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if alpha < 0:
            raise ValueError(f"Invalid alpha value: {alpha}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if momentum < 0:
            raise ValueError(f"Invalid momentum value: {momentum}")

        defaults = {
            "lr": lr,
            "alpha": alpha,
            "eps": eps,
            "weight_decay": weight_decay,
            "momentum": momentum,
            "centered": centered,
        }
        super().__init__(params, defaults)

        # Initialize state buffers
        self.state: Dict[T.Tensor, Dict[str, Any]] = {}

    def step(self) -> None:
        """Perform a single optimization step."""
        for group in self.param_groups:
            lr = group["lr"]
            alpha = group["alpha"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            centered = group["centered"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad

                # Apply weight decay (L2 regularization)
                if weight_decay != 0:
                    grad = grad + weight_decay * param

                # Get or initialize state
                param_state = self.state.get(param)
                if param_state is None:
                    param_state = {
                        "square_avg": T.Tensor.zeros(param.shape, requires_grad=False)
                    }
                    if momentum > 0:
                        param_state["momentum_buffer"] = T.Tensor.zeros(
                            param.shape, requires_grad=False
                        )
                    if centered:
                        param_state["grad_avg"] = T.Tensor.zeros(
                            param.shape, requires_grad=False
                        )
                    self.state[param] = param_state

                square_avg = param_state["square_avg"]

                # Update running average of squared gradients
                square_avg = alpha * square_avg + (1 - alpha) * (grad * grad)
                param_state["square_avg"] = square_avg

                if centered:
                    # Update running average of gradients
                    grad_avg = param_state["grad_avg"]
                    grad_avg = alpha * grad_avg + (1 - alpha) * grad
                    param_state["grad_avg"] = grad_avg

                    # Compute centered variance
                    avg = square_avg.data - grad_avg.data * grad_avg.data
                    # Ensure non-negativity due to numerical errors
                    avg = np.maximum(avg, 0)
                    denom = avg**0.5 + eps
                else:
                    denom = square_avg.data**0.5 + eps

                # Apply momentum
                if momentum > 0:
                    momentum_buffer = param_state["momentum_buffer"]
                    momentum_buffer = momentum * momentum_buffer + grad.data / denom
                    param_state["momentum_buffer"] = momentum_buffer
                    update = momentum_buffer.data
                else:
                    update = grad.data / denom

                # Update parameter
                param.data = param.data - lr * update

    def __repr__(self) -> str:
        """String representation of the RMSprop optimizer."""
        format_string = self.__class__.__name__ + " ("
        for i, group in enumerate(self.param_groups):
            format_string += f"\nParameter Group {i}"
            for key in ["lr", "alpha", "eps", "weight_decay", "momentum", "centered"]:
                format_string += f"\n    {key}: {group[key]}"
            format_string += f"\n    params: {len(group['params'])} tensors"
        format_string += "\n)"
        return format_string
