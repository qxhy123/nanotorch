"""
Activation functions for neural networks.

This module provides activation functions similar to PyTorch's nn module.
"""

from typing import Any
import numpy as np
import nanotorch.tensor as T
from nanotorch.nn.module import Module


class ReLU(Module):
    """Rectified Linear Unit activation function.

    Applies the element-wise function: ReLU(x) = max(0, x)
    """

    def __init__(self) -> None:
        """Initialize the ReLU activation."""
        super().__init__()

    def forward(self, x: T.Tensor) -> T.Tensor:
        """Forward pass of ReLU activation.

        Args:
            x: Input tensor.

        Returns:
            Tensor with ReLU activation applied element-wise.
        """
        return x.relu()

    def extra_repr(self) -> str:
        """Extra representation string for the module."""
        return ""

    def __repr__(self) -> str:
        """String representation of the ReLU activation."""
        return "ReLU()"


class Sigmoid(Module):
    """Sigmoid activation function.

    Applies the element-wise function: Sigmoid(x) = 1 / (1 + exp(-x))
    """

    def __init__(self) -> None:
        """Initialize the Sigmoid activation."""
        super().__init__()

    def forward(self, x: T.Tensor) -> T.Tensor:
        """Forward pass of Sigmoid activation.

        Args:
            x: Input tensor.

        Returns:
            Tensor with Sigmoid activation applied element-wise.
        """
        return x.sigmoid()

    def extra_repr(self) -> str:
        """Extra representation string for the module."""
        return ""

    def __repr__(self) -> str:
        """String representation of the Sigmoid activation."""
        return "Sigmoid()"


class Tanh(Module):
    """Hyperbolic tangent activation function.

    Applies the element-wise function: Tanh(x) = tanh(x)
    """

    def __init__(self) -> None:
        """Initialize the Tanh activation."""
        super().__init__()

    def forward(self, x: T.Tensor) -> T.Tensor:
        """Forward pass of Tanh activation.

        Args:
            x: Input tensor.

        Returns:
            Tensor with Tanh activation applied element-wise.
        """
        return x.tanh()

    def extra_repr(self) -> str:
        """Extra representation string for the module."""
        return ""

    def __repr__(self) -> str:
        """String representation of the Tanh activation."""
        return "Tanh()"


class Softmax(Module):
    """Softmax activation function.

    Applies the softmax function along a dimension.

    Args:
        dim: Dimension along which softmax will be computed (default: -1).
    """

    def __init__(self, dim: int = -1) -> None:
        """Initialize the Softmax activation."""
        super().__init__()
        self.dim = dim

    def forward(self, x: T.Tensor) -> T.Tensor:
        """Forward pass of Softmax activation.

        Args:
            x: Input tensor.

        Returns:
            Tensor with softmax applied along specified dimension.
        """
        return x.softmax(dim=self.dim)

    def extra_repr(self) -> str:
        """Extra representation string for the module."""
        return f"dim={self.dim}"

    def __repr__(self) -> str:
        """String representation of the Softmax activation."""
        return f"Softmax(dim={self.dim})"


class LogSoftmax(Module):
    """Log-softmax activation function.

    Applies the log-softmax function along a dimension.

    Args:
        dim: Dimension along which log-softmax will be computed (default: -1).
    """

    def __init__(self, dim: int = -1) -> None:
        """Initialize the LogSoftmax activation."""
        super().__init__()
        self.dim = dim

    def forward(self, x: T.Tensor) -> T.Tensor:
        """Forward pass of LogSoftmax activation.

        Args:
            x: Input tensor.

        Returns:
            Tensor with log-softmax applied along specified dimension.
        """
        return x.log_softmax(dim=self.dim)

    def extra_repr(self) -> str:
        """Extra representation string for the module."""
        return f"dim={self.dim}"

    def __repr__(self) -> str:
        """String representation of the LogSoftmax activation."""
        return f"LogSoftmax(dim={self.dim})"


# Functional versions of activation functions
def relu(x: T.Tensor) -> T.Tensor:
    """Functional interface for ReLU activation."""
    return x.relu()


def sigmoid(x: T.Tensor) -> T.Tensor:
    """Functional interface for Sigmoid activation."""
    return x.sigmoid()


def tanh(x: T.Tensor) -> T.Tensor:
    """Functional interface for Tanh activation."""
    return x.tanh()


def softmax(x: T.Tensor, dim: int = -1) -> T.Tensor:
    """Functional interface for softmax activation.

    Args:
        x: Input tensor.
        dim: Dimension along which softmax will be computed.

    Returns:
        Tensor with softmax applied along specified dimension.
    """
    return x.softmax(dim=dim)


def log_softmax(x: T.Tensor, dim: int = -1) -> T.Tensor:
    """Functional interface for log-softmax activation.

    Args:
        x: Input tensor.
        dim: Dimension along which log-softmax will be computed.

    Returns:
        Tensor with log-softmax applied along specified dimension.
    """
    return x.log_softmax(dim=dim)


class LeakyReLU(Module):
    """Leaky Rectified Linear Unit activation function.

    Args:
        negative_slope: Slope for negative values. Default: 0.01.

    Formula: LeakyReLU(x) = max(0, x) + negative_slope * min(0, x)
    """

    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: T.Tensor) -> T.Tensor:
        return x.leaky_relu(self.negative_slope)

    def extra_repr(self) -> str:
        return f"negative_slope={self.negative_slope}"

    def __repr__(self) -> str:
        return f"LeakyReLU({self.extra_repr()})"


class ELU(Module):
    """Exponential Linear Unit activation function.

    Args:
        alpha: Scaling factor for negative values. Default: 1.0.

    Formula: ELU(x) = x if x > 0 else alpha * (exp(x) - 1)
    """

    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: T.Tensor) -> T.Tensor:
        return x.elu(self.alpha)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}"

    def __repr__(self) -> str:
        return f"ELU({self.extra_repr()})"


class GELU(Module):
    """Gaussian Error Linear Unit activation function.

    Uses the fast approximation: GELU(x) ≈ x * sigmoid(1.702 * x).
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: T.Tensor) -> T.Tensor:
        return x.gelu()

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        return "GELU()"


class SiLU(Module):
    """Sigmoid Linear Unit activation function (also known as Swish).

    Formula: SiLU(x) = x * sigmoid(x)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: T.Tensor) -> T.Tensor:
        return x.silu()

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        return "SiLU()"


class PReLU(Module):
    """Parametric Rectified Linear Unit activation function.

    Args:
        num_parameters: Number of parameters (1 or num_channels). Default: 1.
        init: Initial value for the parameter. Default: 0.25.

    Formula: PReLU(x) = max(0, x) + weight * min(0, x)
    """

    def __init__(self, num_parameters: int = 1, init: float = 0.25) -> None:
        super().__init__()
        self.num_parameters = num_parameters
        self.weight = T.Tensor(np.full((num_parameters,), init, dtype=np.float32), requires_grad=True)
        self.register_parameter("weight", self.weight)

    def forward(self, x: T.Tensor) -> T.Tensor:
        if self.num_parameters == 1:
            return x.prelu(self.weight)
        else:
            weight = self.weight.reshape((1, -1, 1, 1))
            return x.prelu(weight)

    def extra_repr(self) -> str:
        return f"num_parameters={self.num_parameters}"

    def __repr__(self) -> str:
        return f"PReLU({self.extra_repr()})"


class Softplus(Module):
    """Softplus activation function.

    Formula: Softplus(x) = log(1 + exp(x))
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: T.Tensor) -> T.Tensor:
        return x.softplus()

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        return "Softplus()"


class Hardswish(Module):
    """Hard Swish activation function.

    Formula: Hardswish(x) = x * relu6(x + 3) / 6
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: T.Tensor) -> T.Tensor:
        return x.hardswish()

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        return "Hardswish()"


class Hardsigmoid(Module):
    """Hard Sigmoid activation function.

    Formula: Hardsigmoid(x) = relu6(x + 3) / 6
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: T.Tensor) -> T.Tensor:
        return x.hardsigmoid()

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        return "Hardsigmoid()"


# Functional versions of new activation functions
def leaky_relu(x: T.Tensor, negative_slope: float = 0.01) -> T.Tensor:
    return x.leaky_relu(negative_slope)


def elu(x: T.Tensor, alpha: float = 1.0) -> T.Tensor:
    return x.elu(alpha)


def gelu(x: T.Tensor) -> T.Tensor:
    return x.gelu()


def silu(x: T.Tensor) -> T.Tensor:
    return x.silu()


def softplus(x: T.Tensor) -> T.Tensor:
    return x.softplus()


def hardswish(x: T.Tensor) -> T.Tensor:
    return x.hardswish()


def hardsigmoid(x: T.Tensor) -> T.Tensor:
    return x.hardsigmoid()


class Flatten(Module):
    """Flattens a contiguous range of dims into a single dimension.

    Args:
        start_dim: First dim to flatten (default: 1).
        end_dim: Last dim to flatten (default: -1).

    Shape:
        - Input: (N, *dims)
        - Output: (N, product of dims from start_dim to end_dim)

    Example:
        >>> m = Flatten()
        >>> x = Tensor.randn((32, 3, 28, 28))
        >>> output = m(x)
        >>> output.shape
        (32, 2352)
    """

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: T.Tensor) -> T.Tensor:
        return x.flatten(self.start_dim, self.end_dim)

    def extra_repr(self) -> str:
        return f"start_dim={self.start_dim}, end_dim={self.end_dim}"

    def __repr__(self) -> str:
        return f"Flatten({self.extra_repr()})"


class Identity(Module):
    """Identity module that returns input unchanged.

    Useful as a placeholder or for residual connections.

    Args:
        *args: Ignored positional arguments.
        **kwargs: Ignored keyword arguments.

    Example:
        >>> m = Identity()
        >>> x = Tensor.randn((32, 10))
        >>> output = m(x)
        >>> output.shape
        (32, 10)
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def forward(self, x: T.Tensor) -> T.Tensor:
        return x

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        return "Identity()"
