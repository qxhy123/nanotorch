"""
Linear layer implementation.

This module provides the Linear layer similar to PyTorch's nn.Linear.
"""

import math
from typing import Optional
import nanotorch.tensor as T
from nanotorch.nn.module import Module


class Linear(Module):
    """Linear (fully connected) layer.

    Applies a linear transformation to the incoming data: y = xW^T + b

    Attributes:
        in_features: Size of each input sample.
        out_features: Size of each output sample.
        weight: Weight tensor of shape (out_features, in_features).
        bias: Bias tensor of shape (out_features,) or None.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[str] = None,
        dtype: Optional[str] = None,
    ) -> None:
        """Initialize a Linear layer.

        Args:
            in_features: Size of each input sample.
            out_features: Size of each output sample.
            bias: If set to False, the layer will not learn an additive bias.
            device: Not used in this implementation (for API compatibility).
            dtype: Not used in this implementation (for API compatibility).
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weight tensor
        # Using Kaiming initialization (He initialization) for ReLU activations
        scale = math.sqrt(2.0 / in_features)
        weight_data = (
            T.Tensor.randn((out_features, in_features), requires_grad=True) * scale
        )
        self.weight = weight_data
        self.register_parameter("weight", self.weight)

        # Initialize bias tensor if requested
        self.bias: Optional[T.Tensor] = None
        if bias:
            bias_data = T.Tensor.zeros((out_features,), requires_grad=True)
            self.bias = bias_data
            self.register_parameter("bias", self.bias)

    def forward(self, x: T.Tensor) -> T.Tensor:
        """Forward pass of the linear layer.

        Args:
            x: Input tensor of shape (*, in_features) where * is any number of
                additional dimensions including batch and sequence dimensions.

        Returns:
            Output tensor of shape (*, out_features).

        Raises:
            RuntimeError: If input shape doesn't match in_features.
        """
        if x.shape[-1] != self.in_features:
            raise RuntimeError(
                f"Linear layer expects input with {self.in_features} features, "
                f"got {x.shape[-1]} features"
            )

        output = x.matmul(self.weight.T)

        if self.bias is not None:
            output = output + self.bias

        return output

    def extra_repr(self) -> str:
        """Extra representation string for the module."""
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )

    def __repr__(self) -> str:
        """String representation of the Linear layer."""
        return f"Linear({self.extra_repr()})"
