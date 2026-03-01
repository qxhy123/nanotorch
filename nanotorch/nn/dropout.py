"""
Dropout layer implementation.

This module provides the Dropout layer similar to PyTorch's nn.Dropout.
"""


import numpy as np
from numpy.typing import NDArray
import nanotorch.tensor as T
from nanotorch.nn.module import Module
from nanotorch.tensor import no_grad


class Dropout(Module):
    """Dropout layer.

    Randomly sets input elements to zero with probability p during training.
    Scales the remaining elements by 1/(1-p) to maintain expected value.

    Attributes:
        p: Dropout probability (probability of setting an element to zero).
        inplace: Not used in this implementation (for API compatibility).
    """

    def __init__(self, p: float = 0.5, inplace: bool = False):
        """Initialize a Dropout layer.

        Args:
            p: Dropout probability (probability of setting an element to zero).
                Must be between 0 and 1.
            inplace: Not used in this implementation (for API compatibility).
        """
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError(f"Dropout probability must be between 0 and 1, got {p}")

        self.p = p
        self.inplace = inplace

    def forward(self, x: T.Tensor) -> T.Tensor:
        """Forward pass of dropout layer.

        Args:
            x: Input tensor.

        Returns:
            Output tensor with dropout applied if in training mode,
            otherwise returns the input unchanged.
        """
        if self.training:
            with no_grad():
                # Generate random mask using numpy for efficiency
                rand_array = np.random.rand(*x.shape)
                # Create binary mask: 1 with probability (1-p), 0 with probability p
                mask: NDArray[np.float32] = (rand_array > self.p).astype(np.float32)
                # Scale the mask by 1/(1-p) to maintain expected value
                if self.p < 1.0:
                    scale = 1.0 / (1.0 - self.p)
                    scaled_mask_data = mask * scale
                else:
                    # p == 1.0: all zeros, scale is 0 (no gradient)
                    scaled_mask_data = np.zeros_like(x.data)
                scaled_mask = T.Tensor(scaled_mask_data, requires_grad=False)

            # Apply dropout via element-wise multiplication
            # This creates a tensor with _op='mul' and parents (x, scaled_mask)
            # Gradient will flow through multiplication automatically
            output = x * scaled_mask
            return output
        else:
            # Evaluation mode: identity function
            return x

    def extra_repr(self) -> str:
        """Extra representation string for the module."""
        return f"p={self.p}, inplace={self.inplace}"

    def __repr__(self) -> str:
        """String representation of the Dropout layer."""
        return f"Dropout({self.extra_repr()})"


# Functional interface for dropout
def dropout(x: T.Tensor, p: float = 0.5, training: bool = True) -> T.Tensor:
    """Functional interface for dropout.

    Args:
        x: Input tensor.
        p: Dropout probability.
        training: Whether to apply dropout (training mode).

    Returns:
        Tensor with dropout applied.
    """
    if training:
        with no_grad():
            rand_array = np.random.rand(*x.shape)
            mask: NDArray[np.float32] = (rand_array > p).astype(np.float32)
            if p < 1.0:
                scale = 1.0 / (1.0 - p)
                scaled_mask_data = mask * scale
            else:
                scaled_mask_data = np.zeros_like(x.data)
            scaled_mask = T.Tensor(scaled_mask_data, requires_grad=False)
        return x * scaled_mask
    else:
        return x
