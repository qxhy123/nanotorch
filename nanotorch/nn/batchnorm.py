"""
Batch normalization layer for nanotorch.

This module provides BatchNorm2d layer similar to PyTorch's nn.BatchNorm2d.
"""

from typing import Optional
from nanotorch.tensor import Tensor
from .normalization import _BatchNorm


class BatchNorm2d(_BatchNorm):
    """Batch Normalization layer for 2D inputs (4D tensors).

    Applies Batch Normalization over a 4D input (N, C, H, W) as described in the
    paper "Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift".

    Args:
        num_features: Number of features/channels C in the input.
        eps: A value added to the denominator for numerical stability. Default: 1e-5.
        momentum: The value used for the running_mean and running_var computation.
            Can be set to None for cumulative moving average (i.e., simple average).
            Default: 0.1.
        affine: A boolean that when set to True, this module has learnable
            affine parameters.
            Default: True.
        track_running_stats: A boolean that when set to True, this module tracks
            the running mean and variance; when set to False, this module does not
            track such statistics, and uses batch statistics in both training and
            eval modes. Default: True.

    Shape:
        - Input: (N, C, H, W)
        - Output: (N, C, H, W) (same shape as input)

    Examples:
        >>> bn = BatchNorm2d(64)
        >>> x = Tensor.randn((32, 64, 28, 28))
        >>> output = bn(x)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__(
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            num_spatial_dims=2,
        )

    def _check_input_dim(self, x: Tensor) -> None:
        if x.ndim != 4:
            raise ValueError(
                f"BatchNorm2d expects 4D input (N, C, H, W), got {x.ndim}D"
            )
        if x.shape[1] != self.num_features:
            raise ValueError(
                f"Expected input channels {self.num_features}, got {x.shape[1]}"
            )


# Functional batch normalization (simplified version)
def batch_norm(
    input: Tensor,
    running_mean: Optional[Tensor] = None,
    running_var: Optional[Tensor] = None,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    training: bool = False,
    momentum: float = 0.1,
    eps: float = 1e-5,
) -> Tensor:
    """Functional interface for batch normalization.

    Args:
        input: Input tensor of shape (N, C, H, W).
        running_mean: The running mean tensor of shape (C,).
        running_var: The running variance tensor of shape (C,).
        weight: The weight tensor of shape (C,) (gamma).
        bias: The bias tensor of shape (C,) (beta).
        training: If True, use batch statistics and update running statistics.
            If False, use running statistics.
        momentum: The value used for the running_mean and running_var computation.
        eps: A value added to the denominator for numerical stability.

    Returns:
        Normalized tensor of same shape as input.
    """
    if input.ndim != 4:
        raise ValueError(f"batch_norm expects 4D input (N, C, H, W), got {input.ndim}D")

    N, C, H, W = input.shape

    axes = (0, 2, 3)

    if training:
        mean = input.mean(axis=axes, keepdims=True)
        var = ((input - mean) ** 2).mean(axis=axes, keepdims=True)

        # Update running statistics if provided
        if running_mean is not None and running_var is not None:
            # Squeeze dimensions (0, 2, 3) to get shape (C,)
            mean_squeezed = mean.data.squeeze(axis=(0, 2, 3))
            var_squeezed = var.data.squeeze(axis=(0, 2, 3))
            running_mean.data = (
                1 - momentum
            ) * running_mean.data + momentum * mean_squeezed
            running_var.data = (1 - momentum) * running_var.data + momentum * var_squeezed
    else:
        if running_mean is None or running_var is None:
            raise ValueError(
                "running_mean and running_var must be provided in eval mode"
            )
        # Reshape running stats from (C,) to (1, C, 1, 1) for broadcasting
        mean = running_mean.reshape((1, C, 1, 1))
        var = running_var.reshape((1, C, 1, 1))

    # Normalize
    normalized = (input - mean) / (var + eps) ** 0.5

    # Apply affine transformation if weight and bias are provided
    if weight is not None:
        # Reshape weight from (C,) to (1, C, 1, 1) for broadcasting
        weight_reshaped = weight.reshape((1, C, 1, 1))
        normalized = weight_reshaped * normalized
    if bias is not None:
        # Reshape bias from (C,) to (1, C, 1, 1) for broadcasting
        bias_reshaped = bias.reshape((1, C, 1, 1))
        normalized = normalized + bias_reshaped

    return normalized
