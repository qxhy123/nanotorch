"""
Normalization layers for nanotorch.

This module provides normalization layers similar to PyTorch's nn module,
including LayerNorm, GroupNorm, InstanceNorm, etc.
"""

from typing import Union, Tuple, Optional
from nanotorch.nn.module import Module
from nanotorch.tensor import Tensor
from nanotorch.autograd import LayerNormFunction, GroupNormFunction, InstanceNormFunction


class _BatchNorm(Module):
    """Base class for BatchNorm layers (BatchNorm1d, BatchNorm2d, BatchNorm3d).
    
    This class implements common functionality for batch normalization layers.
    Subclasses should validate input dimensions and call super().__init__ with
    appropriate parameters.
    
    Args:
        num_features: Number of features/channels C in input.
        eps: A value added to denominator for numerical stability.
        momentum: Value used for running_mean/var computation. None = cumulative average.
        affine: Whether to learn affine parameters (gamma, beta).
        track_running_stats: Whether to track running mean/variance.
        num_spatial_dims: Number of spatial dimensions (1 for 1D, 2 for 2D, 3 for 3D).
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: Optional[float] = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        num_spatial_dims: int = 0,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.num_spatial_dims = num_spatial_dims
        
        # Initialize learnable parameters if affine=True
        self.gamma: Optional[Tensor] = None
        self.beta: Optional[Tensor] = None
        if self.affine:
            self.gamma = Tensor.ones((num_features,), requires_grad=True)
            self.beta = Tensor.zeros((num_features,), requires_grad=True)
            self.register_parameter("gamma", self.gamma)
            self.register_parameter("beta", self.beta)
        
        # Initialize running statistics if track_running_stats=True
        self.running_mean: Optional[Tensor] = None
        self.running_var: Optional[Tensor] = None
        self.num_batches_tracked: Optional[Tensor] = None
        if self.track_running_stats:
            self.running_mean = Tensor.zeros((num_features,), requires_grad=False)
            self.running_var = Tensor.ones((num_features,), requires_grad=False)
            self.num_batches_tracked = Tensor([0], requires_grad=False)
            self.register_buffer("running_mean", self.running_mean)
            self.register_buffer("running_var", self.running_var)
            self.register_buffer("num_batches_tracked", self.num_batches_tracked)
    
    def _check_input_dim(self, x: Tensor) -> None:
        """Check input dimensions (to be implemented by subclasses)."""
        raise NotImplementedError
    
    def _get_reduction_axes(self, x: Tensor) -> Tuple[int, ...]:
        """Get axes to reduce over for batch normalization.
        
        For input shape (N, C, *spatial), reduces over batch and spatial dimensions.
        Returns tuple of axes indices.
        """
        # Axes: 0 (batch) + all spatial dimensions (starting from 2)
        return (0,) + tuple(range(2, x.ndim))
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for batch normalization."""
        self._check_input_dim(x)

        C = x.shape[1]
        axes = self._get_reduction_axes(x)
        
        if self.training and self.track_running_stats:
            assert (
                self.running_mean is not None
                and self.running_var is not None
                and self.num_batches_tracked is not None
            )
            # Compute batch statistics (keep dimensions for broadcasting)
            mean = x.mean(axis=axes, keepdims=True)
            var = ((x - mean) ** 2).mean(axis=axes, keepdims=True)
            
            # Squeeze all reduced dimensions for running statistics (shape: (C,))
            # axes are in ascending order, we can squeeze them
            mean_squeezed = mean.squeeze(axis=axes)
            var_squeezed = var.squeeze(axis=axes)
            
            # Update running statistics
            if self.momentum is not None:
                self.running_mean.data = (
                    1 - self.momentum
                ) * self.running_mean.data + self.momentum * mean_squeezed.data
                self.running_var.data = (
                    1 - self.momentum
                ) * self.running_var.data + self.momentum * var_squeezed.data
            else:
                # Cumulative moving average
                n = self.num_batches_tracked.data[0] + 1
                self.running_mean.data = (
                    self.running_mean.data * (n - 1) + mean_squeezed.data
                ) / n
                self.running_var.data = (
                    self.running_var.data * (n - 1) + var_squeezed.data
                ) / n
                self.num_batches_tracked.data[0] = n
            
            # Use batch statistics for normalization
            x_normalized = (x - mean) / (var + self.eps) ** 0.5
        else:
            # Use running statistics
            if self.track_running_stats:
                assert self.running_mean is not None and self.running_var is not None
                # Reshape running stats from (C,) to (1, C, 1, 1, ...) for broadcasting
                # Shape: (1, C) + (1,) * (x.ndim - 2)
                broadcast_shape = (1, C) + (1,) * (x.ndim - 2)
                mean = self.running_mean.reshape(broadcast_shape)
                var = self.running_var.reshape(broadcast_shape)
            else:
                # Fallback to batch statistics (when track_running_stats=False)
                mean = x.mean(axis=axes, keepdims=True)
                var = ((x - mean) ** 2).mean(axis=axes, keepdims=True)
            
            x_normalized = (x - mean) / (var + self.eps) ** 0.5
        
        # Apply affine transformation if enabled
        if self.affine:
            assert self.gamma is not None and self.beta is not None
            # Reshape gamma and beta for broadcasting
            broadcast_shape = (1, C) + (1,) * (x.ndim - 2)
            gamma_reshaped = self.gamma.reshape(broadcast_shape)
            beta_reshaped = self.beta.reshape(broadcast_shape)
            x_normalized = gamma_reshaped * x_normalized + beta_reshaped
        
        return x_normalized
    
    def extra_repr(self) -> str:
        """Extra representation string."""
        s = f"{self.num_features}, eps={self.eps}"
        if self.momentum is not None:
            s += f", momentum={self.momentum}"
        s += f", affine={self.affine}, track_running_stats={self.track_running_stats}"
        return s


class BatchNorm1d(_BatchNorm):
    """Batch Normalization layer for 1D inputs (2D or 3D tensors).

    Applies Batch Normalization over a 2D input (N, C) or 3D input (N, C, L)
    as described in the paper "Batch Normalization: Accelerating Deep Network
    Training by Reducing Internal Covariate Shift".

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
        - Input: (N, C) or (N, C, L)
        - Output: same shape as input

    Examples:
        >>> bn = BatchNorm1d(64)
        >>> x = Tensor.randn((32, 64))
        >>> output = bn(x)
        >>> x2 = Tensor.randn((16, 64, 10))
        >>> output2 = bn(x2)
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
            num_spatial_dims=1,
        )

    def _check_input_dim(self, x: Tensor) -> None:
        if x.ndim not in (2, 3):
            raise ValueError(
                f"BatchNorm1d expects 2D or 3D input (got {x.ndim}D)"
            )
        if x.shape[1] != self.num_features:
            raise ValueError(
                f"Expected input channels {self.num_features}, got {x.shape[1]}"
            )


class BatchNorm2d(_BatchNorm):
    """Batch Normalization layer for 2D inputs (4D tensors).

    Applies Batch Normalization over a 4D input (N, C, H, W)
    as described in the paper "Batch Normalization: Accelerating Deep Network
    Training by Reducing Internal Covariate Shift".

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
        - Output: same shape as input

    Examples:
        >>> bn = BatchNorm2d(64)
        >>> x = Tensor.randn((16, 64, 32, 32))
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


class BatchNorm3d(_BatchNorm):
    """Batch Normalization layer for 3D inputs (5D tensors).

    Applies Batch Normalization over a 5D input (N, C, D, H, W)
    as described in the paper "Batch Normalization: Accelerating Deep Network
    Training by Reducing Internal Covariate Shift".

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
        - Input: (N, C, D, H, W)
        - Output: same shape as input

    Examples:
        >>> bn = BatchNorm3d(64)
        >>> x = Tensor.randn((8, 64, 10, 10, 10))
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
            num_spatial_dims=3,
        )

    def _check_input_dim(self, x: Tensor) -> None:
        if x.ndim != 5:
            raise ValueError(
                f"BatchNorm3d expects 5D input (N, C, D, H, W), got {x.ndim}D"
            )
        if x.shape[1] != self.num_features:
            raise ValueError(
                f"Expected input channels {self.num_features}, got {x.shape[1]}"
            )


class _InstanceNorm(Module):
    """Base class for InstanceNorm layers (InstanceNorm1d, InstanceNorm2d, InstanceNorm3d).
    
    This class implements common functionality for instance normalization layers.
    Subclasses should validate input dimensions and call super().__init__ with
    appropriate parameters.
    
    Args:
        num_features: Number of features/channels C in input.
        eps: A value added to denominator for numerical stability.
        momentum: Value used for running_mean/var computation (if track_running_stats).
        affine: Whether to learn affine parameters (gamma, beta). Default: False.
        track_running_stats: Whether to track running mean/variance. Default: False.
        num_spatial_dims: Number of spatial dimensions (1 for 1D, 2 for 2D, 3 for 3D).
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
        num_spatial_dims: int = 0,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.num_spatial_dims = num_spatial_dims
        
        # Initialize learnable parameters if affine=True
        self.gamma: Optional[Tensor] = None
        self.beta: Optional[Tensor] = None
        if self.affine:
            self.gamma = Tensor.ones((num_features,), requires_grad=True)
            self.beta = Tensor.zeros((num_features,), requires_grad=True)
            self.register_parameter("gamma", self.gamma)
            self.register_parameter("beta", self.beta)
        
        # Initialize running statistics if track_running_stats=True
        self.running_mean: Optional[Tensor] = None
        self.running_var: Optional[Tensor] = None
        self.num_batches_tracked: Optional[Tensor] = None
        if self.track_running_stats:
            self.running_mean = Tensor.zeros((num_features,), requires_grad=False)
            self.running_var = Tensor.ones((num_features,), requires_grad=False)
            self.num_batches_tracked = Tensor([0], requires_grad=False)
            self.register_buffer("running_mean", self.running_mean)
            self.register_buffer("running_var", self.running_var)
            self.register_buffer("num_batches_tracked", self.num_batches_tracked)
    
    def _check_input_dim(self, x: Tensor) -> None:
        """Check input dimensions (to be implemented by subclasses)."""
        raise NotImplementedError
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass using InstanceNormFunction."""
        self._check_input_dim(x)
        
        weight = self.gamma if self.affine else None
        bias = self.beta if self.affine else None
        
        # Handle unbatched inputs (no batch dimension)
        # For spatial dimensions D, expected batched input shape: (N, C, *spatial)
        # Unbatched input shape: (C, *spatial)
        if x.ndim == self.num_spatial_dims + 1:
            # Add batch dimension using reshape
            x_with_batch = x.reshape((1,) + x.shape)
            output = InstanceNormFunction.apply(x_with_batch, weight, bias, self.eps)
            # Remove batch dimension
            return output.reshape(x.shape)
        else:
            # Batched input
            return InstanceNormFunction.apply(x, weight, bias, self.eps)
    
    def extra_repr(self) -> str:
        """Extra representation string."""
        return f"{self.num_features}, eps={self.eps}, affine={self.affine}"


class LayerNorm(Module):
    """Layer Normalization layer.

    Applies Layer Normalization over the last D dimensions where D is the length
    of normalized_shape. As described in the paper "Layer Normalization".

    Args:
        normalized_shape: Input shape from which to compute normalization.
            If a single integer is used, it is treated as a singleton tuple, and
            normalization is computed over the last dimension.
        eps: A value added to the denominator for numerical stability. Default: 1e-5.
        elementwise_affine: A boolean that when set to True, this module has
            learnable affine parameters (gamma and beta). Default: True.

    Shape:
        - Input: (*, S) where S is normalized_shape
        - Output: same as input

    Examples:
        >>> ln = LayerNorm(64)
        >>> x = Tensor.randn((32, 64))
        >>> output = ln(x)
        >>> ln2 = LayerNorm((10, 20))
        >>> x2 = Tensor.randn((5, 10, 20))
        >>> output2 = ln2(x2)
    """

    def __init__(
        self,
        normalized_shape: Union[int, Tuple[int, ...]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        self.gamma: Optional[Tensor] = None
        self.beta: Optional[Tensor] = None
        if self.elementwise_affine:
            self.gamma = Tensor.ones(normalized_shape, requires_grad=True)
            self.beta = Tensor.zeros(normalized_shape, requires_grad=True)
            self.register_parameter("gamma", self.gamma)
            self.register_parameter("beta", self.beta)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of LayerNorm.

        Args:
            x: Input tensor.

        Returns:
            Normalized tensor of same shape as input.
        """
        weight = self.gamma if self.elementwise_affine else None
        bias = self.beta if self.elementwise_affine else None
        return LayerNormFunction.apply(x, self.normalized_shape, weight, bias, self.eps)

    def extra_repr(self) -> str:
        """Extra representation string."""
        s = f"{self.normalized_shape}, eps={self.eps}"
        s += f", elementwise_affine={self.elementwise_affine}"
        return s

    def __repr__(self) -> str:
        """String representation of the LayerNorm layer."""
        return f"LayerNorm({self.extra_repr()})"


class GroupNorm(Module):
    """Group Normalization layer.

    Applies Group Normalization as described in the paper
    "Group Normalization" (https://arxiv.org/abs/1803.08494).

    Args:
        num_groups: Number of groups to separate the channels into.
        num_channels: Number of channels expected in input.
        eps: A value added to the denominator for numerical stability. Default: 1e-5.
        affine: A boolean that when set to True, this module has
            learnable affine parameters (gamma and beta). Default: True.

    Shape:
        - Input: (N, C, *) where * means any number of additional dimensions
        - Output: same as input

    Examples:
        >>> gn = GroupNorm(2, 6)  # 6 channels divided into 2 groups
        >>> x = Tensor.randn((4, 6, 10, 10))
        >>> output = gn(x)
    """

    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
    ) -> None:
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError(
                f"num_channels ({num_channels}) must be divisible by num_groups ({num_groups})"
            )
        
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        self.gamma: Optional[Tensor] = None
        self.beta: Optional[Tensor] = None
        if self.affine:
            self.gamma = Tensor.ones((num_channels,), requires_grad=True)
            self.beta = Tensor.zeros((num_channels,), requires_grad=True)
            self.register_parameter("gamma", self.gamma)
            self.register_parameter("beta", self.beta)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of GroupNorm.

        Args:
            x: Input tensor of shape (N, C, *).

        Returns:
            Normalized tensor of same shape as input.
        """
        weight = self.gamma if self.affine else None
        bias = self.beta if self.affine else None
        return GroupNormFunction.apply(x, self.num_groups, weight, bias, self.eps)

    def extra_repr(self) -> str:
        """Extra representation string."""
        return f"{self.num_groups}, {self.num_channels}, eps={self.eps}, affine={self.affine}"

    def __repr__(self) -> str:
        """String representation of the GroupNorm layer."""
        return f"GroupNorm({self.extra_repr()})"


class InstanceNorm2d(_InstanceNorm):
    """Instance Normalization layer for 2D inputs (3D or 4D tensors).
    
    Applies Instance Normalization as described in the paper
    "Instance Normalization: The Missing Ingredient for Fast Stylization".
    
    Normalizes each channel in each sample independently over spatial dimensions.
    Equivalent to GroupNorm with num_groups=num_channels.
    
    Args:
        num_features: Number of features/channels C in the input.
        eps: A value added to the denominator for numerical stability. Default: 1e-5.
        momentum: The value used for the running_mean and running_var computation.
            Can be set to None for cumulative moving average (i.e., simple average).
            Default: 0.1.
        affine: A boolean that when set to True, this module has learnable
            affine parameters (gamma and beta). Default: False.
        track_running_stats: A boolean that when set to True, this module tracks
            the running mean and variance; when set to False, this module does not
            track such statistics. Default: False.
    
    Shape:
        - Input: (N, C, H, W) or (C, H, W)
        - Output: same shape as input
    
    Examples:
        >>> inorm = InstanceNorm2d(64)
        >>> x = Tensor.randn((32, 64, 28, 28))
        >>> output = inorm(x)
        >>> x2 = Tensor.randn((64, 28, 28))  # unbatched input
        >>> output2 = inorm(x2)
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
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
        if x.ndim not in (3, 4):
            raise ValueError(
                f"InstanceNorm2d expects 3D or 4D input (got {x.ndim}D)"
            )
        # Channel dimension: index 1 for 4D (N, C, H, W), index 0 for 3D (C, H, W)
        channel_dim = 1 if x.ndim == 4 else 0
        if x.shape[channel_dim] != self.num_features:
            raise ValueError(
                f"Expected input channels {self.num_features}, got {x.shape[channel_dim]}"
            )


class InstanceNorm1d(_InstanceNorm):
    """Instance Normalization layer for 1D inputs (2D or 3D tensors).

    Applies Instance Normalization as described in the paper
    "Instance Normalization: The Missing Ingredient for Fast Stylization".

    Normalizes each channel in each sample independently over spatial dimensions.
    Equivalent to GroupNorm with num_groups=num_channels.

    Args:
        num_features: Number of features/channels C in the input.
        eps: A value added to the denominator for numerical stability. Default: 1e-5.
        affine: A boolean that when set to True, this module has learnable
            affine parameters (gamma and beta). Default: False.
        track_running_stats: A boolean that when set to True, this module tracks
            the running mean and variance. Default: False.

    Shape:
        - Input: (N, C, L) or (C, L)
        - Output: same shape as input

    Examples:
        >>> inorm = InstanceNorm1d(64)
        >>> x = Tensor.randn((32, 64, 10))
        >>> output = inorm(x)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = False,
        track_running_stats: bool = False,
    ) -> None:
        super().__init__(
            num_features=num_features,
            eps=eps,
            affine=affine,
            track_running_stats=track_running_stats,
            num_spatial_dims=1,
        )

    def _check_input_dim(self, x: Tensor) -> None:
        if x.ndim not in (2, 3):
            raise ValueError(
                f"InstanceNorm1d expects 2D or 3D input (got {x.ndim}D)"
            )
        # Channel dimension: index 0 for 2D (C, L), index 1 for 3D (N, C, L)
        channel_dim = 1 if x.ndim == 3 else 0
        if x.shape[channel_dim] != self.num_features:
            raise ValueError(
                f"Expected input channels {self.num_features}, got {x.shape[channel_dim]}"
            )


class InstanceNorm3d(_InstanceNorm):
    """Instance Normalization layer for 3D inputs (4D or 5D tensors).

    Applies Instance Normalization as described in the paper
    "Instance Normalization: The Missing Ingredient for Fast Stylization".

    Normalizes each channel in each sample independently over spatial dimensions.
    Equivalent to GroupNorm with num_groups=num_channels.

    Args:
        num_features: Number of features/channels C in the input.
        eps: A value added to the denominator for numerical stability. Default: 1e-5.
        affine: A boolean that when set to True, this module has learnable
            affine parameters (gamma and beta). Default: False.
        track_running_stats: A boolean that when set to True, this module tracks
            the running mean and variance. Default: False.

    Shape:
        - Input: (N, C, D, H, W) or (C, D, H, W)
        - Output: same shape as input

    Examples:
        >>> inorm = InstanceNorm3d(64)
        >>> x = Tensor.randn((8, 64, 10, 10, 10))
        >>> output = inorm(x)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = False,
        track_running_stats: bool = False,
    ) -> None:
        super().__init__(
            num_features=num_features,
            eps=eps,
            affine=affine,
            track_running_stats=track_running_stats,
            num_spatial_dims=3,
        )

    def _check_input_dim(self, x: Tensor) -> None:
        if x.ndim not in (4, 5):
            raise ValueError(
                f"InstanceNorm3d expects 4D or 5D input (got {x.ndim}D)"
            )
        # Channel dimension: index 0 for 4D (C, D, H, W), index 1 for 5D (N, C, D, H, W)
        channel_dim = 1 if x.ndim == 5 else 0
        if x.shape[channel_dim] != self.num_features:
            raise ValueError(
                f"Expected input channels {self.num_features}, got {x.shape[channel_dim]}"
            )


def layer_norm(
    input: Tensor,
    normalized_shape: Union[int, Tuple[int, ...]],
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    """Functional interface for layer normalization.

    Args:
        input: Input tensor.
        normalized_shape: Input shape from which to compute normalization.
        weight: The weight tensor (gamma) of shape normalized_shape.
        bias: The bias tensor (beta) of shape normalized_shape.
        eps: A value added to the denominator for numerical stability.

    Returns:
        Normalized tensor of same shape as input.
    """
    return LayerNormFunction.apply(input, normalized_shape, weight, bias, eps)


def group_norm(
    input: Tensor,
    num_groups: int,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    """Functional interface for group normalization.

    Args:
        input: Input tensor of shape (N, C, *).
        num_groups: Number of groups to separate the channels into.
        weight: The weight tensor (gamma) of shape (C,).
        bias: The bias tensor (beta) of shape (C,).
        eps: A value added to the denominator for numerical stability.

    Returns:
        Normalized tensor of same shape as input.
    """
    return GroupNormFunction.apply(input, num_groups, weight, bias, eps)


def instance_norm(
    input: Tensor,
    weight: Optional[Tensor] = None,
    bias: Optional[Tensor] = None,
    eps: float = 1e-5,
) -> Tensor:
    """Functional interface for instance normalization.

    Applies Instance Normalization independently for each channel in each sample.
    For input shape (N, C, *), computes statistics over spatial dimensions
    for each channel in each sample independently.

    Args:
        input: Input tensor of shape (N, C, *).
        weight: Optional weight tensor (gamma) of shape (C,).
        bias: Optional bias tensor (beta) of shape (C,).
        eps: A value added to the denominator for numerical stability.

    Returns:
        Normalized tensor of same shape as input.
    """
    return InstanceNormFunction.apply(input, weight, bias, eps)