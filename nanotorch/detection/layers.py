"""
YOLO-specific neural network layers.

This module provides building blocks for YOLO v12 and similar object detectors:
- Conv: Standard convolution with BatchNorm and SiLU activation
- DWConv: Depth-wise separable convolution
- Bottleneck: Residual bottleneck block
- C2f: Cross Stage Partial Network with 2 flow paths (YOLO v8 style)
- SPPF: Spatial Pyramid Pooling - Fast
- Concat: Concatenate module for skip connections
- Area Attention: Efficient attention for large feature maps (YOLO v12)
"""

import numpy as np
from typing import Optional, List, Tuple
from nanotorch.tensor import Tensor
from nanotorch.nn.module import Module, Sequential
from nanotorch.nn.conv import Conv2D
from nanotorch.nn.batchnorm import BatchNorm2d
from nanotorch.nn.activation import SiLU, Identity
from nanotorch.nn.pooling import MaxPool2d
from nanotorch.nn.normalization import LayerNorm


class Conv(Module):
    """Standard convolution block with BatchNorm and SiLU activation.
    
    This is the fundamental building block used throughout YOLO models.
    
    Structure: Conv2D -> BatchNorm2d -> SiLU
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size (default: 3)
        stride: Stride (default: 1)
        padding: Padding (default: None, auto-computed)
        groups: Groups for depthwise conv (default: 1)
        activation: Whether to use activation (default: True)
    
    Shape:
        - Input: (N, C_in, H, W)
        - Output: (N, C_out, H_out, W_out)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        activation: bool = True
    ) -> None:
        super().__init__()
        
        # Auto-compute padding for 'same' style when stride=1
        if padding is None:
            padding = kernel_size // 2
        
        self.conv = Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False  # No bias because BatchNorm follows
        )
        
        self.bn = BatchNorm2d(num_features=out_channels)
        
        if activation:
            self.act = SiLU()
        else:
            self.act = Identity()
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class DWConv(Module):
    """Depth-wise separable convolution.
    
    Reduces computation by using depth-wise convolution followed by
    point-wise (1x1) convolution.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size for depth-wise conv (default: 3)
        stride: Stride (default: 1)
    
    Shape:
        - Input: (N, C_in, H, W)
        - Output: (N, C_out, H_out, W_out)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1
    ) -> None:
        super().__init__()
        
        # Depth-wise convolution (groups = in_channels)
        self.dw_conv = Conv(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
            activation=False
        )
        
        # Point-wise convolution (1x1)
        self.pw_conv = Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            activation=True
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class Bottleneck(Module):
    """Residual bottleneck block.
    
    Structure:
    - Conv 1x1 -> Conv 3x3 -> Add input (shortcut)
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        shortcut: Whether to use residual connection (default: True)
        expansion: Channel expansion ratio (default: 0.5)
    
    Shape:
        - Input: (N, C_in, H, W)
        - Output: (N, C_out, H, W)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        shortcut: bool = True,
        expansion: float = 0.5
    ) -> None:
        super().__init__()
        
        hidden_channels = int(out_channels * expansion)
        
        self.cv1 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.cv2 = Conv(hidden_channels, out_channels, kernel_size=3, stride=1)
        
        self.shortcut = shortcut and (in_channels == out_channels)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with optional residual connection."""
        identity = x
        out = self.cv1(x)
        out = self.cv2(out)
        
        if self.shortcut:
            out = out + identity
        
        return out


class C2f(Module):
    """CSP Bottleneck with 2 flow paths (YOLO v8 style).
    
    This module implements the Cross Stage Partial Network (CSP) concept
    with 2 flow paths for efficient gradient flow.
    
    Structure:
    - Split input into 2 parts
    - Pass one part through n Bottlenecks
    - Concatenate all intermediate outputs
    - Final 1x1 conv to merge
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        num_bottlenecks: Number of Bottleneck blocks (default: 1)
        shortcut: Use shortcut in bottlenecks (default: True)
        expansion: Channel expansion ratio (default: 0.5)
    
    Shape:
        - Input: (N, C_in, H, W)
        - Output: (N, C_out, H, W)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_bottlenecks: int = 1,
        shortcut: bool = True,
        expansion: float = 0.5
    ) -> None:
        super().__init__()
        
        hidden_channels = int(out_channels * expansion)
        
        # Initial 1x1 conv to reduce channels and split
        self.cv1 = Conv(in_channels, 2 * hidden_channels, kernel_size=1, stride=1)
        
        # Bottleneck blocks
        self.num_bottlenecks = num_bottlenecks
        self.bottlenecks = [
            Bottleneck(hidden_channels, hidden_channels, shortcut=shortcut, expansion=0.5)
            for _ in range(num_bottlenecks)
        ]
        
        # Final 1x1 conv to merge all features
        # Input: cv1 split (hidden) + all bottleneck outputs (n * hidden)
        self.cv2 = Conv((2 + num_bottlenecks) * hidden_channels, out_channels, kernel_size=1, stride=1)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with CSP structure."""
        # Initial conv and split
        y = self.cv1(x)
        
        # Split into 2 parts along channel dimension
        split_size = y.shape[1] // 2
        y1 = y[:, :split_size, :, :]  # First half
        y2 = y[:, split_size:, :, :]  # Second half
        
        # Collect features
        outputs = [y1, y2]
        
        # Pass through bottlenecks
        current = y2
        for bottleneck in self.bottlenecks:
            current = bottleneck(current)
            outputs.append(current)
        
        # Concatenate all outputs along channel dimension
        y = Tensor.cat(outputs, dim=1)
        
        # Final conv to merge
        out = self.cv2(y)
        
        return out


class SPPF(Module):
    """Spatial Pyramid Pooling - Fast (SPPF).
    
    Equivalent to SPP(k=5, 9, 13) but using only k=5 with 3 sequential
    MaxPool operations for speed.
    
    Structure:
    - Input -> Conv -> MaxPool(k=5) -> MaxPool(k=5) -> MaxPool(k=5)
    - Concatenate input + 3 pooled outputs
    - Final Conv
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Pooling kernel size (default: 5)
    
    Shape:
        - Input: (N, C_in, H, W)
        - Output: (N, C_out, H, W) (same spatial size due to padding)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5
    ) -> None:
        super().__init__()
        
        hidden_channels = in_channels // 2
        
        # Initial conv
        self.cv1 = Conv(in_channels, hidden_channels, kernel_size=1, stride=1)
        
        # Sequential MaxPool blocks
        # Note: MaxPool2d with kernel=5, stride=1, padding=2 maintains spatial size
        self.pool1 = MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.pool2 = MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.pool3 = MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        
        # Final conv to merge (input + 3 pooled = 4x channels)
        self.cv2 = Conv(4 * hidden_channels, out_channels, kernel_size=1, stride=1)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        x = self.cv1(x)
        
        y1 = self.pool1(x)
        y2 = self.pool2(y1)
        y3 = self.pool3(y2)
        
        # Concatenate along channel dimension
        y = Tensor.cat([x, y1, y2, y3], dim=1)
        
        out = self.cv2(y)
        return out


class Focus(Module):
    """Focus layer for spatial-to-channel transformation.
    
    Slices input into 4 parts and concatenates along channel dimension.
    Equivalent to a stride-2 convolution but with less computation.
    
    Input: (N, C, H, W)
    Output: (N, 4*C, H/2, W/2)
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Kernel size for follow-up conv (default: 1)
    
    Note: This is from older YOLO versions (v5, v6). 
          YOLO v8+ uses simple stride-2 conv instead.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1
    ) -> None:
        super().__init__()
        
        self.conv = Conv(in_channels * 4, out_channels, kernel_size=kernel_size, stride=1)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with space-to-depth."""
        # Slice input into 4 parts
        # x: (N, C, H, W) -> (N, 4*C, H/2, W/2)
        N, C, H, W = x.shape
        
        # Get slices at different positions
        x1 = x[:, :, 0::2, 0::2]  # Top-left
        x2 = x[:, :, 1::2, 0::2]  # Bottom-left
        x3 = x[:, :, 0::2, 1::2]  # Top-right
        x4 = x[:, :, 1::2, 1::2]  # Bottom-right
        
        # Concatenate along channel dimension
        y = Tensor.cat([x1, x2, x3, x4], dim=1)
        
        return self.conv(y)


class Concat(Module):
    """Concatenate module for skip connections.
    
    Concatenates a list of tensors along the specified dimension.
    
    Args:
        dim: Dimension to concatenate along (default: 1 for channels)
    """
    
    def __init__(self, dim: int = 1) -> None:
        super().__init__()
        self.dim = dim
    
    def forward(self, inputs: List[Tensor]) -> Tensor:
        """Concatenate input tensors."""
        return Tensor.cat(inputs, dim=self.dim)


class Upsample(Module):
    """Upsample layer using nearest neighbor interpolation.
    
    Args:
        scale_factor: Scale factor for upsampling
        mode: Interpolation mode (only 'nearest' supported)
    """
    
    def __init__(self, scale_factor: int = 2, mode: str = "nearest") -> None:
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
    
    def forward(self, x: Tensor) -> Tensor:
        """Upsample input tensor."""
        if self.mode != "nearest":
            raise ValueError(f"Only 'nearest' mode supported, got {self.mode}")
        
        N, C, H, W = x.shape
        
        # Repeat elements to upscale
        x = x.repeat(1, 1, self.scale_factor, self.scale_factor)
        
        return x


class AreaAttention(Module):
    """Area Attention module for YOLO v12.
    
    Instead of attending to the entire feature map, Area Attention divides
    the feature map into regions and performs attention within each region.
    This reduces computational complexity from O(N^2) to O(N * area).
    
    Reference: YOLO v12 paper
    
    Args:
        in_channels: Number of input channels
        num_heads: Number of attention heads (default: 8)
        area: Area for region division (default: 4, meaning 2x2 grid)
        dropout: Dropout rate (default: 0.0)
    
    Shape:
        - Input: (N, C, H, W)
        - Output: (N, C, H, W)
    """
    
    def __init__(
        self,
        in_channels: int,
        num_heads: int = 8,
        area: int = 4,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.area = area
        self.head_dim = in_channels // num_heads
        
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        
        # Q, K, V projections
        self.q_proj = Conv(in_channels, in_channels, kernel_size=1, activation=False)
        self.k_proj = Conv(in_channels, in_channels, kernel_size=1, activation=False)
        self.v_proj = Conv(in_channels, in_channels, kernel_size=1, activation=False)
        
        # Output projection
        self.out_proj = Conv(in_channels, in_channels, kernel_size=1, activation=False)
        
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout
    
    def forward(self, x: Tensor) -> Tensor:
        """Area attention forward pass."""
        N, C, H, W = x.shape
        
        # Compute Q, K, V
        q = self.q_proj(x)  # (N, C, H, W)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for area-based attention
        # Divide into (area_h, area_w) grid
        area_h = int(np.sqrt(self.area))
        area_w = self.area // area_h
        
        assert H % area_h == 0 and W % area_w == 0, \
            f"Spatial dims ({H}, {W}) must be divisible by area grid ({area_h}, {area_w})"
        
        region_h = H // area_h
        region_w = W // area_w
        
        # Reshape: (N, C, H, W) -> (N * area_h * area_w, C, region_h, region_w)
        # First split into regions
        q_regions = q.reshape(N, C, area_h, region_h, area_w, region_w)
        q_regions = q_regions.transpose(0, 2, 4, 1, 3, 5)  # (N, area_h, area_w, C, region_h, region_w)
        q_regions = q_regions.reshape(N * self.area, C, region_h * region_w)
        
        k_regions = k.reshape(N, C, area_h, region_h, area_w, region_w)
        k_regions = k_regions.transpose(0, 2, 4, 1, 3, 5)
        k_regions = k_regions.reshape(N * self.area, C, region_h * region_w)
        
        v_regions = v.reshape(N, C, area_h, region_h, area_w, region_w)
        v_regions = v_regions.transpose(0, 2, 4, 1, 3, 5)
        v_regions = v_regions.reshape(N * self.area, C, region_h * region_w)
        
        # Reshape for multi-head attention
        # (N * area, C, L) -> (N * area, num_heads, L, head_dim)
        L = region_h * region_w
        q_mh = q_regions.reshape(N * self.area, self.num_heads, self.head_dim, L).transpose(0, 1, 3, 2)
        k_mh = k_regions.reshape(N * self.area, self.num_heads, self.head_dim, L).transpose(0, 1, 3, 2)
        v_mh = v_regions.reshape(N * self.area, self.num_heads, self.head_dim, L).transpose(0, 1, 3, 2)
        
        # Compute attention scores
        # (N * area, num_heads, L, L)
        attn = q_mh.matmul(k_mh.transpose(0, 1, 3, 2)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Apply attention to values
        out = attn.matmul(v_mh)  # (N * area, num_heads, L, head_dim)
        
        # Reshape back
        out = out.transpose(0, 1, 3, 2).reshape(N * self.area, C, L)
        out = out.reshape(N, area_h, area_w, C, region_h, region_w)
        out = out.transpose(0, 3, 1, 4, 2, 5).reshape(N, C, H, W)
        
        # Output projection
        out = self.out_proj(out)
        
        return out


class A2Block(Module):
    """A2 (Area Attention) Block for YOLO v12.
    
    Combines Area Attention with FFN for transformer-style processing.
    
    Args:
        in_channels: Number of input channels
        num_heads: Number of attention heads
        area: Area for region division
        mlp_ratio: MLP expansion ratio (default: 4.0)
        dropout: Dropout rate (default: 0.0)
    """
    
    def __init__(
        self,
        in_channels: int,
        num_heads: int = 8,
        area: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0
    ) -> None:
        super().__init__()
        
        self.norm1 = LayerNorm(normalized_shape=in_channels)
        self.attn = AreaAttention(in_channels, num_heads, area, dropout)
        self.norm2 = LayerNorm(normalized_shape=in_channels)
        
        hidden_channels = int(in_channels * mlp_ratio)
        self.mlp = Sequential([
            Conv(in_channels, hidden_channels, kernel_size=1, activation=True),
            Conv(hidden_channels, in_channels, kernel_size=1, activation=False)
        ])
    
    def forward(self, x: Tensor) -> Tensor:
        """A2 block forward pass with residual connections."""
        # Reshape for LayerNorm: (N, C, H, W) -> (N, H, W, C)
        N, C, H, W = x.shape
        
        x_norm = x.transpose(0, 2, 3, 1)  # (N, H, W, C)
        x_norm = self.norm1(x_norm)
        x_norm = x_norm.transpose(0, 3, 1, 2)  # (N, C, H, W)
        
        # Attention with residual
        x = x + self.attn(x_norm)
        
        # FFN with residual
        x_norm = x.transpose(0, 2, 3, 1)
        x_norm = self.norm2(x_norm)
        x_norm = x_norm.transpose(0, 3, 1, 2)
        
        x = x + self.mlp(x_norm)
        
        return x


class DetectLayer(Module):
    """Detection output layer for anchor-free detection.
    
    This is a simplified detection head that produces raw predictions.
    Full detection head is in yolo_head.py.
    
    Args:
        in_channels: Number of input channels
        num_classes: Number of object classes
        num_anchors: Number of anchors per location (default: 1 for anchor-free)
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_anchors: int = 1
    ) -> None:
        super().__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Output: 4 (bbox) + num_classes
        self.output_channels = 4 + num_classes
        
        # Prediction conv (no BN, no activation for raw predictions)
        self.conv = Conv2D(
            in_channels=in_channels,
            out_channels=num_anchors * self.output_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True
        )
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass producing raw predictions."""
        return self.conv(x)


# Helper functions for building YOLO models

def make_divisible(x: int, divisor: int = 8) -> int:
    """Make channel count divisible by divisor.
    
    This ensures efficient tensor operations on hardware accelerators.
    """
    return int((x + divisor / 2) // divisor * divisor)


def parse_model(d: dict, ch: List[int]) -> Tuple[Module, List[int]]:
    """Parse model configuration and build model.
    
    Args:
        d: Model configuration dictionary
        ch: List of input channels for each layer
    
    Returns:
        Tuple of (model, save_list)
    """
    # This is a simplified version - full implementation would parse YAML config
    raise NotImplementedError("Use specific model builders instead")
