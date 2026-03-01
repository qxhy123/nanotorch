"""
YOLO v12 Backbone Network.

This module implements the backbone architecture for YOLO v12:
- R-ELAN (Residual Efficient Layer Aggregation Network)
- A2 (Area Attention) enhanced backbone

The backbone extracts multi-scale features from input images at strides
8, 16, and 32 (P3, P4, P5 outputs).

Key components:
- Stem: Initial convolution block
- Stage: Conv blocks with downsampling
- R-ELAN Block: Residual ELAN for feature extraction
- A2 Block: Area attention for enhanced feature representation
"""

from typing import Dict
from nanotorch.tensor import Tensor
from nanotorch.nn.module import Module, Sequential
from nanotorch.detection.layers import Conv, C2f, SPPF, AreaAttention, make_divisible


class Stem(Module):
    """Stem block for initial feature extraction.
    
    Uses two 3x3 convolutions with stride 2 for 4x downsampling.
    
    Args:
        in_channels: Number of input channels (usually 3 for RGB)
        out_channels: Number of output channels
    
    Shape:
        - Input: (N, 3, H, W)
        - Output: (N, out_channels, H/4, W/4)
    """
    
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        
        self.conv1 = Conv(in_channels, out_channels // 2, kernel_size=3, stride=2)
        self.conv2 = Conv(out_channels // 2, out_channels, kernel_size=3, stride=2)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with 4x downsampling."""
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Downsample(Module):
    """Downsampling block using strided convolution.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
    """
    
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        
        self.conv = Conv(in_channels, out_channels, kernel_size=3, stride=2)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with 2x downsampling."""
        return self.conv(x)


class RELANBlock(Module):
    """Residual Efficient Layer Aggregation Network (R-ELAN) block.
    
    This is the key innovation in YOLO v12 backbone. It combines:
    1. ELAN structure for efficient feature aggregation
    2. Block-level residual connections with learnable scaling
    
    Structure:
    - Split input
    - Multiple Conv-BN-SiLU branches
    - Concatenate all branches
    - Final 1x1 conv for channel reduction
    - Residual connection with scaling factor
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        num_blocks: Number of conv blocks in aggregation (default: 2)
        expansion: Channel expansion ratio (default: 0.5)
        residual_scale: Residual connection scaling factor (default: 0.1)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 2,
        expansion: float = 0.5,
        residual_scale: float = 0.1
    ) -> None:
        super().__init__()
        
        hidden_channels = int(out_channels * expansion)
        
        # Initial conv for channel adjustment
        self.cv1 = Conv(in_channels, hidden_channels * 2, kernel_size=1)
        
        # Conv blocks for feature aggregation
        self.blocks = [
            Conv(hidden_channels, hidden_channels, kernel_size=3, stride=1)
            for _ in range(num_blocks)
        ]
        
        # Final conv to merge all features
        self.cv2 = Conv(
            hidden_channels * (2 + num_blocks),
            out_channels,
            kernel_size=1
        )
        
        # Residual connection (only if channels match)
        self.residual = (in_channels == out_channels)
        self.residual_scale = residual_scale
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with residual connection."""
        identity = x
        
        # Initial conv and split
        y = self.cv1(x)
        split_size = y.shape[1] // 2
        y1 = y[:, :split_size, :, :]
        y2 = y[:, split_size:, :, :]
        
        # Collect features
        outputs = [y1, y2]
        
        # Pass through conv blocks
        current = y2
        for block in self.blocks:
            current = block(current)
            outputs.append(current)
        
        # Concatenate and merge
        y = Tensor.cat(outputs, dim=1)
        y = self.cv2(y)
        
        # Residual connection with scaling
        if self.residual:
            y = y + identity * self.residual_scale
        
        return y


class A2RELANBlock(Module):
    """R-ELAN block with Area Attention enhancement.
    
    This combines R-ELAN's efficient feature aggregation with
    Area Attention for improved global context modeling.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        num_blocks: Number of conv blocks
        num_heads: Number of attention heads
        area: Area for region division
        expansion: Channel expansion ratio
        residual_scale: Residual connection scaling factor
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 2,
        num_heads: int = 8,
        area: int = 4,
        expansion: float = 0.5,
        residual_scale: float = 0.1
    ) -> None:
        super().__init__()
        
        hidden_channels = int(out_channels * expansion)
        
        # Initial conv
        self.cv1 = Conv(in_channels, hidden_channels * 2, kernel_size=1)
        
        # Conv blocks
        self.blocks = [
            Conv(hidden_channels, hidden_channels, kernel_size=3)
            for _ in range(num_blocks)
        ]
        
        # Area attention (optional enhancement)
        self.use_attention = hidden_channels % num_heads == 0
        if self.use_attention:
            self.attn = AreaAttention(
                hidden_channels * (2 + num_blocks),
                num_heads=num_heads,
                area=area
            )
        
        # Final conv
        self.cv2 = Conv(
            hidden_channels * (2 + num_blocks),
            out_channels,
            kernel_size=1
        )
        
        self.residual = (in_channels == out_channels)
        self.residual_scale = residual_scale
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass with attention and residual."""
        identity = x
        
        # Initial conv and split
        y = self.cv1(x)
        split_size = y.shape[1] // 2
        y1 = y[:, :split_size, :, :]
        y2 = y[:, split_size:, :, :]
        
        # Collect features
        outputs = [y1, y2]
        
        # Conv blocks
        current = y2
        for block in self.blocks:
            current = block(current)
            outputs.append(current)
        
        # Concatenate
        y = Tensor.cat(outputs, dim=1)
        
        # Optional attention
        if self.use_attention:
            y = y + self.attn(y)
        
        # Final conv
        y = self.cv2(y)
        
        # Residual
        if self.residual:
            y = y + identity * self.residual_scale
        
        return y


class YOLOBackbone(Module):
    """YOLO v12 Backbone Network.
    
    Multi-scale feature extraction backbone that outputs features
    at three different scales (P3, P4, P5).
    
    Architecture:
    - Stem: Initial 4x downsampling
    - Stage 1-4: Progressive feature extraction with 2x downsampling
    - SPPF: Spatial pyramid pooling for global context
    - A2 Blocks: Area attention for enhanced features
    
    Outputs are at strides 8, 16, 32 relative to input image.
    
    Args:
        in_channels: Number of input channels (default: 3)
        width_mult: Width multiplier for channel scaling (default: 1.0)
        depth_mult: Depth multiplier for layer repetition (default: 1.0)
        use_attention: Whether to use Area Attention (default: True)
    
    Shape:
        - Input: (N, 3, H, W)
        - Output: Dict with 'p3', 'p4', 'p5' keys
            - p3: (N, C3, H/8, W/8)
            - p4: (N, C4, H/16, W/16)
            - p5: (N, C5, H/32, W/32)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        width_mult: float = 1.0,
        depth_mult: float = 1.0,
        use_attention: bool = True
    ) -> None:
        super().__init__()
        
        # Base channels for different model sizes
        # YOLO v12-n: width_mult=0.25, depth_mult=0.34
        # YOLO v12-s: width_mult=0.50, depth_mult=0.67
        # YOLO v12-m: width_mult=0.75, depth_mult=0.85
        # YOLO v12-l: width_mult=1.00, depth_mult=1.00
        # YOLO v12-x: width_mult=1.25, depth_mult=1.20
        
        base_channels = [64, 128, 256, 512, 1024]
        base_depths = [3, 6, 6, 3]
        
        # Scale channels and depths
        channels = [make_divisible(int(c * width_mult), 8) for c in base_channels]
        depths = [max(round(d * depth_mult), 1) for d in base_depths]
        
        self.use_attention = use_attention
        
        # Stem: 4x downsampling
        self.stem = Stem(in_channels, channels[0])
        
        # Stage 1: stride 4 -> 8 (P3 output)
        self.stage1 = Sequential(
            Downsample(channels[0], channels[1]),
            self._make_elan_stage(channels[1], channels[1], depths[0], use_attention)
        )
        
        # Stage 2: stride 8 -> 16 (P4 output)
        self.stage2 = Sequential(
            Downsample(channels[1], channels[2]),
            self._make_elan_stage(channels[2], channels[2], depths[1], use_attention)
        )
        
        # Stage 3: stride 16 -> 32 (P5 output)
        self.stage3 = Sequential(
            Downsample(channels[2], channels[3]),
            self._make_elan_stage(channels[3], channels[3], depths[2], use_attention)
        )
        
        # Stage 4: Additional feature refinement at P5
        self.stage4 = Sequential(
            self._make_elan_stage(channels[3], channels[4], depths[3], use_attention)
        )
        
        # SPPF for global context at P5
        self.sppf = SPPF(channels[4], channels[4], kernel_size=5)
        
        # Store output channels for neck
        self.out_channels = {
            'p3': channels[1],   # stride 8
            'p4': channels[2],   # stride 16
            'p5': channels[4]    # stride 32
        }
        
        self.strides = {'p3': 8, 'p4': 16, 'p5': 32}
    
    def _make_elan_stage(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        use_attention: bool
    ) -> Module:
        """Create an ELAN stage with optional attention."""
        if use_attention and in_channels >= 64:
            # Use A2-enhanced R-ELAN for larger feature maps
            return A2RELANBlock(
                in_channels, out_channels,
                num_blocks=num_blocks,
                num_heads=max(1, out_channels // 64),
                area=4
            )
        else:
            return RELANBlock(in_channels, out_channels, num_blocks=num_blocks)
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Forward pass returning multi-scale features."""
        # Stem
        x = self.stem(x)  # stride 4
        
        # Stage 1 -> P3 (stride 8)
        p3 = self.stage1(x)
        
        # Stage 2 -> P4 (stride 16)
        p4 = self.stage2(p3)
        
        # Stage 3 -> P5 (stride 32)
        p5 = self.stage3(p4)
        
        # Stage 4: refine P5
        p5 = self.stage4(p5)
        
        # SPPF for global context
        p5 = self.sppf(p5)
        
        return {
            'p3': p3,  # (N, C3, H/8, W/8)
            'p4': p4,  # (N, C4, H/16, W/16)
            'p5': p5   # (N, C5, H/32, W/32)
        }


class YOLOBackboneTiny(Module):
    """Lightweight YOLO Backbone for mobile/embedded deployment.
    
    A simplified backbone without attention, optimized for speed.
    
    Args:
        in_channels: Number of input channels (default: 3)
        width_mult: Width multiplier (default: 0.5)
        depth_mult: Depth multiplier (default: 0.34)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        width_mult: float = 0.5,
        depth_mult: float = 0.34
    ) -> None:
        super().__init__()
        
        base_channels = [32, 64, 128, 256, 512]
        channels = [make_divisible(int(c * width_mult), 8) for c in base_channels]
        depths = [max(round(3 * depth_mult), 1), max(round(6 * depth_mult), 1)]
        
        # Stem
        self.stem = Conv(in_channels, channels[0], kernel_size=3, stride=2)
        
        # Stage 1
        self.stage1 = Sequential(
            Conv(channels[0], channels[1], kernel_size=3, stride=2),
            C2f(channels[1], channels[1], num_bottlenecks=depths[0])
        )
        
        # Stage 2
        self.stage2 = Sequential(
            Conv(channels[1], channels[2], kernel_size=3, stride=2),
            C2f(channels[2], channels[2], num_bottlenecks=depths[1])
        )
        
        # Stage 3
        self.stage3 = Sequential(
            Conv(channels[2], channels[3], kernel_size=3, stride=2),
            C2f(channels[3], channels[3], num_bottlenecks=depths[0])
        )
        
        # SPPF
        self.sppf = SPPF(channels[3], channels[4], kernel_size=5)
        
        self.out_channels = {
            'p3': channels[1],
            'p4': channels[2],
            'p5': channels[4]
        }
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Forward pass."""
        x = self.stem(x)
        p3 = self.stage1(x)
        p4 = self.stage2(p3)
        p5 = self.stage3(p4)
        p5 = self.sppf(p5)
        
        return {'p3': p3, 'p4': p4, 'p5': p5}


def build_backbone(
    model_size: str = 's',
    in_channels: int = 3,
    use_attention: bool = True
) -> YOLOBackbone:
    """Build YOLO backbone by model size.
    
    Args:
        model_size: Model size ('n', 's', 'm', 'l', 'x')
        in_channels: Number of input channels
        use_attention: Whether to use Area Attention
    
    Returns:
        Configured YOLOBackbone instance
    """
    size_configs = {
        'n': {'width_mult': 0.25, 'depth_mult': 0.34},
        's': {'width_mult': 0.50, 'depth_mult': 0.67},
        'm': {'width_mult': 0.75, 'depth_mult': 0.85},
        'l': {'width_mult': 1.00, 'depth_mult': 1.00},
        'x': {'width_mult': 1.25, 'depth_mult': 1.20},
    }
    
    if model_size not in size_configs:
        raise ValueError(f"Unknown model size: {model_size}. "
                        f"Choose from {list(size_configs.keys())}")
    
    config = size_configs[model_size]
    
    return YOLOBackbone(
        in_channels=in_channels,
        width_mult=config['width_mult'],
        depth_mult=config['depth_mult'],
        use_attention=use_attention
    )
