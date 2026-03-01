"""
YOLO v12 Neck Network.

This module implements the neck (feature fusion) architecture for YOLO v12:
- PANet (Path Aggregation Network)
- FPN (Feature Pyramid Network)
- BiFPN-style fusion (optional)

The neck fuses multi-scale features from the backbone to enhance
feature representation for detection at different scales.

Key concepts:
- Top-down pathway: Enriches semantic information in lower-resolution features
- Bottom-up pathway: Propagates localization information to higher-resolution features
"""

import numpy as np
from typing import Dict, List, Optional
from nanotorch.tensor import Tensor
from nanotorch.nn.module import Module
from nanotorch.detection.layers import Conv, C2f, Upsample, Concat


class FPN(Module):
    """Feature Pyramid Network.
    
    Top-down feature fusion that progressively upsamples and merges
    higher-level features with lower-level features.
    
    Structure:
    P5 -> Upsample -> Concat(P4) -> Conv -> P4_out
    P4_out -> Upsample -> Concat(P3) -> Conv -> P3_out
    
    Args:
        in_channels: Dict of input channels {'p3': c3, 'p4': c4, 'p5': c5}
        out_channels: Dict of output channels (same structure)
    
    Shape:
        - Input: Dict with 'p3', 'p4', 'p5' feature maps
        - Output: Dict with enhanced 'p3', 'p4', 'p5' feature maps
    """
    
    def __init__(
        self,
        in_channels: Dict[str, int],
        out_channels: Optional[Dict[str, int]] = None
    ) -> None:
        super().__init__()
        
        if out_channels is None:
            out_channels = in_channels
        
        # Lateral convolutions (1x1 to reduce channels)
        self.lateral_p5 = Conv(in_channels['p5'], out_channels['p4'], kernel_size=1)
        self.lateral_p4 = Conv(in_channels['p4'], out_channels['p3'], kernel_size=1)
        
        # Output convolutions (3x3 to smooth fused features)
        self.output_p4 = Conv(out_channels['p4'], out_channels['p4'], kernel_size=3)
        self.output_p3 = Conv(out_channels['p3'], out_channels['p3'], kernel_size=3)
        
        # Upsampling
        self.upsample = Upsample(scale_factor=2, mode='nearest')
        
        self.out_channels = out_channels
    
    def forward(self, features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """FPN forward pass with top-down fusion."""
        p3, p4, p5 = features['p3'], features['p4'], features['p5']
        
        # Top-down pathway
        # P5 -> P4
        p5_lateral = self.lateral_p5(p5)
        p5_up = self.upsample(p5_lateral)
        p4_fused = Tensor.cat([p4, p5_up], dim=1)
        p4_out = self.output_p4(p4_fused)
        
        # P4 -> P3
        p4_lateral = self.lateral_p4(p4_out)
        p4_up = self.upsample(p4_lateral)
        p3_fused = Tensor.cat([p3, p4_up], dim=1)
        p3_out = self.output_p3(p3_fused)
        
        return {
            'p3': p3_out,
            'p4': p4_out,
            'p5': p5  # P5 unchanged in basic FPN
        }


class PANet(Module):
    """Path Aggregation Network (PANet).
    
    Combines top-down and bottom-up pathways for better feature fusion.
    This is the standard neck used in YOLO models.
    
    Structure:
    1. Top-down pathway (FPN):
       P5 -> Upsample -> Concat(P4) -> C2f -> P4'
       P4' -> Upsample -> Concat(P3) -> C2f -> P3'
    
    2. Bottom-up pathway:
       P3' -> Downsample -> Concat(P4') -> C2f -> P4''
       P4'' -> Downsample -> Concat(P5) -> C2f -> P5''
    
    Args:
        in_channels: Dict of input channels from backbone
        out_channels: Output channels for each scale (default: same as in_channels)
        num_blocks: Number of C2f blocks at each stage (default: 3)
    
    Shape:
        - Input: Dict with 'p3', 'p4', 'p5' features from backbone
        - Output: Dict with fused 'p3', 'p4', 'p5' features
    """
    
    def __init__(
        self,
        in_channels: Dict[str, int],
        out_channels: Optional[Dict[str, int]] = None,
        num_blocks: int = 3
    ) -> None:
        super().__init__()
        
        if out_channels is None:
            out_channels = in_channels
        
        c3_in, c4_in, c5_in = in_channels['p3'], in_channels['p4'], in_channels['p5']
        c3_out, c4_out, c5_out = out_channels['p3'], out_channels['p4'], out_channels['p5']
        
        # Top-down pathway (FPN)
        # P5 -> reduce channels -> upsample
        self.reduce_p5 = Conv(c5_in, c4_out, kernel_size=1)
        self.upsample_p5 = Upsample(scale_factor=2, mode='nearest')
        
        # P4 fusion
        self.p4_fusion = C2f(c4_in + c4_out, c4_out, num_bottlenecks=num_blocks)
        
        # P4 -> reduce channels -> upsample
        self.reduce_p4 = Conv(c4_out, c3_out, kernel_size=1)
        self.upsample_p4 = Upsample(scale_factor=2, mode='nearest')
        
        # P3 fusion
        self.p3_fusion = C2f(c3_in + c3_out, c3_out, num_bottlenecks=num_blocks)
        
        # Bottom-up pathway
        # P3 -> downsample
        self.downsample_p3 = Conv(c3_out, c4_out, kernel_size=3, stride=2)
        
        # P4 fusion (second pass)
        self.p4_fusion2 = C2f(c4_out + c4_out, c4_out, num_bottlenecks=num_blocks)
        
        # P4 -> downsample
        self.downsample_p4 = Conv(c4_out, c5_out, kernel_size=3, stride=2)
        
        # P5 fusion (second pass)
        self.p5_fusion = C2f(c5_in + c5_out, c5_out, num_bottlenecks=num_blocks)
        
        self.out_channels = out_channels
    
    def forward(self, features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """PANet forward pass with bidirectional fusion."""
        p3, p4, p5 = features['p3'], features['p4'], features['p5']
        
        # ===== Top-down pathway (FPN) =====
        # P5 -> P4
        p5_reduced = self.reduce_p5(p5)
        p5_up = self.upsample_p5(p5_reduced)
        p4_concat = Tensor.cat([p4, p5_up], dim=1)
        p4_fpn = self.p4_fusion(p4_concat)
        
        # P4 -> P3
        p4_reduced = self.reduce_p4(p4_fpn)
        p4_up = self.upsample_p4(p4_reduced)
        p3_concat = Tensor.cat([p3, p4_up], dim=1)
        p3_out = self.p3_fusion(p3_concat)
        
        # ===== Bottom-up pathway =====
        # P3 -> P4
        p3_down = self.downsample_p3(p3_out)
        p4_concat2 = Tensor.cat([p4_fpn, p3_down], dim=1)
        p4_out = self.p4_fusion2(p4_concat2)
        
        # P4 -> P5
        p4_down = self.downsample_p4(p4_out)
        p5_concat = Tensor.cat([p5, p4_down], dim=1)
        p5_out = self.p5_fusion(p5_concat)
        
        return {
            'p3': p3_out,
            'p4': p4_out,
            'p5': p5_out
        }


class BiFPN(Module):
    """Bi-directional Feature Pyramid Network.
    
    An enhanced FPN with weighted feature fusion and cross-scale connections.
    
    Key innovations:
    1. Weighted fusion: Learnable weights for each input feature
    2. Cross-scale connections: Direct paths from P3 to P5 and vice versa
    3. Repeated blocks: Multiple BiFPN layers for better fusion
    
    Reference: EfficientDet paper
    
    Args:
        in_channels: Dict of input channels
        out_channels: Output channels
        num_repeats: Number of repeated BiFPN blocks (default: 3)
        epsilon: Small constant for numerical stability (default: 1e-4)
    """
    
    def __init__(
        self,
        in_channels: Dict[str, int],
        out_channels: Optional[Dict[str, int]] = None,
        num_repeats: int = 3,
        epsilon: float = 1e-4
    ) -> None:
        super().__init__()
        
        if out_channels is None:
            out_channels = in_channels
        
        self.num_repeats = num_repeats
        self.epsilon = epsilon
        
        # Initial channel adjustment
        self.p3_adjust = Conv(in_channels['p3'], out_channels['p3'], kernel_size=1)
        self.p4_adjust = Conv(in_channels['p4'], out_channels['p4'], kernel_size=1)
        self.p5_adjust = Conv(in_channels['p5'], out_channels['p5'], kernel_size=1)
        
        # BiFPN blocks
        self.bifpn_blocks = [
            BiFPNBlock(out_channels, epsilon)
            for _ in range(num_repeats)
        ]
        
        self.out_channels = out_channels
    
    def forward(self, features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """BiFPN forward pass."""
        # Initial channel adjustment
        p3 = self.p3_adjust(features['p3'])
        p4 = self.p4_adjust(features['p4'])
        p5 = self.p5_adjust(features['p5'])
        
        features = {'p3': p3, 'p4': p4, 'p5': p5}
        
        # Apply BiFPN blocks
        for block in self.bifpn_blocks:
            features = block(features)
        
        return features


class BiFPNBlock(Module):
    """Single BiFPN block with weighted feature fusion."""
    
    def __init__(self, channels: Dict[str, int], epsilon: float = 1e-4):
        super().__init__()
        
        self.epsilon = epsilon
        
        c3, c4, c5 = channels['p3'], channels['p4'], channels['p5']
        
        # Upsampling
        self.upsample = Upsample(scale_factor=2, mode='nearest')
        
        # Downsampling
        self.downsample_p3 = Conv(c3, c4, kernel_size=3, stride=2)
        self.downsample_p4 = Conv(c4, c5, kernel_size=3, stride=2)
        
        # Weighted fusion convolutions
        # P4_td = Conv(Resize(P5) + P4)
        self.p4_td_conv = Conv(c4, c4, kernel_size=3)
        
        # P3_out = Conv(Resize(P4_td) + P3)
        self.p3_out_conv = Conv(c3, c3, kernel_size=3)
        
        # P4_out = Conv(Resize(P3_out) + P4_td + P4)
        self.p4_out_conv = Conv(c4, c4, kernel_size=3)
        
        # P5_out = Conv(Resize(P4_out) + P5)
        self.p5_out_conv = Conv(c5, c5, kernel_size=3)
        
        # Learnable fusion weights
        self.w_p4_td = Tensor(np.ones(2, dtype=np.float32), requires_grad=True)
        self.w_p3_out = Tensor(np.ones(2, dtype=np.float32), requires_grad=True)
        self.w_p4_out = Tensor(np.ones(3, dtype=np.float32), requires_grad=True)
        self.w_p5_out = Tensor(np.ones(2, dtype=np.float32), requires_grad=True)
    
    def _weighted_fusion(self, features: List[Tensor], weights: Tensor) -> Tensor:
        """Weighted feature fusion with fast normalization."""
        # Fast normalized fusion: sum(w_i * f_i) / sum(w_i) + epsilon
        weights = weights.relu()  # Ensure non-negative
        weight_sum = weights.sum() + self.epsilon
        
        normalized_weights = [w / weight_sum for w in weights.data]
        
        result = features[0] * normalized_weights[0]
        for i, f in enumerate(features[1:], 1):
            result = result + f * normalized_weights[i]
        
        return result
    
    def forward(self, features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """BiFPN block forward pass."""
        p3, p4, p5 = features['p3'], features['p4'], features['p5']
        
        # Top-down pathway
        p5_up = self.upsample(p5)
        p4_td = self._weighted_fusion([p5_up, p4], self.w_p4_td)
        p4_td = self.p4_td_conv(p4_td)
        
        p4_up = self.upsample(p4_td)
        p3_out = self._weighted_fusion([p4_up, p3], self.w_p3_out)
        p3_out = self.p3_out_conv(p3_out)
        
        # Bottom-up pathway
        p3_down = self.downsample_p3(p3_out)
        p4_out = self._weighted_fusion([p3_down, p4_td, p4], self.w_p4_out)
        p4_out = self.p4_out_conv(p4_out)
        
        p4_down = self.downsample_p4(p4_out)
        p5_out = self._weighted_fusion([p4_down, p5], self.w_p5_out)
        p5_out = self.p5_out_conv(p5_out)
        
        return {
            'p3': p3_out,
            'p4': p4_out,
            'p5': p5_out
        }


class YOLONeck(Module):
    """YOLO v12 Neck Network.
    
    Standard PANet neck with optional BiFPN enhancement.
    
    Args:
        in_channels: Dict of input channels from backbone
        out_channels: Output channels (default: same as in_channels)
        neck_type: Type of neck ('panet', 'bifpn', 'fpn')
        num_blocks: Number of C2f/blocks per stage
    """
    
    def __init__(
        self,
        in_channels: Dict[str, int],
        out_channels: Optional[Dict[str, int]] = None,
        neck_type: str = 'panet',
        num_blocks: int = 3
    ) -> None:
        super().__init__()
        
        if out_channels is None:
            out_channels = in_channels
        
        self.neck_type = neck_type
        
        if neck_type == 'panet':
            self.neck = PANet(in_channels, out_channels, num_blocks)
        elif neck_type == 'bifpn':
            self.neck = BiFPN(in_channels, out_channels, num_repeats=num_blocks)
        elif neck_type == 'fpn':
            self.neck = FPN(in_channels, out_channels)
        else:
            raise ValueError(f"Unknown neck type: {neck_type}. "
                           f"Choose from 'panet', 'bifpn', 'fpn'")
        
        self.out_channels = out_channels
    
    def forward(self, features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Forward pass through neck."""
        return self.neck(features)


def build_neck(
    neck_type: str = 'panet',
    in_channels: Optional[Dict[str, int]] = None,
    out_channels: Optional[Dict[str, int]] = None,
    num_blocks: int = 3
) -> YOLONeck:
    """Build YOLO neck by type.
    
    Args:
        neck_type: Type of neck ('panet', 'bifpn', 'fpn')
        in_channels: Input channels from backbone
        out_channels: Output channels (default: same as in_channels)
        num_blocks: Number of blocks per stage
    
    Returns:
        Configured YOLONeck instance
    """
    if in_channels is None:
        # Default channels for YOLO-s
        in_channels = {'p3': 128, 'p4': 256, 'p5': 512}
    
    return YOLONeck(
        in_channels=in_channels,
        out_channels=out_channels,
        neck_type=neck_type,
        num_blocks=num_blocks
    )
