"""
YOLO v4 (You Only Look Once) - Fourth Generation

YOLO v4 introduced several key improvements:
1. CSPDarknet53 backbone with Cross Stage Partial connections
2. SPP (Spatial Pyramid Pooling) module
3. PANet (Path Aggregation Network) for better feature fusion
4. Mish activation function
5. Bag of Freebies (BoF) and Bag of Specials (BoS)

Reference:
    "YOLOv4: Optimal Speed and Accuracy of Object Detection"
    Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao
    2020
    https://arxiv.org/abs/2004.10934
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Union

from nanotorch.tensor import Tensor
from nanotorch.nn.module import Module, Sequential
from nanotorch.nn.conv import Conv2D
from nanotorch.nn.linear import Linear
from nanotorch.nn.pooling import MaxPool2d, AvgPool2d
from nanotorch.nn.activation import LeakyReLU, Sigmoid, ReLU
from nanotorch.nn.normalization import BatchNorm2d


class Mish(Module):
    """Mish activation function: x * tanh(softplus(x))
    
    Mish is a smooth, non-monotonic activation function that tends to 
    work better than ReLU in deep networks.
    """
    
    def forward(self, x: Tensor) -> Tensor:
        # softplus = ln(1 + exp(x))
        # mish = x * tanh(softplus)
        # Approximate with x * sigmoid for efficiency
        # Full implementation: x * tanh(ln(1 + exp(x)))
        data = x.data
        # softplus
        sp = np.log1p(np.exp(np.clip(data, -20, 20)))
        # tanh of softplus
        tanh_sp = np.tanh(sp)
        # mish = x * tanh(softplus)
        result = data * tanh_sp
        return Tensor(result, requires_grad=x.requires_grad)


class ConvBNMish(Module):
    """Convolution + BatchNorm + Mish activation block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0
    ):
        super().__init__()
        self.conv = Conv2D(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = BatchNorm2d(out_channels)
        self.activation = Mish()
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class ConvBNLeaky(Module):
    """Convolution + BatchNorm + LeakyReLU activation block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 0
    ):
        super().__init__()
        self.conv = Conv2D(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = BatchNorm2d(out_channels)
        self.activation = LeakyReLU(0.1)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.activation(x)
        return x


class ResBlock(Module):
    """Residual block with two conv layers."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBNMish(channels, channels // 2, kernel_size=1)
        self.conv2 = ConvBNMish(channels // 2, channels, kernel_size=3, padding=1)
    
    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual


class CSPResBlock(Module):
    """CSP (Cross Stage Partial) Residual Block.
    
    Splits input into two parts, processes one through residual blocks,
    then concatenates and merges.
    """
    
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 1):
        super().__init__()
        mid_channels = out_channels // 2
        
        # Split path 1: direct connection
        self.split_conv = ConvBNMish(in_channels, mid_channels, kernel_size=1)
        
        # Split path 2: through residual blocks
        self.transition_conv = ConvBNMish(in_channels, mid_channels, kernel_size=1)
        
        # Residual blocks
        self.res_blocks = Sequential(*[
            ResBlock(mid_channels) for _ in range(num_blocks)
        ])
        
        # Merge
        self.merge_conv = ConvBNMish(out_channels, out_channels, kernel_size=1)
    
    def forward(self, x: Tensor) -> Tensor:
        # Split
        path1 = self.split_conv(x)
        path2 = self.transition_conv(x)
        
        # Process path2 through residual blocks
        path2 = self.res_blocks(path2)
        
        # Concatenate
        merged = Tensor(np.concatenate([path1.data, path2.data], axis=1), requires_grad=x.requires_grad)
        
        # Merge
        return self.merge_conv(merged)


class SPP(Module):
    """Spatial Pyramid Pooling module.
    
    Applies multiple pooling operations at different scales and concatenates
    the results to capture multi-scale context.
    """
    
    def __init__(self, in_channels: int, out_channels: int, pool_sizes: List[int] = [5, 9, 13]):
        super().__init__()
        self.pool_sizes = pool_sizes
        
        # Reduce channels first
        self.conv1 = ConvBNMish(in_channels, out_channels // 2, kernel_size=1)
        
        # After pooling and concat, we have:
        # original (out_channels // 2) + len(pool_sizes) * (out_channels // 2)
        concat_channels = out_channels // 2 * (len(pool_sizes) + 1)
        self.conv2 = ConvBNMish(concat_channels, out_channels, kernel_size=1)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        
        pool_outputs = [x]
        
        for pool_size in self.pool_sizes:
            pooled = self._max_pool(x, pool_size)
            pool_outputs.append(pooled)
        
        concat = Tensor(np.concatenate([p.data for p in pool_outputs], axis=1), requires_grad=x.requires_grad)
        
        return self.conv2(concat)
    
    def _max_pool(self, x: Tensor, kernel_size: int) -> Tensor:
        """Max pooling with padding to maintain size."""
        data = x.data
        n, c, h, w = data.shape
        k = min(kernel_size, h, w)
        
        if k <= 1:
            return x
        
        output = np.zeros_like(data)
        
        for i in range(h):
            for j in range(w):
                i_start = max(0, i - k // 2)
                i_end = min(h, i + k // 2 + 1)
                j_start = max(0, j - k // 2)
                j_end = min(w, j + k // 2 + 1)
                
                output[:, :, i, j] = np.max(data[:, :, i_start:i_end, j_start:j_end], axis=(2, 3))
        
        return Tensor(output, requires_grad=x.requires_grad)


class CSPDarknet53(Module):
    """CSPDarknet53 backbone network.
    
    CSPDarknet53 uses Cross Stage Partial connections to reduce
    computation while maintaining accuracy.
    
    Architecture:
        - CSP-based residual blocks
        - Mish activation function
        - Output: 3 feature maps at different scales
    """
    
    def __init__(self, in_channels: int = 3):
        super().__init__()
        
        # Stem
        self.conv1 = ConvBNMish(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = ConvBNMish(32, 64, kernel_size=3, stride=2, padding=1)
        
        # CSP Block 1
        self.csp1 = CSPResBlock(64, 64, num_blocks=1)
        self.conv3 = ConvBNMish(64, 128, kernel_size=3, stride=2, padding=1)
        
        # CSP Block 2
        self.csp2 = CSPResBlock(128, 128, num_blocks=2)
        self.conv4 = ConvBNMish(128, 256, kernel_size=3, stride=2, padding=1)
        
        # CSP Block 3 - Output scale 3 (large feature map)
        self.csp3 = CSPResBlock(256, 256, num_blocks=8)
        self.conv5 = ConvBNMish(256, 512, kernel_size=3, stride=2, padding=1)
        
        # CSP Block 4 - Output scale 2 (medium feature map)
        self.csp4 = CSPResBlock(512, 512, num_blocks=8)
        self.conv6 = ConvBNMish(512, 1024, kernel_size=3, stride=2, padding=1)
        
        # CSP Block 5 - Output scale 1 (small feature map)
        self.csp5 = CSPResBlock(1024, 1024, num_blocks=4)
        
        # SPP module
        self.spp = SPP(1024, 512, pool_sizes=[5, 9, 13])
        
        self.out_channels = [256, 512, 512]
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # Stem
        x = self.conv1(x)
        x = self.conv2(x)
        
        # CSP blocks
        x = self.csp1(x)
        x = self.conv3(x)
        
        x = self.csp2(x)
        x = self.conv4(x)
        
        x = self.csp3(x)
        scale3 = x  # 256 channels
        
        x = self.conv5(x)
        x = self.csp4(x)
        scale2 = x  # 512 channels
        
        x = self.conv6(x)
        x = self.csp5(x)
        x = self.spp(x)
        scale1 = x  # 512 channels (after SPP)
        
        return {
            'scale1': scale1,
            'scale2': scale2,
            'scale3': scale3
        }


class PANet(Module):
    """Path Aggregation Network for YOLO v4.
    
    PANet improves feature fusion by:
    1. Bottom-up pathway (FPN-like)
    2. Top-down pathway for better localization
    3. Adaptive feature pooling
    """
    
    def __init__(self, in_channels: List[int] = None):
        super().__init__()
        
        if in_channels is None:
            in_channels = [512, 512, 256]  # scale1, scale2, scale3
        
        C5, C4, C3 = in_channels
        
        # Top-down pathway
        self.conv5_1 = ConvBNLeaky(C5, 256, kernel_size=1)
        self.conv5_2 = ConvBNLeaky(256, 512, kernel_size=3, padding=1)
        self.conv5_upsample = ConvBNLeaky(512, 256, kernel_size=1)
        
        self.conv4_1 = ConvBNLeaky(C4 + 256, 256, kernel_size=1)
        self.conv4_2 = ConvBNLeaky(256, 512, kernel_size=3, padding=1)
        self.conv4_upsample = ConvBNLeaky(512, 128, kernel_size=1)
        
        self.conv3_1 = ConvBNLeaky(C3 + 128, 128, kernel_size=1)
        self.conv3_2 = ConvBNLeaky(128, 256, kernel_size=3, padding=1)
        
        # Bottom-up pathway
        self.conv3_down = ConvBNLeaky(256, 128, kernel_size=3, stride=2, padding=1)
        self.conv4_3 = ConvBNLeaky(256, 256, kernel_size=1)  # 128 + 128 = 256
        self.conv4_4 = ConvBNLeaky(256, 512, kernel_size=3, padding=1)
        
        self.conv4_down = ConvBNLeaky(512, 256, kernel_size=3, stride=2, padding=1)
        self.conv5_3 = ConvBNLeaky(512, 256, kernel_size=1)  # 256 + 256 = 512
        self.conv5_4 = ConvBNLeaky(256, 512, kernel_size=3, padding=1)
    
    def _upsample(self, x: Tensor, target_size: Tuple[int, int]) -> Tensor:
        """Nearest neighbor upsampling."""
        h, w = target_size
        data = x.data
        n, c, _, _ = data.shape
        
        upsampled = np.zeros((n, c, h, w), dtype=data.dtype)
        
        for i in range(h):
            for j in range(w):
                src_i = min(i, data.shape[2] - 1)
                src_j = min(j, data.shape[3] - 1)
                upsampled[:, :, i, j] = data[:, :, src_i, src_j]
        
        return Tensor(upsampled, requires_grad=x.requires_grad)
    
    def forward(self, features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        scale1 = features['scale1']
        scale2 = features['scale2']
        scale3 = features['scale3']
        
        # Top-down pathway
        x = self.conv5_1(scale1)
        x = self.conv5_2(x)
        p5 = x
        
        x = self.conv5_upsample(x)
        target_size = (scale2.shape[2], scale2.shape[3])
        x_up = self._upsample(x, target_size)
        x = Tensor(np.concatenate([x_up.data, scale2.data], axis=1), requires_grad=x.requires_grad)
        
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        p4 = x
        
        x = self.conv4_upsample(x)
        target_size = (scale3.shape[2], scale3.shape[3])
        x_up = self._upsample(x, target_size)
        x = Tensor(np.concatenate([x_up.data, scale3.data], axis=1), requires_grad=x.requires_grad)
        
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        p3 = x
        
        # Bottom-up pathway
        x = self.conv3_down(p3)
        x = Tensor(np.concatenate([x.data, self.conv4_upsample(p4).data], axis=1), 
                   requires_grad=x.requires_grad)
        x = self.conv4_3(x)
        x = self.conv4_4(x)
        n4 = x
        
        x = self.conv4_down(n4)
        x = Tensor(np.concatenate([x.data, self.conv5_upsample(p5).data], axis=1),
                   requires_grad=x.requires_grad)
        x = self.conv5_3(x)
        x = self.conv5_4(x)
        n5 = x
        
        return {
            'p3': p3,
            'p4': n4,
            'p5': n5
        }


class YOLOHead(Module):
    """YOLO v4 detection head.
    
    Each scale produces predictions for 3 anchor boxes.
    Output per anchor: (x, y, w, h, confidence, class_probs)
    """
    
    def __init__(
        self,
        in_channels: int,
        num_anchors: int = 3,
        num_classes: int = 80
    ):
        super().__init__()
        
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        
        self.output_channels = num_anchors * (5 + num_classes)
        
        self.conv1 = ConvBNLeaky(in_channels, in_channels * 2, kernel_size=3, padding=1)
        self.conv2 = Conv2D(in_channels * 2, self.output_channels, kernel_size=1)
        self.sigmoid = Sigmoid()
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x


class YOLOv4(Module):
    """Complete YOLO v4 model.
    
    Features:
        - CSPDarknet53 backbone with SPP
        - PANet for feature fusion
        - 3 detection scales
        - 9 anchor boxes total (3 per scale)
    
    Args:
        num_classes: Number of object classes (default: 80 for COCO)
        input_size: Input image size (default: 416)
    """
    
    # YOLO v4 anchors (similar to v3 but optimized)
    ANCHORS = [
        [(142, 110), (192, 243), (459, 401)],      # Large scale (13x13)
        [(36, 75), (76, 55), (72, 146)],           # Medium scale (26x26)
        [(12, 16), (19, 36), (40, 28)]             # Small scale (52x52)
    ]
    
    def __init__(
        self,
        num_classes: int = 80,
        input_size: int = 416
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        
        self.backbone = CSPDarknet53(in_channels=3)
        self.neck = PANet(self.backbone.out_channels[::-1])
        
        self.head_small = YOLOHead(512, num_anchors=3, num_classes=num_classes)
        self.head_medium = YOLOHead(512, num_anchors=3, num_classes=num_classes)
        self.head_large = YOLOHead(256, num_anchors=3, num_classes=num_classes)
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        features = self.backbone(x)
        neck_features = self.neck(features)
        
        pred_small = self.head_small(neck_features['p5'])
        pred_medium = self.head_medium(neck_features['p4'])
        pred_large = self.head_large(neck_features['p3'])
        
        return {
            'small': pred_small,
            'medium': pred_medium,
            'large': pred_large
        }


class YOLOv4Tiny(Module):
    """Tiny version of YOLO v4 for faster inference.
    
    Uses simplified CSP and fewer layers.
    """
    
    def __init__(
        self,
        num_classes: int = 80,
        input_size: int = 416
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        
        # Simplified backbone
        self.conv1 = ConvBNLeaky(3, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = ConvBNLeaky(32, 64, kernel_size=3, stride=2, padding=1)
        
        self.csp1 = CSPResBlock(64, 64, num_blocks=1)
        self.conv3 = ConvBNLeaky(64, 128, kernel_size=3, stride=2, padding=1)
        
        self.csp2 = CSPResBlock(128, 128, num_blocks=2)
        self.conv4 = ConvBNLeaky(128, 256, kernel_size=3, stride=2, padding=1)
        
        self.csp3 = CSPResBlock(256, 256, num_blocks=2)
        self.conv5 = ConvBNLeaky(256, 512, kernel_size=3, stride=2, padding=1)
        
        self.csp4 = CSPResBlock(512, 512, num_blocks=1)
        self.spp = SPP(512, 256, pool_sizes=[5, 9, 13])
        
        # Head
        self.conv6 = ConvBNLeaky(256, 128, kernel_size=1)
        self.conv7 = ConvBNLeaky(256, 256, kernel_size=3, padding=1)  # 128 + 128 = 256
        self.conv8 = Conv2D(256, 3 * (5 + num_classes), kernel_size=1)
        self.sigmoid = Sigmoid()
        
        # Route layer output
        self.route_conv = ConvBNLeaky(256, 128, kernel_size=1)
        
        self.out_channels = [256, 128]
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # Backbone
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.csp1(x)
        x = self.conv3(x)
        
        x = self.csp2(x)
        x = self.conv4(x)
        route = x  # Save for later
        
        x = self.csp3(x)
        x = self.conv5(x)
        
        x = self.csp4(x)
        x = self.spp(x)
        
        # Head
        x = self.conv6(x)
        
        # Upsample and concatenate with route
        h, w = route.shape[2], route.shape[3]
        x_up = self._upsample(x, (h, w))
        route_conv = self.route_conv(route)
        x = Tensor(np.concatenate([x_up.data, route_conv.data], axis=1), requires_grad=x.requires_grad)
        
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.sigmoid(x)
        
        return {
            'small': x,
            'route': route
        }
    
    def _upsample(self, x: Tensor, target_size: Tuple[int, int]) -> Tensor:
        """Nearest neighbor upsampling."""
        h, w = target_size
        data = x.data
        n, c, _, _ = data.shape
        
        upsampled = np.zeros((n, c, h, w), dtype=data.dtype)
        
        for i in range(h):
            for j in range(w):
                src_i = min(i, data.shape[2] - 1)
                src_j = min(j, data.shape[3] - 1)
                upsampled[:, :, i, j] = data[:, :, src_i, src_j]
        
        return Tensor(upsampled, requires_grad=x.requires_grad)


def build_yolov4(
    model_type: str = 'full',
    num_classes: int = 80,
    input_size: int = 416
) -> Union[YOLOv4, YOLOv4Tiny]:
    """Build YOLO v4 model.
    
    Args:
        model_type: 'full' or 'tiny'
        num_classes: Number of object classes
        input_size: Input image size
    
    Returns:
        YOLOv4 or YOLOv4Tiny model
    """
    if model_type == 'tiny':
        return YOLOv4Tiny(num_classes=num_classes, input_size=input_size)
    return YOLOv4(num_classes=num_classes, input_size=input_size)
