"""
YOLO v5 - Ultralytics YOLO

YOLO v5 introduced by Ultralytics:
1. C3 modules (Cross Stage Partial with 3 conv layers)
2. SiLU/Swish activation
3. SPPF (Spatial Pyramid Pooling Fast)
4. PANet neck
5. Auto anchor calculation
6. Mosaic data augmentation support

Reference:
    Ultralytics YOLOv5
    https://github.com/ultralytics/yolov5
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Union

from nanotorch.tensor import Tensor
from nanotorch.nn.module import Module, Sequential
from nanotorch.nn.conv import Conv2D
from nanotorch.nn.activation import SiLU, Sigmoid
from nanotorch.utils import cat


class ConvBN(Module):
    """Convolution + BatchNorm + SiLU block (standard YOLOv5 conv)."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = None
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        
        self.conv = Conv2D(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = BatchNorm2d(out_channels)
        self.act = SiLU()
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


from nanotorch.nn.normalization import BatchNorm2d


class Bottleneck(Module):
    """Standard bottleneck block with residual connection."""
    
    def __init__(self, in_channels: int, out_channels: int, shortcut: bool = True):
        super().__init__()
        self.conv1 = ConvBN(in_channels, out_channels, kernel_size=1)
        self.conv2 = ConvBN(out_channels, out_channels, kernel_size=3)
        self.shortcut = shortcut and (in_channels == out_channels)
    
    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        if self.shortcut:
            out = out + x
        return out


class C3(Module):
    """CSP Bottleneck with 3 convolutions (C3 module).
    
    This is the core building block of YOLOv5.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 1,
        shortcut: bool = True
    ):
        super().__init__()
        mid_channels = out_channels // 2
        
        self.conv1 = ConvBN(in_channels, mid_channels, kernel_size=1)
        self.conv2 = ConvBN(in_channels, mid_channels, kernel_size=1)
        self.conv3 = ConvBN(2 * mid_channels, out_channels, kernel_size=1)
        
        self.blocks = Sequential(*[
            Bottleneck(mid_channels, mid_channels, shortcut)
            for _ in range(num_blocks)
        ])
    
    def forward(self, x: Tensor) -> Tensor:
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        
        x1 = self.blocks(x1)
        
        x = cat([x1, x2], dim=1)
        x = self.conv3(x)
        
        return x


class SPPF(Module):
    """Spatial Pyramid Pooling Fast.
    
    Uses a single max pool with varying kernel sizes through
    successive pooling operations.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        mid_channels = in_channels // 2
        
        self.conv1 = ConvBN(in_channels, mid_channels, kernel_size=1)
        self.conv2 = ConvBN(mid_channels * 4, out_channels, kernel_size=1)
        self.kernel_size = kernel_size
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        
        y1 = x
        y2 = self._max_pool(y1)
        y3 = self._max_pool(y2)
        y4 = self._max_pool(y3)
        
        x = cat([y1, y2, y3, y4], dim=1)
        x = self.conv2(x)
        
        return x
    
    def _max_pool(self, x: Tensor) -> Tensor:
        k = self.kernel_size
        data = x.data
        n, c, h, w = data.shape
        
        output = np.zeros_like(data)
        
        for i in range(h):
            for j in range(w):
                i_start = max(0, i - k // 2)
                i_end = min(h, i + k // 2 + 1)
                j_start = max(0, j - k // 2)
                j_end = min(w, j + k // 2 + 1)
                
                output[:, :, i, j] = np.max(data[:, :, i_start:i_end, j_start:j_end], axis=(2, 3))
        
        return Tensor(output, requires_grad=x.requires_grad)


class Backbone(Module):
    """YOLOv5 backbone (CSPDarknet-based)."""
    
    def __init__(self, in_channels: int = 3, version: str = 's'):
        super().__init__()
        
        depth_multiple = {'n': 0.33, 's': 0.33, 'm': 0.67, 'l': 1.0, 'x': 1.33}.get(version, 0.33)
        width_multiple = {'n': 0.25, 's': 0.50, 'm': 0.75, 'l': 1.00, 'x': 1.25}.get(version, 0.50)
        
        def make_round(x):
            return max(int(x * depth_multiple), 1)
        
        def make_divisible(x):
            return max(int(x * width_multiple), 1)
        
        c1 = make_divisible(64)
        c2 = make_divisible(128)
        c3 = make_divisible(256)
        c4 = make_divisible(512)
        c5 = make_divisible(1024)
        
        self.conv1 = ConvBN(in_channels, c1, kernel_size=6, stride=2, padding=2)
        self.conv2 = ConvBN(c1, c2, kernel_size=3, stride=2)
        self.c3_1 = C3(c2, c2, num_blocks=make_round(3))
        
        self.conv3 = ConvBN(c2, c3, kernel_size=3, stride=2)
        self.c3_2 = C3(c3, c3, num_blocks=make_round(6))
        
        self.conv4 = ConvBN(c3, c4, kernel_size=3, stride=2)
        self.c3_3 = C3(c4, c4, num_blocks=make_round(9))
        
        self.conv5 = ConvBN(c4, c5, kernel_size=3, stride=2)
        self.c3_4 = C3(c5, c5, num_blocks=make_round(3))
        self.sppf = SPPF(c5, c5, kernel_size=5)
        
        self.out_channels = [c3, c4, c5]
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.c3_1(x)
        
        x = self.conv3(x)
        x = self.c3_2(x)
        scale3 = x
        
        x = self.conv4(x)
        x = self.c3_3(x)
        scale2 = x
        
        x = self.conv5(x)
        x = self.c3_4(x)
        x = self.sppf(x)
        scale1 = x
        
        return {
            'scale1': scale1,
            'scale2': scale2,
            'scale3': scale3
        }


class Neck(Module):
    """YOLOv5 PANet neck."""
    
    def __init__(self, in_channels: List[int] = None):
        super().__init__()
        
        if in_channels is None:
            in_channels = [256, 512, 1024]
        
        c3, c4, c5 = in_channels
        
        # Top-down pathway
        self.up1 = ConvBN(c5, c4, kernel_size=1)
        self.c3_1 = C3(c4 + c4, c4, num_blocks=3, shortcut=False)
        
        self.up2 = ConvBN(c4, c3, kernel_size=1)
        self.c3_2 = C3(c3 + c3, c3, num_blocks=3, shortcut=False)
        
        # Bottom-up pathway
        self.down1 = ConvBN(c3, c3, kernel_size=3, stride=2)
        self.c3_3 = C3(c3 + c4, c4, num_blocks=3, shortcut=False)
        
        self.down2 = ConvBN(c4, c4, kernel_size=3, stride=2)
        self.c3_4 = C3(c4 + c5, c5, num_blocks=3, shortcut=False)
    
    def _upsample(self, x: Tensor, target_size: Tuple[int, int]) -> Tensor:
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
        
        # Top-down
        x = self.up1(scale1)
        target_size = (scale2.shape[2], scale2.shape[3])
        x = self._upsample(x, target_size)
        x = cat([x, scale2], dim=1)
        x = self.c3_1(x)
        p4 = x
        
        x = self.up2(p4)
        target_size = (scale3.shape[2], scale3.shape[3])
        x = self._upsample(x, target_size)
        x = cat([x, scale3], dim=1)
        x = self.c3_2(x)
        p3 = x
        
        # Bottom-up
        x = self.down1(p3)
        x = cat([x, p4], dim=1)
        x = self.c3_3(x)
        n4 = x
        
        x = self.down2(n4)
        x = cat([x, scale1], dim=1)
        x = self.c3_4(x)
        n5 = x
        
        return {
            'p3': p3,
            'p4': n4,
            'p5': n5
        }


class DetectHead(Module):
    """YOLOv5 detection head."""
    
    def __init__(self, in_channels: int, num_anchors: int = 3, num_classes: int = 80):
        super().__init__()
        
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        
        self.conv = Conv2D(in_channels, num_anchors * (5 + num_classes), kernel_size=1)
        self.sigmoid = Sigmoid()
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.sigmoid(x)
        return x


class YOLOv5(Module):
    """Complete YOLOv5 model.
    
    Features:
        - C3 modules
        - SPPF for spatial pyramid pooling
        - PANet neck
        - Auto anchor support
    
    Args:
        num_classes: Number of object classes
        input_size: Input image size
        version: Model size ('n', 's', 'm', 'l', 'x')
    """
    
    DEFAULT_ANCHORS = [
        [(10, 13), (16, 30), (33, 23)],
        [(30, 61), (62, 45), (59, 119)],
        [(116, 90), (156, 198), (373, 326)]
    ]
    
    def __init__(
        self,
        num_classes: int = 80,
        input_size: int = 640,
        version: str = 's'
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        self.version = version
        
        self.backbone = Backbone(in_channels=3, version=version)
        self.neck = Neck(self.backbone.out_channels)
        
        c3, c4, c5 = self.backbone.out_channels
        
        self.head_small = DetectHead(c5, num_anchors=3, num_classes=num_classes)
        self.head_medium = DetectHead(c4, num_anchors=3, num_classes=num_classes)
        self.head_large = DetectHead(c3, num_anchors=3, num_classes=num_classes)
    
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


class YOLOv5Nano(YOLOv5):
    """YOLOv5 Nano - smallest version."""
    
    def __init__(self, num_classes: int = 80, input_size: int = 640):
        super().__init__(num_classes=num_classes, input_size=input_size, version='n')


class YOLOv5Small(YOLOv5):
    """YOLOv5 Small - default version."""
    
    def __init__(self, num_classes: int = 80, input_size: int = 640):
        super().__init__(num_classes=num_classes, input_size=input_size, version='s')


def build_yolov5(
    version: str = 's',
    num_classes: int = 80,
    input_size: int = 640
) -> YOLOv5:
    """Build YOLOv5 model.
    
    Args:
        version: Model size ('n', 's', 'm', 'l', 'x')
        num_classes: Number of object classes
        input_size: Input image size
    
    Returns:
        YOLOv5 model
    """
    return YOLOv5(num_classes=num_classes, input_size=input_size, version=version)
