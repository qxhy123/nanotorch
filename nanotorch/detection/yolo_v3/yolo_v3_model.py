"""
YOLO v3 (You Only Look Once) - Third Generation

YOLO v3 introduced several key improvements:
1. Darknet-53 backbone with residual connections
2. Feature Pyramid Network (FPN) for multi-scale detection
3. Three detection scales: 13x13, 26x26, 52x52 for 416x416 input
4. Independent logistic classifiers instead of softmax
5. 9 anchor boxes (3 per scale)

Reference:
    "YOLOv3: An Incremental Improvement"
    Joseph Redmon, Ali Farhadi
    2018
    https://arxiv.org/abs/1804.02767
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Union

from nanotorch.tensor import Tensor
from nanotorch.nn.module import Module, Sequential
from nanotorch.nn.conv import Conv2D
from nanotorch.nn.pooling import MaxPool2d, AvgPool2d
from nanotorch.nn.activation import LeakyReLU, Sigmoid
from nanotorch.nn.normalization import BatchNorm2d
from nanotorch.utils import cat


class ConvBN(Module):
    """Convolution + BatchNorm + LeakyReLU block."""
    
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


class ResidualBlock(Module):
    """Residual block with two conv layers."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBN(channels, channels // 2, kernel_size=1)
        self.conv2 = ConvBN(channels // 2, channels, kernel_size=3, padding=1)
    
    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        return x + residual


class Darknet53(Module):
    """Darknet-53 backbone network.
    
    Architecture:
        - 53 layers total
        - Residual connections
        - No fully connected layers
        - Output: 3 feature maps at different scales
    """
    
    def __init__(self, in_channels: int = 3):
        super().__init__()
        
        self.conv1 = ConvBN(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = ConvBN(32, 64, kernel_size=3, stride=2, padding=1)
        
        self.res1 = self._make_residual(64, 1)
        self.conv3 = ConvBN(64, 128, kernel_size=3, stride=2, padding=1)
        
        self.res2 = self._make_residual(128, 2)
        self.conv4 = ConvBN(128, 256, kernel_size=3, stride=2, padding=1)
        
        self.res3 = self._make_residual(256, 8)
        self.conv5 = ConvBN(256, 512, kernel_size=3, stride=2, padding=1)
        
        self.res4 = self._make_residual(512, 8)
        self.conv6 = ConvBN(512, 1024, kernel_size=3, stride=2, padding=1)
        
        self.res5 = self._make_residual(1024, 4)
        
        self.out_channels = [256, 512, 1024]
    
    def _make_residual(self, channels: int, num_blocks: int) -> Sequential:
        layers = []
        for _ in range(num_blocks):
            layers.append(ResidualBlock(channels))
        return Sequential(*layers)
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.res1(x)
        x = self.conv3(x)
        
        x = self.res2(x)
        x = self.conv4(x)
        
        x = self.res3(x)
        scale3 = x
        
        x = self.conv5(x)
        x = self.res4(x)
        scale2 = x
        
        x = self.conv6(x)
        x = self.res5(x)
        scale1 = x
        
        return {
            'scale1': scale1,
            'scale2': scale2,
            'scale3': scale3
        }


class FPN(Module):
    """Feature Pyramid Network for YOLO v3.
    
    Fuses features from different scales through upsampling and concatenation.
    """
    
    def __init__(self, in_channels: List[int] = None):
        super().__init__()
        
        if in_channels is None:
            in_channels = [1024, 512, 256]
        
        C5, C4, C3 = in_channels
        
        self.conv5_1 = ConvBN(C5, 512, kernel_size=1)
        self.conv5_2 = ConvBN(512, 1024, kernel_size=3, padding=1)
        self.conv5_3 = ConvBN(1024, 512, kernel_size=1)
        self.conv5_4 = ConvBN(512, 1024, kernel_size=3, padding=1)
        self.conv5_5 = ConvBN(1024, 512, kernel_size=1)
        
        self.conv5_upsample = ConvBN(512, 256, kernel_size=1)
        
        self.conv4_1 = ConvBN(C4 + 256, 256, kernel_size=1)
        self.conv4_2 = ConvBN(256, 512, kernel_size=3, padding=1)
        self.conv4_3 = ConvBN(512, 256, kernel_size=1)
        self.conv4_4 = ConvBN(256, 512, kernel_size=3, padding=1)
        self.conv4_5 = ConvBN(512, 256, kernel_size=1)
        
        self.conv4_upsample = ConvBN(256, 128, kernel_size=1)
        
        self.conv3_1 = ConvBN(C3 + 128, 128, kernel_size=1)
        self.conv3_2 = ConvBN(128, 256, kernel_size=3, padding=1)
        self.conv3_3 = ConvBN(256, 128, kernel_size=1)
        self.conv3_4 = ConvBN(128, 256, kernel_size=3, padding=1)
        self.conv3_5 = ConvBN(256, 128, kernel_size=1)
    
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
        
        x = self.conv5_1(scale1)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.conv5_4(x)
        x = self.conv5_5(x)
        
        p5 = x
        
        x = self.conv5_upsample(x)
        target_size = (scale2.shape[2], scale2.shape[3])
        x_up = self._upsample(x, target_size)
        x = cat([x_up, scale2], dim=1)
        
        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)
        x = self.conv4_5(x)
        
        p4 = x
        
        x = self.conv4_upsample(x)
        target_size = (scale3.shape[2], scale3.shape[3])
        x_up = self._upsample(x, target_size)
        x = cat([x_up, scale3], dim=1)
        
        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)
        x = self.conv3_5(x)
        
        p3 = x
        
        return {
            'p3': p3,
            'p4': p4,
            'p5': p5
        }


class YOLOHead(Module):
    """YOLO v3 detection head.
    
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
        
        self.conv1 = ConvBN(in_channels, in_channels * 2, kernel_size=3, padding=1)
        self.conv2 = Conv2D(in_channels * 2, self.output_channels, kernel_size=1)
        self.sigmoid = Sigmoid()
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sigmoid(x)
        return x


class YOLOv3(Module):
    """Complete YOLO v3 model.
    
    Features:
        - Darknet-53 backbone
        - FPN for multi-scale features
        - 3 detection scales
        - 9 anchor boxes total (3 per scale)
    
    Args:
        num_classes: Number of object classes (default: 80 for COCO)
        input_size: Input image size (default: 416)
    """
    
    COCO_ANCHORS = [
        [(116, 90), (156, 198), (373, 326)],
        [(30, 61), (62, 45), (59, 119)],
        [(10, 13), (16, 30), (33, 23)]
    ]
    
    def __init__(
        self,
        num_classes: int = 80,
        input_size: int = 416
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        
        self.backbone = Darknet53(in_channels=3)
        self.fpn = FPN(self.backbone.out_channels[::-1])
        
        self.head_small = YOLOHead(512, num_anchors=3, num_classes=num_classes)
        self.head_medium = YOLOHead(256, num_anchors=3, num_classes=num_classes)
        self.head_large = YOLOHead(128, num_anchors=3, num_classes=num_classes)
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        features = self.backbone(x)
        fpn_features = self.fpn(features)
        
        pred_small = self.head_small(fpn_features['p5'])
        pred_medium = self.head_medium(fpn_features['p4'])
        pred_large = self.head_large(fpn_features['p3'])
        
        return {
            'small': pred_small,
            'medium': pred_medium,
            'large': pred_large
        }


class YOLOv3Tiny(Module):
    """Tiny version of YOLO v3 for faster inference.
    
    Uses fewer layers in the backbone for reduced computation.
    """
    
    def __init__(
        self,
        num_classes: int = 80,
        input_size: int = 416
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        
        self.features = Sequential(
            ConvBN(3, 16, kernel_size=3, padding=1),
            MaxPool2d(2, 2),
            
            ConvBN(16, 32, kernel_size=3, padding=1),
            MaxPool2d(2, 2),
            
            ConvBN(32, 64, kernel_size=3, padding=1),
            MaxPool2d(2, 2),
            
            ConvBN(64, 128, kernel_size=3, padding=1),
            MaxPool2d(2, 2),
            
            ConvBN(128, 256, kernel_size=3, padding=1),
            MaxPool2d(2, 2),
            
            ConvBN(256, 512, kernel_size=3, padding=1),
            MaxPool2d(2, 1),
            
            ConvBN(512, 1024, kernel_size=3, padding=1),
        )
        
        self.conv1 = ConvBN(1024, 256, kernel_size=1)
        self.conv2 = ConvBN(256, 512, kernel_size=3, padding=1)
        self.conv3 = Conv2D(512, 3 * (5 + num_classes), kernel_size=1)
        self.sigmoid = Sigmoid()
        
        self.out_channels = [256, 512]
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.features(x)
        
        x = self.conv1(x)
        route = x
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.sigmoid(x)
        
        return {
            'small': x,
            'route': route
        }


def build_yolov3(
    model_type: str = 'full',
    num_classes: int = 80,
    input_size: int = 416
) -> Union[YOLOv3, YOLOv3Tiny]:
    """Build YOLO v3 model.
    
    Args:
        model_type: 'full' or 'tiny'
        num_classes: Number of object classes
        input_size: Input image size
    
    Returns:
        YOLOv3 or YOLOv3Tiny model
    """
    if model_type == 'tiny':
        return YOLOv3Tiny(num_classes=num_classes, input_size=input_size)
    return YOLOv3(num_classes=num_classes, input_size=input_size)
