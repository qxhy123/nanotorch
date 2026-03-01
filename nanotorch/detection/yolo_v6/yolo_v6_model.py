"""
YOLO v6 - MT-YOLOv6

YOLO v6 introduced by Meituan:
1. RepVGG-style backbone (reparameterization)
2. Decoupled head (separate classification and regression)
3. SiLU activation
4. Efficient training and inference

Reference:
    "YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications"
    https://github.com/meituan/YOLOv6
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Union

from nanotorch.tensor import Tensor
from nanotorch.nn.module import Module, Sequential
from nanotorch.nn.conv import Conv2D
from nanotorch.nn.activation import SiLU, Sigmoid, ReLU
from nanotorch.nn.normalization import BatchNorm2d


class ConvBN(Module):
    """Convolution + BatchNorm + SiLU."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = None):
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


class RepVGGBlock(Module):
    """RepVGG block with reparameterization support.
    
    Training: Uses 3x3 conv + 1x1 conv + identity
    Inference: Can be fused into single 3x3 conv
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = Conv2D(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = BatchNorm2d(out_channels)
        
        self.conv2 = Conv2D(in_channels, out_channels, 1, stride, 0, bias=False)
        self.bn2 = BatchNorm2d(out_channels)
        
        self.identity = (in_channels == out_channels) and (stride == 1)
        
        self.act = SiLU()
    
    def forward(self, x: Tensor) -> Tensor:
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        
        out2 = self.conv2(x)
        out2 = self.bn2(out2)
        
        if self.identity:
            identity = x
            out = Tensor(out1.data + out2.data + identity.data, requires_grad=x.requires_grad)
        else:
            out = Tensor(out1.data + out2.data, requires_grad=x.requires_grad)
        
        return self.act(out)


class SimConv(Module):
    """Simple ConvBN block."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        super().__init__()
        self.conv = ConvBN(in_channels, out_channels, kernel_size, stride)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class SimSPPF(Module):
    """Simplified SPPF."""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super().__init__()
        mid = in_channels // 2
        self.conv1 = ConvBN(in_channels, mid, 1)
        self.conv2 = ConvBN(mid * 4, out_channels, 1)
        self.k = kernel_size
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        
        y1 = x
        y2 = self._pool(y1)
        y3 = self._pool(y2)
        y4 = self._pool(y3)
        
        x = Tensor(np.concatenate([y1.data, y2.data, y3.data, y4.data], axis=1), requires_grad=x.requires_grad)
        return self.conv2(x)
    
    def _pool(self, x: Tensor) -> Tensor:
        data = x.data
        n, c, h, w = data.shape
        k = self.k
        output = np.zeros_like(data)
        
        for i in range(h):
            for j in range(w):
                i_s, i_e = max(0, i - k//2), min(h, i + k//2 + 1)
                j_s, j_e = max(0, j - k//2), min(w, j + k//2 + 1)
                output[:, :, i, j] = np.max(data[:, :, i_s:i_e, j_s:j_e], axis=(2, 3))
        
        return Tensor(output, requires_grad=x.requires_grad)


class Backbone(Module):
    """YOLOv6 backbone using RepVGG blocks."""
    
    def __init__(self, in_channels: int = 3, variant: str = 's'):
        super().__init__()
        
        scales = {'n': 0.33, 's': 0.33, 'm': 0.60, 'l': 1.0}
        scale = scales.get(variant, 0.33)
        
        def ch(x):
            return max(int(x * scale), 1)
        
        c1, c2, c3, c4, c5 = ch(64), ch(128), ch(256), ch(512), ch(1024)
        
        self.stem = Sequential(
            ConvBN(in_channels, c1, 3, 2),
            ConvBN(c1, c2, 3, 2),
        )
        
        self.stage2 = Sequential(
            RepVGGBlock(c2, c3, stride=2),
            RepVGGBlock(c3, c3),
        )
        
        self.stage3 = Sequential(
            RepVGGBlock(c3, c4, stride=2),
            RepVGGBlock(c4, c4),
            RepVGGBlock(c4, c4),
        )
        
        self.stage4 = Sequential(
            RepVGGBlock(c4, c5, stride=2),
            RepVGGBlock(c5, c5),
            RepVGGBlock(c5, c5),
        )
        
        self.sppf = SimSPPF(c5, c5)
        
        self.out_channels = [c3, c4, c5]
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.stem(x)
        
        x = self.stage2(x)
        scale3 = x
        
        x = self.stage3(x)
        scale2 = x
        
        x = self.stage4(x)
        x = self.sppf(x)
        scale1 = x
        
        return {'scale1': scale1, 'scale2': scale2, 'scale3': scale3}


class Neek(Module):
    """YOLOv6 neck (FPN + PAN)."""
    
    def __init__(self, in_channels: List[int] = None):
        super().__init__()
        
        if in_channels is None:
            in_channels = [256, 512, 1024]
        
        c3, c4, c5 = in_channels
        
        self.up1 = ConvBN(c5, c4, 1)
        self.c3_1 = ConvBN(c4 + c4, c4, 3)
        
        self.up2 = ConvBN(c4, c3, 1)
        self.c3_2 = ConvBN(c3 + c3, c3, 3)
        
        self.down1 = ConvBN(c3, c3, 3, 2)
        self.c3_3 = ConvBN(c3 + c4, c4, 3)
        
        self.down2 = ConvBN(c4, c4, 3, 2)
        self.c3_4 = ConvBN(c4 + c5, c5, 3)
    
    def _upsample(self, x: Tensor, target: Tuple[int, int]) -> Tensor:
        h, w = target
        data = x.data
        n, c, _, _ = data.shape
        out = np.zeros((n, c, h, w), dtype=data.dtype)
        for i in range(h):
            for j in range(w):
                out[:, :, i, j] = data[:, :, min(i, data.shape[2]-1), min(j, data.shape[3]-1)]
        return Tensor(out, requires_grad=x.requires_grad)
    
    def forward(self, features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        s1, s2, s3 = features['scale1'], features['scale2'], features['scale3']
        
        x = self.up1(s1)
        x = self._upsample(x, (s2.shape[2], s2.shape[3]))
        x = Tensor(np.concatenate([x.data, s2.data], axis=1), requires_grad=x.requires_grad)
        x = self.c3_1(x)
        p4 = x
        
        x = self.up2(p4)
        x = self._upsample(x, (s3.shape[2], s3.shape[3]))
        x = Tensor(np.concatenate([x.data, s3.data], axis=1), requires_grad=x.requires_grad)
        x = self.c3_2(x)
        p3 = x
        
        x = self.down1(p3)
        x = Tensor(np.concatenate([x.data, p4.data], axis=1), requires_grad=x.requires_grad)
        x = self.c3_3(x)
        n4 = x
        
        x = self.down2(n4)
        x = Tensor(np.concatenate([x.data, s1.data], axis=1), requires_grad=x.requires_grad)
        x = self.c3_4(x)
        n5 = x
        
        return {'p3': p3, 'p4': n4, 'p5': n5}


class DecoupledHead(Module):
    """Decoupled detection head (separate cls and reg)."""
    
    def __init__(self, in_channels: int, num_anchors: int = 1, num_classes: int = 80):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        
        self.cls_conv = ConvBN(in_channels, in_channels, 3)
        self.cls_pred = Conv2D(in_channels, num_anchors * num_classes, 1)
        
        self.reg_conv = ConvBN(in_channels, in_channels, 3)
        self.reg_pred = Conv2D(in_channels, num_anchors * 4, 1)
        
        self.sigmoid = Sigmoid()
    
    def forward(self, x: Tensor) -> Tensor:
        cls = self.cls_conv(x)
        cls = self.cls_pred(cls)
        cls = self.sigmoid(cls)
        
        reg = self.reg_conv(x)
        reg = self.reg_pred(reg)
        reg = self.sigmoid(reg)
        
        n, _, h, w = cls.data.shape
        cls = cls.data.reshape(n, self.num_anchors, self.num_classes, h, w)
        reg = reg.data.reshape(n, self.num_anchors, 4, h, w)
        
        conf = np.ones((n, self.num_anchors, 1, h, w), dtype=np.float32)
        out = np.concatenate([reg, conf, cls], axis=2)
        out = out.reshape(n, -1, h, w)
        
        return Tensor(out, requires_grad=x.requires_grad)


class YOLOv6(Module):
    """YOLOv6 model."""
    
    def __init__(self, num_classes: int = 80, input_size: int = 640, variant: str = 's'):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        
        self.backbone = Backbone(in_channels=3, variant=variant)
        self.neck = Neek(self.backbone.out_channels)
        
        c3, c4, c5 = self.backbone.out_channels
        
        self.head_small = DecoupledHead(c5, num_anchors=1, num_classes=num_classes)
        self.head_medium = DecoupledHead(c4, num_anchors=1, num_classes=num_classes)
        self.head_large = DecoupledHead(c3, num_anchors=1, num_classes=num_classes)
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        features = self.backbone(x)
        neck_features = self.neck(features)
        
        return {
            'small': self.head_small(neck_features['p5']),
            'medium': self.head_medium(neck_features['p4']),
            'large': self.head_large(neck_features['p3'])
        }


class YOLOv6Nano(YOLOv6):
    def __init__(self, num_classes: int = 80, input_size: int = 640):
        super().__init__(num_classes, input_size, 'n')


class YOLOv6Small(YOLOv6):
    def __init__(self, num_classes: int = 80, input_size: int = 640):
        super().__init__(num_classes, input_size, 's')


def build_yolov6(variant: str = 's', num_classes: int = 80, input_size: int = 640) -> YOLOv6:
    return YOLOv6(num_classes=num_classes, input_size=input_size, variant=variant)
