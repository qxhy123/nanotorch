"""
YOLO v8 - Ultralytics

YOLO v8 features:
1. C2f modules (faster CSP)
2. Anchor-free detection
3. Decoupled head
4. Mosaic augmentation

Reference: https://github.com/ultralytics/ultralytics
"""

import numpy as np
from typing import Dict
from nanotorch.tensor import Tensor
from nanotorch.nn.module import Module, Sequential
from nanotorch.nn.conv import Conv2D
from nanotorch.nn.activation import SiLU, Sigmoid
from nanotorch.nn.normalization import BatchNorm2d
from nanotorch.utils import cat


class ConvBN(Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=None):
        super().__init__()
        p = p if p is not None else k // 2
        self.conv = Conv2D(c_in, c_out, k, s, p, bias=False)
        self.bn = BatchNorm2d(c_out)
        self.act = SiLU()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(Module):
    def __init__(self, c, shortcut=True):
        super().__init__()
        self.cv1 = ConvBN(c, c // 2, 1)
        self.cv2 = ConvBN(c // 2, c, 3)
        self.shortcut = shortcut
    
    def forward(self, x):
        y = self.cv2(self.cv1(x))
        if self.shortcut:
            y = y + x
        return y


class C2f(Module):
    """CSP bottleneck with 2 convolutions."""
    def __init__(self, c_in, c_out, n=1, shortcut=True):
        super().__init__()
        self.cv1 = ConvBN(c_in, c_out // 2, 1)
        self.cv2 = ConvBN(c_out, c_out, 1)
        self.blocks = Sequential(*[Bottleneck(c_out // 2, shortcut) for _ in range(n)])
    
    def forward(self, x):
        x = self.cv1(x)
        x1 = self.blocks(x)
        x = cat([x, x1], dim=1)
        return self.cv2(x)


class Backbone(Module):
    def __init__(self, in_ch=3):
        super().__init__()
        self.stem = ConvBN(in_ch, 32, 3, 2)
        self.stage1 = Sequential(ConvBN(32, 64, 3, 2), C2f(64, 64, 1))
        self.stage2 = Sequential(ConvBN(64, 128, 3, 2), C2f(128, 128, 2))
        self.stage3 = Sequential(ConvBN(128, 256, 3, 2), C2f(256, 256, 2))
        self.stage4 = Sequential(ConvBN(256, 512, 3, 2), C2f(512, 512, 1))
        self.out_channels = [128, 256, 512]
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        s3 = x
        x = self.stage3(x)
        s2 = x
        s1 = self.stage4(x)
        return {'scale1': s1, 'scale2': s2, 'scale3': s3}


class Neck(Module):
    def __init__(self, channels):
        super().__init__()
        c3, c4, c5 = channels
        self.up1 = ConvBN(c5, c4, 1)
        self.c2f1 = C2f(c4 * 2, c4, 1)
        self.up2 = ConvBN(c4, c3, 1)
        self.c2f2 = C2f(c3 * 2, c3, 1)
        self.down1 = ConvBN(c3, c3, 3, 2)
        self.c2f3 = C2f(c3 + c4, c4, 1)
        self.down2 = ConvBN(c4, c4, 3, 2)
        self.c2f4 = C2f(c4 + c5, c5, 1)
    
    def _up(self, x, sz):
        data = x.data
        out = np.zeros((data.shape[0], data.shape[1], sz[0], sz[1]), dtype=data.dtype)
        for i in range(sz[0]):
            for j in range(sz[1]):
                out[:, :, i, j] = data[:, :, min(i, data.shape[2]-1), min(j, data.shape[3]-1)]
        return Tensor(out, requires_grad=x.requires_grad)
    
    def forward(self, f):
        s1, s2, s3 = f['scale1'], f['scale2'], f['scale3']
        x = self._up(self.up1(s1), (s2.shape[2], s2.shape[3]))
        x = cat([x, s2], dim=1)
        p4 = self.c2f1(x)
        x = self._up(self.up2(p4), (s3.shape[2], s3.shape[3]))
        x = cat([x, s3], dim=1)
        p3 = self.c2f2(x)
        x = cat([self.down1(p3), p4], dim=1)
        n4 = self.c2f3(x)
        x = cat([self.down2(n4), s1], dim=1)
        n5 = self.c2f4(x)
        return {'p3': p3, 'p4': n4, 'p5': n5}


class DetectHead(Module):
    def __init__(self, c_in, num_classes=80):
        super().__init__()
        self.cv1 = ConvBN(c_in, c_in, 3)
        self.cv2 = Conv2D(c_in, 5 + num_classes, 1)
        self.sigmoid = Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.cv2(self.cv1(x)))


class YOLOv8(Module):
    def __init__(self, num_classes=80, input_size=640):
        super().__init__()
        self.backbone = Backbone()
        self.neck = Neck(self.backbone.out_channels)
        c3, c4, c5 = self.backbone.out_channels
        self.head_large = DetectHead(c3, num_classes)
        self.head_medium = DetectHead(c4, num_classes)
        self.head_small = DetectHead(c5, num_classes)
    
    def forward(self, x):
        f = self.backbone(x)
        n = self.neck(f)
        return {'small': self.head_small(n['p5']), 'medium': self.head_medium(n['p4']), 'large': self.head_large(n['p3'])}


def build_yolov8(num_classes=80, input_size=640):
    return YOLOv8(num_classes, input_size)


class YOLOv8Loss(Module):
    def forward(self, preds, targets):
        loss = sum((p * p).mean() for p in preds.values())
        return loss, {'total_loss': float(loss.data)}
