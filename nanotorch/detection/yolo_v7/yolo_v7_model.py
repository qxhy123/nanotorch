"""
YOLO v7 - Trainable Bag-of-Freebies

YOLO v7 introduced:
1. E-ELAN (Extended Efficient Layer Aggregation Network)
2. Model scaling techniques
3. Auxiliary training heads

Reference: https://github.com/WongKinYiu/yolov7
"""

import numpy as np
from typing import Dict, List, Tuple
from nanotorch.tensor import Tensor
from nanotorch.nn.module import Module, Sequential
from nanotorch.nn.conv import Conv2D
from nanotorch.nn.activation import SiLU, Sigmoid
from nanotorch.utils import cat
from nanotorch.nn.normalization import BatchNorm2d


class ConvBN(Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=None):
        super().__init__()
        p = p if p is not None else k // 2
        self.conv = Conv2D(c_in, c_out, k, s, p, bias=False)
        self.bn = BatchNorm2d(c_out)
        self.act = SiLU()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ELAN(Module):
    """Efficient Layer Aggregation Network."""
    def __init__(self, c_in, c_out, n=1):
        super().__init__()
        c_mid = c_out // 2
        self.cv1 = ConvBN(c_in, c_mid, 1)
        self.cv2 = ConvBN(c_in, c_mid, 1)
        self.blocks = Sequential(*[ConvBN(c_mid, c_mid, 3) for _ in range(n)])
        self.cv3 = ConvBN(c_mid * 2, c_out, 1)
    
    def forward(self, x):
        y1 = self.cv1(x)
        y2 = self.cv2(x)
        y2 = self.blocks(y2)
        y = cat([y1, y2], dim=1)
        return self.cv3(y)


class Backbone(Module):
    def __init__(self, in_ch=3):
        super().__init__()
        self.stem = Sequential(ConvBN(in_ch, 32, 3, 2), ConvBN(32, 64, 3, 2))
        self.stage1 = ELAN(64, 64, 1)
        self.down1 = ConvBN(64, 128, 3, 2)
        self.stage2 = ELAN(128, 128, 2)
        self.down2 = ConvBN(128, 256, 3, 2)
        self.stage3 = ELAN(256, 256, 3)
        self.down3 = ConvBN(256, 512, 3, 2)
        self.stage4 = ELAN(512, 512, 1)
        self.out_channels = [128, 256, 512]
    
    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.down1(x)
        x = self.stage2(x)
        s3 = x
        x = self.down2(x)
        x = self.stage3(x)
        s2 = x
        x = self.down3(x)
        s1 = self.stage4(x)
        return {'scale1': s1, 'scale2': s2, 'scale3': s3}


class Head(Module):
    def __init__(self, c_in, num_classes=80):
        super().__init__()
        self.conv = ConvBN(c_in, c_in, 3)
        self.pred = Conv2D(c_in, 1 * (5 + num_classes), 1)
        self.sigmoid = Sigmoid()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pred(x)
        return self.sigmoid(x)


class Neck(Module):
    def __init__(self, channels):
        super().__init__()
        c3, c4, c5 = channels
        self.up1 = ConvBN(c5, c4, 1)
        self.c3_1 = ConvBN(c4 * 2, c4, 3)
        self.up2 = ConvBN(c4, c3, 1)
        self.c3_2 = ConvBN(c3 * 2, c3, 3)
        self.down1 = ConvBN(c3, c3, 3, 2)
        self.c3_3 = ConvBN(c3 + c4, c4, 3)
        self.down2 = ConvBN(c4, c4, 3, 2)
        self.c3_4 = ConvBN(c4 + c5, c5, 3)
    
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
        p4 = self.c3_1(x)
        x = self._up(self.up2(p4), (s3.shape[2], s3.shape[3]))
        x = cat([x, s3], dim=1)
        p3 = self.c3_2(x)
        x = cat([self.down1(p3), p4], dim=1)
        n4 = self.c3_3(x)
        x = cat([self.down2(n4), s1], dim=1)
        n5 = self.c3_4(x)
        return {'p3': p3, 'p4': n4, 'p5': n5}


class YOLOv7(Module):
    def __init__(self, num_classes=80, input_size=640):
        super().__init__()
        self.backbone = Backbone()
        self.neck = Neck(self.backbone.out_channels)
        c3, c4, c5 = self.backbone.out_channels
        self.head_large = Head(c3, num_classes)
        self.head_medium = Head(c4, num_classes)
        self.head_small = Head(c5, num_classes)
    
    def forward(self, x):
        f = self.backbone(x)
        n = self.neck(f)
        return {'small': self.head_small(n['p5']), 'medium': self.head_medium(n['p4']), 'large': self.head_large(n['p3'])}


def build_yolov7(num_classes=80, input_size=640):
    return YOLOv7(num_classes, input_size)


class YOLOv7Loss(Module):
    def forward(self, preds, targets):
        loss = sum((p * p).mean() for p in preds.values())
        return loss, {'total_loss': float(loss.data)}
