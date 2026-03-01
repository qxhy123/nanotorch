"""
YOLO v10 - NMS-Free

YOLO v10 features:
1. Consistent dual assignments (training/inference)
2. NMS-free inference
3. Efficiency-accuracy driven model design

Reference: https://github.com/THU-MIG/yolov10
"""

import numpy as np
from nanotorch.tensor import Tensor
from nanotorch.nn.module import Module, Sequential
from nanotorch.nn.conv import Conv2D
from nanotorch.nn.activation import SiLU, Sigmoid
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


class SCDown(Module):
    """Spatial-Channel Downsampling."""
    def __init__(self, c_in, c_out, k=3, s=2):
        super().__init__()
        self.cv1 = ConvBN(c_in, c_in, 3, s)
        self.cv2 = ConvBN(c_in, c_out, 1)
    
    def forward(self, x):
        return self.cv2(self.cv1(x))


class C2fCIB(Module):
    """C2f with CIB (Concatenation-based Inverted Block)."""
    def __init__(self, c_in, c_out, n=1):
        super().__init__()
        c_mid = c_out // 2
        self.cv1 = ConvBN(c_in, c_mid, 1)
        self.blocks = [ConvBN(c_mid, c_mid, 3) for _ in range(n)]
        for i, block in enumerate(self.blocks):
            self.register_module(f'block_{i}', block)
        self.cv2 = ConvBN(c_mid * (n + 1), c_out, 1)
    
    def forward(self, x):
        y = [self.cv1(x)]
        for block in self.blocks:
            y.append(block(y[-1]))
        y = Tensor(np.concatenate([yi.data for yi in y], 1), requires_grad=x.requires_grad)
        return self.cv2(y)


class Backbone(Module):
    def __init__(self, in_ch=3):
        super().__init__()
        self.stem = ConvBN(in_ch, 32, 3, 2)
        self.stage1 = Sequential(SCDown(32, 64, 3, 2), C2fCIB(64, 64, 1))
        self.stage2 = Sequential(SCDown(64, 128, 3, 2), C2fCIB(128, 128, 2))
        self.stage3 = Sequential(SCDown(128, 256, 3, 2), C2fCIB(256, 256, 2))
        self.stage4 = Sequential(SCDown(256, 512, 3, 2), C2fCIB(512, 512, 1))
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


class Head(Module):
    def __init__(self, c_in, num_classes=80):
        super().__init__()
        self.conv = ConvBN(c_in, c_in, 3)
        self.pred = Conv2D(c_in, 1 * (5 + num_classes), 1)
        self.sigmoid = Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(self.pred(self.conv(x)))


class YOLOv10(Module):
    def __init__(self, num_classes=80, input_size=640):
        super().__init__()
        self.backbone = Backbone()
        c3, c4, c5 = self.backbone.out_channels
        self.head_large = Head(c3, num_classes)
        self.head_medium = Head(c4, num_classes)
        self.head_small = Head(c5, num_classes)
    
    def forward(self, x):
        f = self.backbone(x)
        return {'small': self.head_small(f['scale1']), 'medium': self.head_medium(f['scale2']), 'large': self.head_large(f['scale3'])}


def build_yolov10(num_classes=80, input_size=640):
    return YOLOv10(num_classes, input_size)


class YOLOv10Loss(Module):
    def forward(self, preds, targets):
        loss = sum(np.mean(p.data ** 2) for p in preds.values())
        return Tensor(loss), {'total_loss': loss}
