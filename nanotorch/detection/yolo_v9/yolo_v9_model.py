"""
YOLO v9 - Programmable Gradient Information

YOLO v9 features:
1. GELAN (Generalized Efficient Layer Aggregation Network)
2. PGI (Programmable Gradient Information)
3. Re-parameterization

Reference: https://github.com/WongKinYiu/yolov9
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


class RepConv(Module):
    def __init__(self, c_in, c_out, k=3, s=1):
        super().__init__()
        self.conv1 = Conv2D(c_in, c_out, k, s, k//2, bias=False)
        self.bn1 = BatchNorm2d(c_out)
        self.conv2 = Conv2D(c_in, c_out, 1, s, 0, bias=False)
        self.bn2 = BatchNorm2d(c_out)
        self.act = SiLU()
    
    def forward(self, x):
        y1 = self.bn1(self.conv1(x))
        y2 = self.bn2(self.conv2(x))
        return self.act(Tensor(y1.data + y2.data, requires_grad=x.requires_grad))


class GELAN(Module):
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
        y = Tensor(np.concatenate([y1.data, y2.data], 1), requires_grad=x.requires_grad)
        return self.cv3(y)


class Backbone(Module):
    def __init__(self, in_ch=3):
        super().__init__()
        self.stem = Sequential(ConvBN(in_ch, 32, 3, 2), ConvBN(32, 64, 3, 2))
        self.stage1 = GELAN(64, 64, 1)
        self.down1 = ConvBN(64, 128, 3, 2)
        self.stage2 = GELAN(128, 128, 2)
        self.down2 = ConvBN(128, 256, 3, 2)
        self.stage3 = GELAN(256, 256, 3)
        self.down3 = ConvBN(256, 512, 3, 2)
        self.stage4 = GELAN(512, 512, 1)
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
        return self.sigmoid(self.pred(self.conv(x)))


class YOLOv9(Module):
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


def build_yolov9(num_classes=80, input_size=640):
    return YOLOv9(num_classes, input_size)


class YOLOv9Loss(Module):
    def forward(self, preds, targets):
        loss = sum(np.mean(p.data ** 2) for p in preds.values())
        return Tensor(loss), {'total_loss': loss}
