"""
YOLO v2 (YOLO9000) - Better, Faster, Stronger

YOLO v2 introduced several key improvements:
1. Batch Normalization on all layers
2. High Resolution Classifier (fine-tune on 448x448)
3. Convolutional With Anchor Boxes (instead of FC layers)
4. Dimension Clusters (K-means for anchor selection)
5. Direct location prediction (sigmoid for tx, ty)
6. Fine-Grained Features (passthrough layer)
7. Multi-Scale Training
8. Darknet-19 backbone

Reference:
    "YOLO9000: Better, Faster, Stronger"
    Joseph Redmon, Ali Farhadi
    CVPR 2017
    https://arxiv.org/abs/1612.08242
"""

import numpy as np
from typing import Tuple, Dict, List, Optional, Union

from nanotorch.tensor import Tensor
from nanotorch.nn.module import Module, Sequential
from nanotorch.nn.conv import Conv2D
from nanotorch.nn.pooling import MaxPool2d
from nanotorch.nn.activation import LeakyReLU, Sigmoid
from nanotorch.nn.normalization import BatchNorm2d
from nanotorch.utils import cat


class ConvBN(Module):
    """Convolution + BatchNorm + LeakyReLU block.
    
    This is the basic building block for Darknet-19.
    All convolutional layers in YOLO v2 use batch normalization.
    """
    
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


class Darknet19(Module):
    """Darknet-19 backbone network.
    
    Architecture (19 convolutional layers):
        - Input: 416 x 416 x 3
        - Uses BatchNorm on all conv layers
        - No fully connected layers (fully convolutional)
        - 5 maxpool layers for downsampling
        
    Layer structure:
        conv 3x3 (32) -> maxpool
        conv 3x3 (64) -> maxpool
        conv 3x3 (128) -> conv 1x1 (64) -> conv 3x3 (128) -> maxpool
        conv 3x3 (256) -> conv 1x1 (128) -> conv 3x3 (256) -> maxpool
        conv 3x3 (512) -> [conv 1x1 (256) -> conv 3x3 (512)] x4 -> maxpool
        conv 3x3 (1024) -> [conv 1x1 (512) -> conv 3x3 (1024)] x2
    
    Output channels: [512, 1024, 1024]
    """
    
    def __init__(self, in_channels: int = 3):
        super().__init__()
        
        self.conv1 = ConvBN(in_channels, 32, kernel_size=3, padding=1)
        self.pool1 = MaxPool2d(2, 2)
        
        self.conv2 = ConvBN(32, 64, kernel_size=3, padding=1)
        self.pool2 = MaxPool2d(2, 2)
        
        self.conv3 = ConvBN(64, 128, kernel_size=3, padding=1)
        self.conv4 = ConvBN(128, 64, kernel_size=1)
        self.conv5 = ConvBN(64, 128, kernel_size=3, padding=1)
        self.pool3 = MaxPool2d(2, 2)
        
        self.conv6 = ConvBN(128, 256, kernel_size=3, padding=1)
        self.conv7 = ConvBN(256, 128, kernel_size=1)
        self.conv8 = ConvBN(128, 256, kernel_size=3, padding=1)
        self.pool4 = MaxPool2d(2, 2)
        
        self.conv9 = ConvBN(256, 512, kernel_size=3, padding=1)
        self.conv10 = ConvBN(512, 256, kernel_size=1)
        self.conv11 = ConvBN(256, 512, kernel_size=3, padding=1)
        self.conv12 = ConvBN(512, 256, kernel_size=1)
        self.conv13 = ConvBN(256, 512, kernel_size=3, padding=1)
        self.conv14 = ConvBN(512, 256, kernel_size=1)
        self.conv15 = ConvBN(256, 512, kernel_size=3, padding=1)
        self.conv16 = ConvBN(512, 256, kernel_size=1)
        self.conv17 = ConvBN(256, 512, kernel_size=3, padding=1)
        self.pool5 = MaxPool2d(2, 2)
        
        self.conv18 = ConvBN(512, 1024, kernel_size=3, padding=1)
        self.conv19 = ConvBN(1024, 512, kernel_size=1)
        self.conv20 = ConvBN(512, 1024, kernel_size=3, padding=1)
        self.conv21 = ConvBN(1024, 512, kernel_size=1)
        self.conv22 = ConvBN(512, 1024, kernel_size=3, padding=1)
        
        self.out_channels = [512, 1024, 1024]
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pool3(x)
        
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        route1 = x
        x = self.pool4(x)
        
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.conv16(x)
        x = self.conv17(x)
        route2 = x
        x = self.pool5(x)
        
        x = self.conv18(x)
        x = self.conv19(x)
        x = self.conv20(x)
        x = self.conv21(x)
        x = self.conv22(x)
        
        return {
            'output': x,
            'route1': route1,
            'route2': route2
        }


class PassthroughLayer(Module):
    """Passthrough layer for fine-grained features.
    
    Reorganizes feature map from higher resolution to lower resolution
    by stacking adjacent features into channels.
    
    Example (stride=2):
        Input: (N, C, H, W)
        Output: (N, C*4, H/2, W/2)
    
    This allows the network to access fine-grained features from
    earlier layers, improving detection of small objects.
    """
    
    def __init__(self, stride: int = 2):
        super().__init__()
        self.stride = stride
    
    def forward(self, x: Tensor) -> Tensor:
        data = x.data
        n, c, h, w = data.shape
        s = self.stride
        
        out_h, out_w = h // s, w // s
        out_c = c * s * s
        
        out = np.zeros((n, out_c, out_h, out_w), dtype=data.dtype)
        
        for i in range(out_h):
            for j in range(out_w):
                for si in range(s):
                    for sj in range(s):
                        src_i = i * s + si
                        src_j = j * s + sj
                        out_ch = (si * s + sj) * c
                        out[:, out_ch:out_ch + c, i, j] = data[:, :, src_i, src_j]
        
        return Tensor(out, requires_grad=x.requires_grad)


class YOLOv2Head(Module):
    """YOLO v2 detection head.
    
    Features:
        - Uses anchor boxes (5 anchors for YOLO v2)
        - Direct location prediction (sigmoid)
        - Passthrough layer for fine-grained features
        - Output: (N, num_anchors * (5 + num_classes), H, W)
    
    Anchor boxes for 416x416 input:
        (0.57273, 0.677385), (1.87446, 2.06253), 
        (3.33843, 5.47434), (7.88282, 3.52778), (9.77052, 9.16828)
    """
    
    VOC_ANCHORS = [
        (1.3221, 1.73145),
        (3.19275, 4.00944),
        (5.05587, 8.09892),
        (9.47112, 4.84053),
        (11.2364, 10.0071)
    ]
    
    def __init__(
        self,
        in_channels: int = 1024,
        passthrough_channels: int = 512,
        num_anchors: int = 5,
        num_classes: int = 20
    ):
        super().__init__()
        
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.output_channels = num_anchors * (5 + num_classes)
        
        self.passthrough = PassthroughLayer(stride=2)
        self.passthrough_conv = ConvBN(passthrough_channels * 4, 256, kernel_size=1)
        
        self.conv1 = ConvBN(in_channels, 1024, kernel_size=3, padding=1)
        self.conv2 = ConvBN(1024, 1024, kernel_size=3, padding=1)
        
        self.conv3 = ConvBN(1024 + 256, 1024, kernel_size=3, padding=1)
        self.conv4 = Conv2D(1024, self.output_channels, kernel_size=1)
        
        self.sigmoid = Sigmoid()
    
    def forward(self, x: Tensor, passthrough_features: Tensor) -> Tensor:
        passthrough = self.passthrough(passthrough_features)
        passthrough = self.passthrough_conv(passthrough)
        
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = cat([x, passthrough], dim=1)
        
        x = self.conv3(x)
        x = self.conv4(x)
        
        x = self.sigmoid(x)
        
        return x


class YOLOv2(Module):
    """Complete YOLO v2 model.
    
    Features:
        - Darknet-19 backbone
        - Passthrough layer for fine-grained features
        - 5 anchor boxes per grid cell
        - Direct location prediction
        - Multi-scale training support
    
    Args:
        num_classes: Number of object classes (default: 20 for PASCAL VOC)
        input_size: Input image size (default: 416)
        num_anchors: Number of anchor boxes (default: 5)
    
    Shape:
        - Input: (N, 3, input_size, input_size)
        - Output: (N, num_anchors * (5 + num_classes), H, W)
          where H = W = input_size / 32
    """
    
    def __init__(
        self,
        num_classes: int = 20,
        input_size: int = 416,
        num_anchors: int = 5
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        self.num_anchors = num_anchors
        
        self.backbone = Darknet19(in_channels=3)
        self.head = YOLOv2Head(
            in_channels=1024,
            passthrough_channels=512,
            num_anchors=num_anchors,
            num_classes=num_classes
        )
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        features = self.backbone(x)
        
        output = self.head(features['output'], features['route2'])
        
        return {'output': output}


class YOLOv2Tiny(Module):
    """Tiny version of YOLO v2 for faster inference.
    
    Uses fewer layers in the backbone for reduced computation.
    """
    
    def __init__(
        self,
        num_classes: int = 20,
        input_size: int = 416,
        num_anchors: int = 5
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        self.num_anchors = num_anchors
        
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
        
        self.output_channels = num_anchors * (5 + num_classes)
        self.conv1 = ConvBN(1024, 256, kernel_size=1)
        self.conv2 = ConvBN(256, 512, kernel_size=3, padding=1)
        self.conv3 = Conv2D(512, self.output_channels, kernel_size=1)
        self.sigmoid = Sigmoid()
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        x = self.features(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.sigmoid(x)
        
        return {'output': x}


def build_yolov2(
    model_type: str = 'full',
    num_classes: int = 20,
    input_size: int = 416,
    num_anchors: int = 5
) -> Union[YOLOv2, YOLOv2Tiny]:
    """Build YOLO v2 model.
    
    Args:
        model_type: 'full' or 'tiny'
        num_classes: Number of object classes
        input_size: Input image size
        num_anchors: Number of anchor boxes
    
    Returns:
        YOLOv2 or YOLOv2Tiny model
    """
    if model_type == 'tiny':
        return YOLOv2Tiny(
            num_classes=num_classes,
            input_size=input_size,
            num_anchors=num_anchors
        )
    return YOLOv2(
        num_classes=num_classes,
        input_size=input_size,
        num_anchors=num_anchors
    )
