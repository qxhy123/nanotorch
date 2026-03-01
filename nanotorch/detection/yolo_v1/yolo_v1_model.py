"""
YOLO v1 (You Only Look Once) - Original Implementation

YOLO v1 was the first single-stage object detector, proposed in 2015.
It frames object detection as a regression problem to spatially separated
bounding boxes and associated class probabilities.

Key features:
- Single neural network predicts bounding boxes and class probabilities
- Divides image into S×S grid (S=7 by default)
- Each grid cell predicts B bounding boxes (B=2 by default)
- Each bounding box has 5 values: (x, y, w, h, confidence)
- Each grid cell also predicts C class probabilities (C=20 for PASCAL VOC)
- Final output: S × S × (B×5 + C) = 7 × 7 × 30 = 1470 values

Reference:
    "You Only Look Once: Unified, Real-Time Object Detection"
    Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi
    CVPR 2016
    https://arxiv.org/abs/1506.02640
"""

from typing import Tuple, Dict, Optional, Union
from nanotorch.tensor import Tensor
from nanotorch.nn.module import Module, Sequential
from nanotorch.nn.conv import Conv2D
from nanotorch.nn.linear import Linear
from nanotorch.nn.pooling import MaxPool2d
from nanotorch.nn.activation import LeakyReLU
from nanotorch.nn.dropout import Dropout
from nanotorch.nn.activation import Flatten


class ConvBlock(Module):
    """Convolutional block: Conv2D -> BatchNorm (optional) -> LeakyReLU.
    
    This is the basic building block for the Darknet backbone.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride (default: 1)
        padding: Padding (default: auto-computed for 'same' when stride=1)
        use_bn: Whether to use batch normalization (default: False for original YOLO v1)
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        use_bn: bool = False
    ) -> None:
        super().__init__()
        
        if padding is None:
            padding = kernel_size // 2
        
        self.conv = Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )
        
        self.activation = LeakyReLU(negative_slope=0.1)
        
        self.use_bn = use_bn
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.activation(x)
        return x


class Darknet(Module):
    """Darknet-24 backbone for YOLO v1.
    
    The original YOLO v1 uses a 24-layer convolutional network inspired by
    GoogLeNet, followed by 2 fully connected layers.
    
    Architecture (input 448×448):
        Conv 7×7, 64, stride 2  -> 224×224×64
        MaxPool 2×2, stride 2   -> 112×112×64
        Conv 3×3, 192           -> 112×112×192
        MaxPool 2×2, stride 2   -> 56×56×192
        Conv 1×1, 128           -> 56×56×128
        Conv 3×3, 256           -> 56×56×256
        Conv 1×1, 256           -> 56×56×256
        Conv 3×3, 512           -> 56×56×512
        MaxPool 2×2, stride 2   -> 28×28×512
        [4×] Conv 1×1, 256 + Conv 3×3, 512
        Conv 1×1, 512           -> 28×28×512
        Conv 3×3, 1024          -> 28×28×1024
        MaxPool 2×2, stride 2   -> 14×14×1024
        [2×] Conv 1×1, 512 + Conv 3×3, 1024
        Conv 3×3, 1024          -> 14×14×1024
        Conv 3×3, 1024          -> 14×14×1024
        Conv 3×3, 1024, stride 2-> 7×7×1024
        Conv 3×3, 1024          -> 7×7×1024
        Conv 3×3, 1024          -> 7×7×1024
    
    Args:
        in_channels: Number of input channels (default: 3 for RGB)
    """
    
    def __init__(self, in_channels: int = 3) -> None:
        super().__init__()
        
        self.features = Sequential(
            ConvBlock(in_channels, 64, kernel_size=7, stride=2, padding=3),
            MaxPool2d(kernel_size=2, stride=2),
            
            ConvBlock(64, 192, kernel_size=3, stride=1, padding=1),
            MaxPool2d(kernel_size=2, stride=2),
            
            ConvBlock(192, 128, kernel_size=1, stride=1, padding=0),
            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1),
            ConvBlock(256, 256, kernel_size=1, stride=1, padding=0),
            ConvBlock(256, 512, kernel_size=3, stride=1, padding=1),
            MaxPool2d(kernel_size=2, stride=2),
            
            ConvBlock(512, 256, kernel_size=1, stride=1, padding=0),
            ConvBlock(256, 512, kernel_size=3, stride=1, padding=1),
            ConvBlock(512, 256, kernel_size=1, stride=1, padding=0),
            ConvBlock(256, 512, kernel_size=3, stride=1, padding=1),
            ConvBlock(512, 256, kernel_size=1, stride=1, padding=0),
            ConvBlock(256, 512, kernel_size=3, stride=1, padding=1),
            ConvBlock(512, 256, kernel_size=1, stride=1, padding=0),
            ConvBlock(256, 512, kernel_size=3, stride=1, padding=1),
            
            ConvBlock(512, 512, kernel_size=1, stride=1, padding=0),
            ConvBlock(512, 1024, kernel_size=3, stride=1, padding=1),
            MaxPool2d(kernel_size=2, stride=2),
            
            ConvBlock(1024, 512, kernel_size=1, stride=1, padding=0),
            ConvBlock(512, 1024, kernel_size=3, stride=1, padding=1),
            ConvBlock(1024, 512, kernel_size=1, stride=1, padding=0),
            ConvBlock(512, 1024, kernel_size=3, stride=1, padding=1),
            ConvBlock(1024, 1024, kernel_size=3, stride=1, padding=1),
            ConvBlock(1024, 1024, kernel_size=3, stride=2, padding=1),
            
            ConvBlock(1024, 1024, kernel_size=3, stride=1, padding=1),
            ConvBlock(1024, 1024, kernel_size=3, stride=1, padding=1),
        )
        
        self.out_channels = 1024
    
    def forward(self, x: Tensor) -> Tensor:
        return self.features(x)


class YOLOv1Head(Module):
    """YOLO v1 detection head.
    
    The detection head consists of two fully connected layers that transform
    the feature map into the final detection output.
    
    Args:
        in_channels: Number of input channels (default: 1024)
        hidden_dim: Hidden layer dimension (default: 4096)
        S: Grid size (default: 7)
        B: Number of bounding boxes per grid cell (default: 2)
        C: Number of classes (default: 20)
        dropout: Dropout rate (default: 0.5)
    """
    
    def __init__(
        self,
        in_channels: int = 1024,
        hidden_dim: int = 4096,
        S: int = 7,
        B: int = 2,
        C: int = 20,
        dropout: float = 0.5
    ) -> None:
        super().__init__()
        
        self.S = S
        self.B = B
        self.C = C
        
        output_dim = S * S * (B * 5 + C)
        
        self.flatten = Flatten()
        self.fc1 = Linear(in_channels * S * S, hidden_dim)
        self.dropout = Dropout(dropout)
        self.activation = LeakyReLU(negative_slope=0.1)
        self.fc2 = Linear(hidden_dim, output_dim)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def reshape_output(self, x: Tensor) -> Tensor:
        """Reshape flat output to (N, S, S, B*5+C)."""
        N = x.shape[0]
        return x.reshape((N, self.S, self.S, self.B * 5 + self.C))


class YOLOv1(Module):
    """Complete YOLO v1 model.
    
    YOLO v1 divides the input image into an S×S grid. Each grid cell predicts
    B bounding boxes and C class probabilities.
    
    Output format per grid cell (B*5 + C values):
        - For each of B boxes: [x, y, w, h, confidence]
        - C class probabilities: [p1, p2, ..., pC]
    
    Args:
        input_size: Input image size (default: 448)
        S: Grid size (default: 7)
        B: Number of bounding boxes per grid cell (default: 2)
        C: Number of classes (default: 20)
        dropout: Dropout rate in detection head (default: 0.5)
    
    Shape:
        - Input: (N, 3, input_size, input_size)
        - Output: (N, S, S, B*5+C) or (N, S*S*(B*5+C))
    """
    
    def __init__(
        self,
        input_size: int = 448,
        S: int = 7,
        B: int = 2,
        C: int = 20,
        dropout: float = 0.5
    ) -> None:
        super().__init__()
        
        self.input_size = input_size
        self.S = S
        self.B = B
        self.C = C
        
        self.backbone = Darknet(in_channels=3)
        self.head = YOLOv1Head(
            in_channels=self.backbone.out_channels,
            hidden_dim=4096,
            S=S,
            B=B,
            C=C,
            dropout=dropout
        )
    
    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)
        output = self.head(features)
        return output
    
    def predict(self, x: Tensor) -> Dict[str, Tensor]:
        """Make predictions and reshape output.
        
        Returns:
            Dict with:
                - 'output': Raw output (N, S*S*(B*5+C))
                - 'reshaped': Reshaped output (N, S, S, B*5+C)
        """
        output = self.forward(x)
        reshaped = self.head.reshape_output(output)
        
        return {
            'output': output,
            'reshaped': reshaped
        }


class YOLOv1Tiny(Module):
    """Tiny version of YOLO v1 for faster inference/testing.
    
    Uses fewer layers in the backbone for faster computation.
    Useful for testing and debugging.
    
    Args:
        input_size: Input image size (default: 224)
        S: Grid size (default: 7)
        B: Number of bounding boxes per grid cell (default: 2)
        C: Number of classes (default: 20)
    """
    
    def __init__(
        self,
        input_size: int = 224,
        S: int = 7,
        B: int = 2,
        C: int = 20
    ) -> None:
        super().__init__()
        
        self.input_size = input_size
        self.S = S
        self.B = B
        self.C = C
        
        self.features = Sequential(
            ConvBlock(3, 16, kernel_size=3, stride=1, padding=1),
            MaxPool2d(kernel_size=2, stride=2),
            
            ConvBlock(16, 32, kernel_size=3, stride=1, padding=1),
            MaxPool2d(kernel_size=2, stride=2),
            
            ConvBlock(32, 64, kernel_size=3, stride=1, padding=1),
            MaxPool2d(kernel_size=2, stride=2),
            
            ConvBlock(64, 128, kernel_size=3, stride=1, padding=1),
            MaxPool2d(kernel_size=2, stride=2),
            
            ConvBlock(128, 256, kernel_size=3, stride=1, padding=1),
            MaxPool2d(kernel_size=2, stride=2),
            
            ConvBlock(256, 512, kernel_size=3, stride=1, padding=1),
            
            ConvBlock(512, 1024, kernel_size=3, stride=1, padding=1),
            ConvBlock(1024, 1024, kernel_size=3, stride=1, padding=1),
        )
        
        self.flatten = Flatten()
        self.fc1 = Linear(1024 * S * S, 512)
        self.activation = LeakyReLU(negative_slope=0.1)
        self.fc2 = Linear(512, S * S * (B * 5 + C))
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


def build_yolov1(
    model_type: str = 'full',
    input_size: int = 448,
    S: int = 7,
    B: int = 2,
    C: int = 20
) -> Union[YOLOv1, YOLOv1Tiny]:
    """Build YOLO v1 model.
    
    Args:
        model_type: 'full' or 'tiny'
        input_size: Input image size
        S: Grid size
        B: Number of bounding boxes per grid cell
        C: Number of classes
    
    Returns:
        YOLOv1 or YOLOv1Tiny model
    """
    if model_type == 'tiny':
        return YOLOv1Tiny(input_size=input_size, S=S, B=B, C=C)
    else:
        return YOLOv1(input_size=input_size, S=S, B=B, C=C)
