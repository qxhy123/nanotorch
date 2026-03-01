"""
YOLO v1 (You Only Look Once) - Original Implementation

This module provides the original YOLO v1 object detection implementation:
- Darknet backbone network
- YOLO v1 detection head
- YOLO v1 loss function
- Target encoding/decoding utilities

Reference:
    "You Only Look Once: Unified, Real-Time Object Detection"
    Joseph Redmon et al., CVPR 2016
    https://arxiv.org/abs/1506.02640

Basic Usage:
    from nanotorch.detection.yolo_v1 import YOLOv1, YOLOv1Loss, build_yolov1
    
    # Build model
    model = build_yolov1('tiny', input_size=224, S=7, B=2, C=20)
    
    # Forward pass
    output = model(images)
    
    # Compute loss
    loss_fn = YOLOv1Loss(S=7, B=2, C=20)
    loss, loss_dict = loss_fn(output, targets)
"""

from nanotorch.detection.yolo_v1.yolo_v1_model import (
    ConvBlock,
    Darknet,
    YOLOv1Head,
    YOLOv1,
    YOLOv1Tiny,
    build_yolov1
)

from nanotorch.detection.yolo_v1.yolo_v1_loss import (
    YOLOv1Loss,
    YOLOv1LossSimple,
    encode_targets,
    decode_predictions
)

__all__ = [
    "ConvBlock",
    "Darknet",
    "YOLOv1Head",
    "YOLOv1",
    "YOLOv1Tiny",
    "build_yolov1",
    "YOLOv1Loss",
    "YOLOv1LossSimple",
    "encode_targets",
    "decode_predictions",
]
