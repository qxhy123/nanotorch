"""
YOLO v3 Object Detection Module

This module provides a complete implementation of YOLO v3 for object detection.
"""

from nanotorch.detection.yolo_v3.yolo_v3_model import (
    ConvBN,
    ResidualBlock,
    Darknet53,
    FPN,
    YOLOHead,
    YOLOv3,
    YOLOv3Tiny,
    build_yolov3
)

from nanotorch.detection.yolo_v3.yolo_v3_loss import (
    YOLOv3Loss,
    YOLOv3LossSimple,
    encode_targets_v3,
    decode_predictions_v3
)

__all__ = [
    'ConvBN',
    'ResidualBlock',
    'Darknet53',
    'FPN',
    'YOLOHead',
    'YOLOv3',
    'YOLOv3Tiny',
    'build_yolov3',
    'YOLOv3Loss',
    'YOLOv3LossSimple',
    'encode_targets_v3',
    'decode_predictions_v3'
]
