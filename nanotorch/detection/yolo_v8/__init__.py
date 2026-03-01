"""
YOLO v8 - Ultralytics

YOLO v8 features:
1. C2f modules (faster CSP)
2. Anchor-free detection
3. Decoupled head
4. Mosaic augmentation

Reference: https://github.com/ultralytics/ultralytics
"""

from nanotorch.detection.yolo_v8.yolo_v8_model import (
    ConvBN, Bottleneck, C2f, Backbone, Neck, DetectHead, YOLOv8, build_yolov8
)
from nanotorch.detection.yolo_v8.yolo_v8_loss import (
    YOLOv8Loss, YOLOv8LossSimple, encode_targets_v8, decode_predictions_v8
)

__all__ = [
    'ConvBN', 'Bottleneck', 'C2f', 'Backbone', 'Neck', 'DetectHead', 'YOLOv8', 'build_yolov8',
    'YOLOv8Loss', 'YOLOv8LossSimple', 'encode_targets_v8', 'decode_predictions_v8'
]
