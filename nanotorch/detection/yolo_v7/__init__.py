"""
YOLO v7 - Trainable Bag-of-Freebies

YOLO v7 introduced:
1. E-ELAN (Extended Efficient Layer Aggregation Network)
2. Model scaling techniques
3. Auxiliary training heads

Reference: https://github.com/WongKinYiu/yolov7
"""

from nanotorch.detection.yolo_v7.yolo_v7_model import (
    ConvBN, ELAN, Backbone, Neck, Head, YOLOv7, build_yolov7
)
from nanotorch.detection.yolo_v7.yolo_v7_loss import (
    YOLOv7Loss, YOLOv7LossSimple, encode_targets_v7, decode_predictions_v7
)

__all__ = [
    'ConvBN', 'ELAN', 'Backbone', 'Neck', 'Head', 'YOLOv7', 'build_yolov7',
    'YOLOv7Loss', 'YOLOv7LossSimple', 'encode_targets_v7', 'decode_predictions_v7'
]
