"""
YOLO v10 - NMS-Free

YOLO v10 features:
1. Consistent dual assignments (training/inference)
2. NMS-free inference
3. Efficiency-accuracy driven model design

Reference: https://github.com/THU-MIG/yolov10
"""

from nanotorch.detection.yolo_v10.yolo_v10_model import (
    ConvBN, SCDown, C2fCIB, Backbone, Head, YOLOv10, build_yolov10
)
from nanotorch.detection.yolo_v10.yolo_v10_loss import (
    YOLOv10Loss, YOLOv10LossSimple, encode_targets_v10, decode_predictions_v10
)

__all__ = [
    'ConvBN', 'SCDown', 'C2fCIB', 'Backbone', 'Head', 'YOLOv10', 'build_yolov10',
    'YOLOv10Loss', 'YOLOv10LossSimple', 'encode_targets_v10', 'decode_predictions_v10'
]
