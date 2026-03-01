"""
YOLO v5 - Ultralytics YOLO

YOLO v5 introduced by Ultralytics with:
- C3 modules
- SPPF (Spatial Pyramid Pooling Fast)
- SiLU activation
- Auto anchor calculation
- Mosaic data augmentation

Reference:
    https://github.com/ultralytics/yolov5
"""

from nanotorch.detection.yolo_v5.yolo_v5_model import (
    ConvBN,
    Bottleneck,
    C3,
    SPPF,
    Backbone,
    Neck,
    DetectHead,
    YOLOv5,
    YOLOv5Nano,
    YOLOv5Small,
    build_yolov5,
)

from nanotorch.detection.yolo_v5.yolo_v5_loss import (
    YOLOv5Loss,
    YOLOv5LossSimple,
    encode_targets_v5,
    decode_predictions_v5,
)


__all__ = [
    'ConvBN',
    'Bottleneck',
    'C3',
    'SPPF',
    'Backbone',
    'Neck',
    'DetectHead',
    'YOLOv5',
    'YOLOv5Nano',
    'YOLOv5Small',
    'build_yolov5',
    'YOLOv5Loss',
    'YOLOv5LossSimple',
    'encode_targets_v5',
    'decode_predictions_v5',
]
