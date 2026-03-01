"""
YOLO v11 - Ultralytics Latest

YOLO v11 (2024) features:
1. Enhanced architecture with improved efficiency
2. C3k2 modules (faster C3k)
3. Improved feature extraction
4. State-of-the-art accuracy-speed tradeoff

Reference: https://github.com/ultralytics/ultralytics
"""

from nanotorch.detection.yolo_v11.yolo_v11_model import (
    ConvBN, Bottleneck, C3k2, SPPF, Backbone, Neck, DetectHead, YOLOv11, build_yolov11
)
from nanotorch.detection.yolo_v11.yolo_v11_loss import (
    YOLOv11Loss, YOLOv11LossSimple, encode_targets_v11, decode_predictions_v11
)

__all__ = [
    'ConvBN', 'Bottleneck', 'C3k2', 'SPPF', 'Backbone', 'Neck', 'DetectHead', 'YOLOv11', 'build_yolov11',
    'YOLOv11Loss', 'YOLOv11LossSimple', 'encode_targets_v11', 'decode_predictions_v11'
]
