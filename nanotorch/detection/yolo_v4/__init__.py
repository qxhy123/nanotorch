"""
YOLO v4 - Optimal Speed and Accuracy of Object Detection

YOLO v4 improvements over v3:
- CSPDarknet53 backbone with Cross Stage Partial connections
- SPP (Spatial Pyramid Pooling) module
- PANet (Path Aggregation Network) for better feature fusion
- Mish activation function
- CIoU loss for bounding box regression
- Bag of Freebies (BoF) and Bag of Specials (BoS)

Reference:
    "YOLOv4: Optimal Speed and Accuracy of Object Detection"
    Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao
    2020
    https://arxiv.org/abs/2004.10934
"""

from nanotorch.detection.yolo_v4.yolo_v4_model import (
    # Activation
    Mish,
    
    # Basic blocks
    ConvBNMish,
    ConvBNLeaky,
    ResBlock,
    CSPResBlock,
    SPP,
    
    # Backbone
    CSPDarknet53,
    
    # Neck
    PANet,
    
    # Head
    YOLOHead,
    
    # Complete models
    YOLOv4,
    YOLOv4Tiny,
    build_yolov4,
)

from nanotorch.detection.yolo_v4.yolo_v4_loss import (
    YOLOv4Loss,
    YOLOv4LossSimple,
    encode_targets_v4,
    decode_predictions_v4,
)


__all__ = [
    # Activation
    'Mish',
    
    # Basic blocks
    'ConvBNMish',
    'ConvBNLeaky',
    'ResBlock',
    'CSPResBlock',
    'SPP',
    
    # Backbone
    'CSPDarknet53',
    
    # Neck
    'PANet',
    
    # Head
    'YOLOHead',
    
    # Models
    'YOLOv4',
    'YOLOv4Tiny',
    'build_yolov4',
    
    # Loss
    'YOLOv4Loss',
    'YOLOv4LossSimple',
    
    # Encode/Decode
    'encode_targets_v4',
    'decode_predictions_v4',
]
