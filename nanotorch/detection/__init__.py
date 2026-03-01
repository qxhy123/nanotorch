
"""
YOLO v12 Object Detection Module for nanotorch.

This module provides a complete object detection implementation including:
- Bounding box utilities (bbox.py)
- IoU variants (iou.py)
- Non-Maximum Suppression (nms.py)
- YOLO layers (layers.py)
- Backbone network (yolo_backbone.py)
- Feature fusion neck (yolo_neck.py)
- Detection head (yolo_head.py)
- Loss functions (losses.py)
"""

# Bounding box utilities
from nanotorch.detection.bbox import (
    xyxy_to_xywh,
    xywh_to_xyxy,
    xyxy_to_cxcywh,
    cxcywh_to_xyxy,
    normalize_boxes,
    denormalize_boxes,
    box_area,
    box_intersection,
    box_iou,
    clip_boxes,
    scale_boxes,
    encode_boxes,
    decode_boxes,
    generate_anchors,
    generate_anchors_for_grid
)

# IoU variants
from nanotorch.detection.iou import (
    iou,
    giou,
    diou,
    ciou,
    siou,
    compute_iou_loss,
    compute_iou_loss_vectorized
)

# NMS
from nanotorch.detection.nms import (
    nms,
    batched_nms,
    soft_nms,
    nms_rotated,
    postprocess_detections,
    batch_postprocess_detections
)

# YOLO layers
from nanotorch.detection.layers import (
    Conv,
    DWConv,
    Bottleneck,
    C2f,
    SPPF,
    Focus,
    Concat,
    Upsample,
    AreaAttention,
    A2Block,
    DetectLayer,
    make_divisible
)

# Backbone
from nanotorch.detection.yolo_backbone import (
    Stem,
    Downsample,
    RELANBlock,
    A2RELANBlock,
    YOLOBackbone,
    YOLOBackboneTiny,
    build_backbone
)

# Neck
from nanotorch.detection.yolo_neck import (
    FPN,
    PANet,
    BiFPN,
    BiFPNBlock,
    YOLONeck,
    build_neck
)

# Detection head
from nanotorch.detection.yolo_head import (
    DFLHead,
    ClassHead,
    DecoupledHead,
    YOLOHead,
    AnchorHead,
    build_head
)

# Losses
from nanotorch.detection.losses import (
    CIoULoss,
    DFLoss,
    VarifocalLoss,
    BCELoss,
    YOLOLoss,
    SimpleYOLOLoss
)

__all__ = [
    # Bounding box
    "xyxy_to_xywh",
    "xywh_to_xyxy",
    "xyxy_to_cxcywh",
    "cxcywh_to_xyxy",
    "normalize_boxes",
    "denormalize_boxes",
    "box_area",
    "box_intersection",
    "box_iou",
    "clip_boxes",
    "scale_boxes",
    "encode_boxes",
    "decode_boxes",
    "generate_anchors",
    "generate_anchors_for_grid",
    # IoU
    "iou",
    "giou",
    "diou",
    "ciou",
    "siou",
    "compute_iou_loss",
    "compute_iou_loss_vectorized",
    # NMS
    "nms",
    "batched_nms",
    "soft_nms",
    "nms_rotated",
    "postprocess_detections",
    "batch_postprocess_detections",
    # Layers
    "Conv",
    "DWConv",
    "Bottleneck",
    "C2f",
    "SPPF",
    "Focus",
    "Concat",
    "Upsample",
    "AreaAttention",
    "A2Block",
    "DetectLayer",
    "make_divisible",
    # Backbone
    "Stem",
    "Downsample",
    "RELANBlock",
    "A2RELANBlock",
    "YOLOBackbone",
    "YOLOBackboneTiny",
    "build_backbone",
    # Neck
    "FPN",
    "PANet",
    "BiFPN",
    "BiFPNBlock",
    "YOLONeck",
    "build_neck",
    # Head
    "DFLHead",
    "ClassHead",
    "DecoupledHead",
    "YOLOHead",
    "AnchorHead",
    "build_head",
    # Losses
    "CIoULoss",
    "DFLoss",
    "VarifocalLoss",
    "BCELoss",
    "YOLOLoss",
    "SimpleYOLOLoss",
]
