"""
Bounding box utilities for object detection.

This module provides functions for converting between different bounding box formats,
encoding/decoding boxes for anchor-based detection, and other bbox operations.
"""

import numpy as np
from typing import Union, Tuple
from nanotorch.tensor import Tensor


def _to_numpy(data: Union[Tensor, np.ndarray]) -> np.ndarray:
    """Convert input to numpy array."""
    if isinstance(data, Tensor):
        return data.data
    return np.asarray(data)


def xyxy_to_xywh(boxes: Union[Tensor, np.ndarray]) -> np.ndarray:
    """Convert boxes from [x1, y1, x2, y2] to [x_center, y_center, width, height].
    
    Args:
        boxes: (N, 4) array of boxes in [x1, y1, x2, y2] format
    
    Returns:
        (N, 4) array of boxes in [cx, cy, w, h] format
    """
    boxes = _to_numpy(boxes)
    x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return np.stack([cx, cy, w, h], axis=-1)


def xywh_to_xyxy(boxes: Union[Tensor, np.ndarray]) -> np.ndarray:
    """Convert boxes from [x_center, y_center, width, height] to [x1, y1, x2, y2].
    
    Args:
        boxes: (N, 4) array of boxes in [cx, cy, w, h] format
    
    Returns:
        (N, 4) array of boxes in [x1, y1, x2, y2] format
    """
    boxes = _to_numpy(boxes)
    cx, cy, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2
    return np.stack([x1, y1, x2, y2], axis=-1)


def xyxy_to_cxcywh(boxes: Union[Tensor, np.ndarray]) -> np.ndarray:
    """Convert boxes from [x1, y1, x2, y2] to [cx, cy, w, h] (normalized).
    
    This is an alias for xyxy_to_xywh.
    """
    return xyxy_to_xywh(boxes)


def cxcywh_to_xyxy(boxes: Union[Tensor, np.ndarray]) -> np.ndarray:
    """Convert boxes from [cx, cy, w, h] to [x1, y1, x2, y2].
    
    This is an alias for xywh_to_xyxy.
    """
    return xywh_to_xyxy(boxes)


def normalize_boxes(boxes: Union[Tensor, np.ndarray], 
                   img_size: Tuple[int, int]) -> np.ndarray:
    """Normalize boxes to [0, 1] range.
    
    Args:
        boxes: (N, 4) array in [x1, y1, x2, y2] format
        img_size: (width, height) of the image
    
    Returns:
        (N, 4) array with coordinates normalized to [0, 1]
    """
    boxes = _to_numpy(boxes).astype(np.float32)
    w, h = img_size
    boxes[..., 0] /= w  # x1
    boxes[..., 1] /= h  # y1
    boxes[..., 2] /= w  # x2
    boxes[..., 3] /= h  # y2
    return boxes


def denormalize_boxes(boxes: Union[Tensor, np.ndarray],
                      img_size: Tuple[int, int]) -> np.ndarray:
    """Denormalize boxes from [0, 1] to pixel coordinates.
    
    Args:
        boxes: (N, 4) array with normalized coordinates
        img_size: (width, height) of the image
    
    Returns:
        (N, 4) array in pixel coordinates
    """
    boxes = _to_numpy(boxes).astype(np.float32)
    w, h = img_size
    boxes[..., 0] *= w  # x1
    boxes[..., 1] *= h  # y1
    boxes[..., 2] *= w  # x2
    boxes[..., 3] *= h  # y2
    return boxes


def box_area(boxes: Union[Tensor, np.ndarray]) -> np.ndarray:
    """Compute area of boxes.
    
    Args:
        boxes: (N, 4) array in [x1, y1, x2, y2] format
    
    Returns:
        (N,) array of areas
    """
    boxes = _to_numpy(boxes)
    return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])


def box_intersection(boxes1: Union[Tensor, np.ndarray],
                    boxes2: Union[Tensor, np.ndarray]) -> np.ndarray:
    """Compute pairwise intersection areas between two sets of boxes.
    
    Args:
        boxes1: (N, 4) array
        boxes2: (M, 4) array
    
    Returns:
        (N, M) array of intersection areas
    """
    boxes1 = _to_numpy(boxes1)
    boxes2 = _to_numpy(boxes2)
    
    n = boxes1.shape[0]
    m = boxes2.shape[0]
    
    # Expand dimensions for broadcasting
    # boxes1: (N, 1, 4), boxes2: (1, M, 4)
    boxes1 = boxes1[:, np.newaxis, :]
    boxes2 = boxes2[np.newaxis, :, :]
    
    # Compute intersection coordinates
    lt = np.maximum(boxes1[..., :2], boxes2[..., :2])  # left-top
    rb = np.minimum(boxes1[..., 2:], boxes2[..., 2:])  # right-bottom
    
    # Compute intersection area
    wh = np.clip(rb - lt, 0, None)
    return wh[..., 0] * wh[..., 1]


def box_iou(boxes1: Union[Tensor, np.ndarray],
            boxes2: Union[Tensor, np.ndarray]) -> np.ndarray:
    """Compute pairwise IoU between two sets of boxes.
    
    Args:
        boxes1: (N, 4) array in [x1, y1, x2, y2] format
        boxes2: (M, 4) array in [x1, y1, x2, y2] format
    
    Returns:
        (N, M) array of IoU values
    """
    boxes1 = _to_numpy(boxes1)
    boxes2 = _to_numpy(boxes2)
    
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    
    inter = box_intersection(boxes1, boxes2)
    union = area1[:, np.newaxis] + area2[np.newaxis, :] - inter
    
    return inter / np.clip(union, 1e-7, None)


def clip_boxes(boxes: Union[Tensor, np.ndarray],
               img_size: Tuple[int, int]) -> np.ndarray:
    """Clip boxes to image boundaries.
    
    Args:
        boxes: (N, 4) array in [x1, y1, x2, y2] format
        img_size: (width, height) of the image
    
    Returns:
        (N, 4) array of clipped boxes
    """
    boxes = _to_numpy(boxes).astype(np.float32)
    w, h = img_size
    boxes[..., 0] = np.clip(boxes[..., 0], 0, w)
    boxes[..., 1] = np.clip(boxes[..., 1], 0, h)
    boxes[..., 2] = np.clip(boxes[..., 2], 0, w)
    boxes[..., 3] = np.clip(boxes[..., 3], 0, h)
    return boxes


def scale_boxes(boxes: Union[Tensor, np.ndarray],
                scale: float) -> np.ndarray:
    """Scale boxes by a factor.
    
    Args:
        boxes: (N, 4) array in [cx, cy, w, h] format
        scale: Scaling factor
    
    Returns:
        (N, 4) array of scaled boxes
    """
    boxes = _to_numpy(boxes).astype(np.float32)
    boxes[..., 2:] *= scale
    return boxes


def encode_boxes(boxes: Union[Tensor, np.ndarray],
                 anchors: Union[Tensor, np.ndarray]) -> np.ndarray:
    """Encode boxes relative to anchors (for training).
    
    Transforms target boxes into anchor-offset format for training.
    
    Args:
        boxes: (N, 4) target boxes in [cx, cy, w, h] format
        anchors: (N, 4) anchor boxes in [cx, cy, w, h] format
    
    Returns:
        (N, 4) encoded boxes [tx, ty, tw, th] where:
        - tx = (cx - anchor_cx) / anchor_w
        - ty = (cy - anchor_cy) / anchor_h
        - tw = log(w / anchor_w)
        - th = log(h / anchor_h)
    """
    boxes = _to_numpy(boxes).astype(np.float32)
    anchors = _to_numpy(anchors).astype(np.float32)
    
    # Decode anchor
    ax, ay, aw, ah = anchors[..., 0], anchors[..., 1], anchors[..., 2], anchors[..., 3]
    
    # Target box
    tx = (boxes[..., 0] - ax) / aw
    ty = (boxes[..., 1] - ay) / ah
    tw = np.log(boxes[..., 2] / aw + 1e-7)
    th = np.log(boxes[..., 3] / ah + 1e-7)
    
    return np.stack([tx, ty, tw, th], axis=-1)


def decode_boxes(deltas: Union[Tensor, np.ndarray],
                 anchors: Union[Tensor, np.ndarray]) -> np.ndarray:
    """Decode boxes from anchor-offset format.
    
    Args:
        deltas: (N, 4) encoded deltas [tx, ty, tw, th]
        anchors: (N, 4) anchor boxes in [cx, cy, w, h] format
    
    Returns:
        (N, 4) decoded boxes in [cx, cy, w, h] format
    """
    deltas = _to_numpy(deltas).astype(np.float32)
    anchors = _to_numpy(anchors).astype(np.float32)
    
    ax, ay, aw, ah = anchors[..., 0], anchors[..., 1], anchors[..., 2], anchors[..., 3]
    
    cx = deltas[..., 0] * aw + ax
    cy = deltas[..., 1] * ah + ay
    w = np.exp(deltas[..., 2]) * aw
    h = np.exp(deltas[..., 3]) * ah
    
    return np.stack([cx, cy, w, h], axis=-1)


def generate_anchors(sizes: Tuple[int, ...] = (32, 64, 128, 256, 512),
                     ratios: Tuple[float, ...] = (0.5, 1.0, 2.0)) -> np.ndarray:
    """Generate anchor boxes for a single feature map level.
    
    Args:
        sizes: Anchor sizes (scales)
        ratios: Aspect ratios
    
    Returns:
        (len(sizes) * len(ratios), 4) array of anchors in [cx, cy, w, h] format
    """
    anchors = []
    for size in sizes:
        for ratio in ratios:
            w = size * np.sqrt(ratio)
            h = size / np.sqrt(ratio)
            anchors.append([0, 0, w, h])
    return np.array(anchors, dtype=np.float32)


def generate_anchors_for_grid(feature_size: Tuple[int, int],
                               stride: int,
                               sizes: Tuple[int, ...] = (32, 64, 128),
                               ratios: Tuple[float, ...] = (0.5, 1.0, 2.0)) -> np.ndarray:
    """Generate anchors for all positions in a feature map.
    
    Args:
        feature_size: (H, W) of the feature map
        stride: Stride of the feature map relative to input image
        sizes: Anchor sizes
        ratios: Aspect ratios
    
    Returns:
        (H * W * num_anchors, 4) array of anchors
    """
    h, w = feature_size
    
    # Generate base anchors
    base_anchors = generate_anchors(sizes, ratios)
    num_anchors = base_anchors.shape[0]
    
    # Create grid of anchor centers
    shift_x = np.arange(0, w) * stride + stride / 2
    shift_y = np.arange(0, h) * stride + stride / 2
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    
    # Flatten and repeat for each anchor
    shifts = np.stack([
        shift_x.flatten(),
        shift_y.flatten(),
        np.zeros_like(shift_x.flatten()),
        np.zeros_like(shift_y.flatten())
    ], axis=1)
    
    # Broadcast: (H*W, 4) + (num_anchors, 4) -> (H*W*num_anchors, 4)
    shifts = shifts[:, np.newaxis, :]  # (H*W, 1, 4)
    base_anchors = base_anchors[np.newaxis, :, :]  # (1, num_anchors, 4)
    
    anchors = shifts + base_anchors  # (H*W, num_anchors, 4)
    return anchors.reshape(-1, 4)
