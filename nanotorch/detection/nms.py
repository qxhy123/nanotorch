"""
Non-Maximum Suppression (NMS) for object detection.

This module provides NMS implementations:
- nms: Standard NMS
- batched_nms: NMS with class indices
- soft_nms: Soft-NMS (soft selection instead of hard thresholding)
"""

import numpy as np
from typing import Union, Tuple
from nanotorch.tensor import Tensor
from nanotorch.detection.bbox import _to_numpy, box_iou


def nms(boxes: Union[Tensor, np.ndarray],
        scores: Union[Tensor, np.ndarray],
        iou_threshold: float) -> np.ndarray:
    """Non-Maximum Suppression.
    
    For each box, suppress all other boxes that have IoU > iou_threshold
    and lower score.
    
    Algorithm:
    1. Sort boxes by score (descending)
    2. Select highest-score box as detection
    3. Remove all boxes with IoU > threshold with selected box
    4. Repeat until no boxes left
    
    Args:
        boxes: (N, 4) array of boxes in [x1, y1, x2, y2] format
        scores: (N,) array of confidence scores
        iou_threshold: IoU threshold for suppression
    
    Returns:
        (K,) array of indices of selected boxes (sorted by score)
    """
    boxes = _to_numpy(boxes)
    scores = _to_numpy(scores)
    
    if boxes.shape[0] == 0:
        return np.array([], dtype=np.int64)
    
    # Sort by score (descending)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.shape[0] > 0:
        # Select highest-score box
        idx = order[0]
        keep.append(idx)
        
        if order.shape[0] == 1:
            break
        
        # Compute IoU with remaining boxes
        remaining_boxes = boxes[order[1:]]
        ious = box_iou(boxes[idx:idx+1], remaining_boxes)[0]
        
        # Keep boxes with IoU <= threshold
        mask = ious <= iou_threshold
        order = order[1:][mask]
    
    return np.array(keep, dtype=np.int64)


def batched_nms(boxes: Union[Tensor, np.ndarray],
                scores: Union[Tensor, np.ndarray],
                class_ids: Union[Tensor, np.ndarray],
                iou_threshold: float) -> np.ndarray:
    """NMS with class indices (per-class NMS).
    
    Applies NMS independently for each class. This ensures that boxes
    from different classes don't suppress each other.
    
    Args:
        boxes: (N, 4) array of boxes in [x1, y1, x2, y2] format
        scores: (N,) array of confidence scores
        class_ids: (N,) array of class indices
        iou_threshold: IoU threshold for suppression
    
    Returns:
        (K,) array of indices of selected boxes
    """
    boxes = _to_numpy(boxes)
    scores = _to_numpy(scores)
    class_ids = _to_numpy(class_ids)
    
    if boxes.shape[0] == 0:
        return np.array([], dtype=np.int64)
    
    # Strategy: Offset boxes by class_id * max_coordinate
    # This ensures boxes from different classes have 0 IoU
    max_coordinate = boxes.max()
    offsets = class_ids.astype(np.float32) * (max_coordinate + 1)
    boxes_for_nms = boxes + offsets[:, np.newaxis]
    
    return nms(boxes_for_nms, scores, iou_threshold)


def soft_nms(boxes: Union[Tensor, np.ndarray],
             scores: Union[Tensor, np.ndarray],
             iou_threshold: float,
             score_threshold: float = 0.001,
             sigma: float = 0.5,
             method: str = "gaussian") -> Tuple[np.ndarray, np.ndarray]:
    """Soft-NMS: Soft Non-Maximum Suppression.
    
    Instead of hard suppression (removing boxes), Soft-NMS decays
    the scores of overlapping boxes based on IoU overlap.
    
    Methods:
    - "linear": score *= (1 - iou) if iou > threshold else score
    - "gaussian": score *= exp(-iou^2 / sigma)
    
    Reference: https://arxiv.org/abs/1704.04503
    
    Args:
        boxes: (N, 4) array of boxes in [x1, y1, x2, y2] format
        scores: (N,) array of confidence scores
        iou_threshold: IoU threshold (for linear method)
        score_threshold: Minimum score to keep
        sigma: Gaussian sigma (for gaussian method)
        method: "linear" or "gaussian"
    
    Returns:
        keep_indices: (K,) array of kept box indices
        keep_scores: (K,) array of adjusted scores
    """
    boxes = _to_numpy(boxes).astype(np.float32)
    scores = _to_numpy(scores).astype(np.float32).copy()
    
    n = boxes.shape[0]
    if n == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
    
    keep_indices = []
    keep_scores = []
    
    indices = np.arange(n)
    
    for _ in range(n):
        # Find box with max score
        max_idx = scores.argmax()
        max_score = scores[max_idx]
        
        if max_score < score_threshold:
            break
        
        # Keep this box
        keep_indices.append(indices[max_idx])
        keep_scores.append(max_score)
        
        # Compute IoU with all other boxes
        max_box = boxes[max_idx:max_idx+1]
        ious = box_iou(max_box, boxes)[0]
        
        # Update scores based on IoU
        if method == "linear":
            # Linear decay: if IoU > threshold, decay score by (1 - IoU)
            # Otherwise, keep original score
            suppress_mask = ious > iou_threshold
            scores = np.where(suppress_mask, scores * (1 - ious), scores)
        elif method == "gaussian":
            # Gaussian decay
            scores = scores * np.exp(-(ious ** 2) / sigma)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'linear' or 'gaussian'")
        
        # Set max score to -inf so it won't be selected again
        scores[max_idx] = -1.0
    
    return np.array(keep_indices, dtype=np.int64), np.array(keep_scores, dtype=np.float32)


def nms_rotated(boxes: Union[Tensor, np.ndarray],
                scores: Union[Tensor, np.ndarray],
                iou_threshold: float) -> np.ndarray:
    """NMS for rotated (oriented) bounding boxes.
    
    Rotated boxes are in [cx, cy, w, h, angle] format where angle is
    in radians (counter-clockwise from positive x-axis).
    
    Note: This is a simplified implementation using axis-aligned approximation
    for IoU computation. For accurate rotated IoU, use a more sophisticated
    implementation.
    
    Args:
        boxes: (N, 5) array of rotated boxes [cx, cy, w, h, angle]
        scores: (N,) array of confidence scores
        iou_threshold: IoU threshold for suppression
    
    Returns:
        (K,) array of indices of selected boxes
    """
    boxes = _to_numpy(boxes)
    scores = _to_numpy(scores)
    
    if boxes.shape[0] == 0:
        return np.array([], dtype=np.int64)
    
    # Convert rotated boxes to axis-aligned (outer rectangle)
    # This is an approximation - for accurate rotated IoU,
    # you need polygon intersection computation
    
    cx, cy, w, h, angle = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3], boxes[..., 4]
    
    # Compute corner points of rotated rectangle
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    # Corner offsets (relative to center)
    dx = w / 2
    dy = h / 2
    
    # Four corners
    corners = np.zeros((boxes.shape[0], 4, 2))
    
    # Rotated corner coordinates
    corners[:, 0, 0] = cx + dx * cos_a - dy * sin_a  # top-right
    corners[:, 0, 1] = cy + dx * sin_a + dy * cos_a
    corners[:, 1, 0] = cx - dx * cos_a - dy * sin_a  # top-left
    corners[:, 1, 1] = cy - dx * sin_a + dy * cos_a
    corners[:, 2, 0] = cx - dx * cos_a + dy * sin_a  # bottom-left
    corners[:, 2, 1] = cy - dx * sin_a - dy * cos_a
    corners[:, 3, 0] = cx + dx * cos_a + dy * sin_a  # bottom-right
    corners[:, 3, 1] = cy + dx * sin_a - dy * cos_a
    
    # Axis-aligned bounding box (approximation)
    x1 = corners[..., 0].min(axis=1)
    y1 = corners[..., 1].min(axis=1)
    x2 = corners[..., 0].max(axis=1)
    y2 = corners[..., 1].max(axis=1)
    
    aabb_boxes = np.stack([x1, y1, x2, y2], axis=1)
    
    # Apply standard NMS
    return nms(aabb_boxes, scores, iou_threshold)


def postprocess_detections(predictions: np.ndarray,
                           conf_threshold: float = 0.25,
                           iou_threshold: float = 0.45,
                           max_detections: int = 300,
                           multi_label: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Post-process raw YOLO predictions.
    
    Takes raw prediction output and applies:
    1. Confidence filtering
    2. Class score computation
    3. Non-maximum suppression
    
    Args:
        predictions: (N, 4 + num_classes) array where each row is
                     [x1, y1, x2, y2, class_score_0, class_score_1, ...]
        conf_threshold: Minimum confidence threshold
        iou_threshold: IoU threshold for NMS
        max_detections: Maximum number of detections to return
        multi_label: If True, use class score for NMS; if False, use objectness
    
    Returns:
        boxes: (K, 4) array of final boxes
        scores: (K,) array of confidence scores
        class_ids: (K,) array of class indices
    """
    predictions = _to_numpy(predictions)
    
    if predictions.shape[0] == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.int64)
    
    # Split into boxes and class scores
    boxes = predictions[:, :4]
    class_scores = predictions[:, 4:]
    
    if multi_label:
        # Multi-label: each box can belong to multiple classes
        # Use max class score for filtering
        scores = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1)
    else:
        # Single-label: objectness * class_score
        # Assume first element is objectness, rest are class scores
        obj_conf = predictions[:, 4]
        class_scores = predictions[:, 5:]
        scores = obj_conf * class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1)
    
    # Filter by confidence
    mask = scores >= conf_threshold
    boxes = boxes[mask]
    scores = scores[mask]
    class_ids = class_ids[mask]
    
    if boxes.shape[0] == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32), np.array([], dtype=np.int64)
    
    # Apply batched NMS
    keep = batched_nms(boxes, scores, class_ids, iou_threshold)
    
    # Limit to max detections
    if len(keep) > max_detections:
        # Keep top-k by score
        keep_scores = scores[keep]
        top_k_idx = np.argsort(keep_scores)[::-1][:max_detections]
        keep = keep[top_k_idx]
    
    return boxes[keep], scores[keep], class_ids[keep]


def batch_postprocess_detections(batch_predictions: np.ndarray,
                                 conf_threshold: float = 0.25,
                                 iou_threshold: float = 0.45,
                                 max_detections: int = 300) -> list:
    """Post-process batch of YOLO predictions.
    
    Args:
        batch_predictions: (B, N, 4 + num_classes) array of batch predictions
        conf_threshold: Minimum confidence threshold
        iou_threshold: IoU threshold for NMS
        max_detections: Maximum number of detections per image
    
    Returns:
        List of (boxes, scores, class_ids) tuples for each image in batch
    """
    batch_predictions = _to_numpy(batch_predictions)
    batch_size = batch_predictions.shape[0]
    
    results = []
    for i in range(batch_size):
        boxes, scores, class_ids = postprocess_detections(
            batch_predictions[i],
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            max_detections=max_detections
        )
        results.append((boxes, scores, class_ids))
    
    return results
