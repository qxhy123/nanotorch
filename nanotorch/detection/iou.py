"""
IoU (Intersection over Union) variants for object detection.

This module provides various IoU implementations:
- IoU: Standard Intersection over Union
- GIoU: Generalized IoU (handles non-overlapping boxes)
- DIoU: Distance IoU (considers center distance)
- CIoU: Complete IoU (adds aspect ratio consistency)

Reference:
- GIoU: https://arxiv.org/abs/1902.09630
- DIoU/CIoU: https://arxiv.org/abs/1911.08287
"""

import numpy as np
from typing import Union
from nanotorch.tensor import Tensor
from nanotorch.detection.bbox import (
    _to_numpy, box_area, box_intersection
)


def iou(boxes1: Union[Tensor, np.ndarray],
        boxes2: Union[Tensor, np.ndarray]) -> np.ndarray:
    """Compute pairwise IoU (Intersection over Union) between two sets of boxes.
    
    IoU = Intersection / Union
    
    Args:
        boxes1: (N, 4) array in [x1, y1, x2, y2] format
        boxes2: (M, 4) array in [x1, y1, x2, y2] format
    
    Returns:
        (N, M) array of IoU values in range [0, 1]
    """
    boxes1 = _to_numpy(boxes1)
    boxes2 = _to_numpy(boxes2)
    
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    
    inter = box_intersection(boxes1, boxes2)
    union = area1[:, np.newaxis] + area2[np.newaxis, :] - inter
    
    return inter / np.clip(union, 1e-7, None)


def giou(boxes1: Union[Tensor, np.ndarray],
         boxes2: Union[Tensor, np.ndarray]) -> np.ndarray:
    """Compute pairwise GIoU (Generalized Intersection over Union).
    
    GIoU = IoU - (Ac - U) / Ac
    
    Where Ac is the area of the smallest enclosing box, and U is the union area.
    GIoU ranges from [-1, 1], where:
    - 1: Perfect overlap
    - 0: No overlap but boxes are adjacent
    - -1: Boxes are far apart
    
    Args:
        boxes1: (N, 4) array in [x1, y1, x2, y2] format
        boxes2: (M, 4) array in [x1, y1, x2, y2] format
    
    Returns:
        (N, M) array of GIoU values in range [-1, 1]
    """
    boxes1 = _to_numpy(boxes1)
    boxes2 = _to_numpy(boxes2)
    
    # Compute IoU
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    inter = box_intersection(boxes1, boxes2)
    union = area1[:, np.newaxis] + area2[np.newaxis, :] - inter
    iou_val = inter / np.clip(union, 1e-7, None)

    # Compute smallest enclosing box
    boxes1_exp = boxes1[:, np.newaxis, :]  # (N, 1, 4)
    boxes2_exp = boxes2[np.newaxis, :, :]  # (1, M, 4)
    
    # Enclosing box corners
    enclose_x1 = np.minimum(boxes1_exp[..., 0], boxes2_exp[..., 0])
    enclose_y1 = np.minimum(boxes1_exp[..., 1], boxes2_exp[..., 1])
    enclose_x2 = np.maximum(boxes1_exp[..., 2], boxes2_exp[..., 2])
    enclose_y2 = np.maximum(boxes1_exp[..., 3], boxes2_exp[..., 3])
    
    # Enclosing box area
    enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
    
    # GIoU
    giou_val = iou_val - (enclose_area - union) / np.clip(enclose_area, 1e-7, None)
    
    return giou_val


def diou(boxes1: Union[Tensor, np.ndarray],
         boxes2: Union[Tensor, np.ndarray]) -> np.ndarray:
    """Compute pairwise DIoU (Distance Intersection over Union).
    
    DIoU = IoU - R_DIoU
    
    Where R_DIoU = (rho^2(boxes1, boxes2)) / (c^2)
    - rho: Euclidean distance between box centers
    - c: Diagonal length of the smallest enclosing box
    
    DIoU considers the distance between box centers, providing better
    gradient for non-overlapping boxes compared to GIoU.
    
    Args:
        boxes1: (N, 4) array in [x1, y1, x2, y2] format
        boxes2: (M, 4) array in [x1, y1, x2, y2] format
    
    Returns:
        (N, M) array of DIoU values in range [-1, 1]
    """
    boxes1 = _to_numpy(boxes1)
    boxes2 = _to_numpy(boxes2)
    
    # Compute IoU
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    inter = box_intersection(boxes1, boxes2)
    union = area1[:, np.newaxis] + area2[np.newaxis, :] - inter
    iou_val = inter / np.clip(union, 1e-7, None)
    
    # Box centers
    cx1 = (boxes1[..., 0] + boxes1[..., 2]) / 2
    cy1 = (boxes1[..., 1] + boxes1[..., 3]) / 2
    cx2 = (boxes2[..., 0] + boxes2[..., 2]) / 2
    cy2 = (boxes2[..., 1] + boxes2[..., 3]) / 2
    
    # Center distance squared (rho^2)
    cx1 = cx1[:, np.newaxis]  # (N, 1)
    cy1 = cy1[:, np.newaxis]  # (N, 1)
    cx2 = cx2[np.newaxis, :]  # (1, M)
    cy2 = cy2[np.newaxis, :]  # (1, M)
    
    center_dist_sq = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    
    # Smallest enclosing box diagonal (c^2)
    boxes1_exp = boxes1[:, np.newaxis, :]  # (N, 1, 4)
    boxes2_exp = boxes2[np.newaxis, :, :]  # (1, M, 4)
    
    enclose_x1 = np.minimum(boxes1_exp[..., 0], boxes2_exp[..., 0])
    enclose_y1 = np.minimum(boxes1_exp[..., 1], boxes2_exp[..., 1])
    enclose_x2 = np.maximum(boxes1_exp[..., 2], boxes2_exp[..., 2])
    enclose_y2 = np.maximum(boxes1_exp[..., 3], boxes2_exp[..., 3])
    
    enclose_diag_sq = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2
    
    # DIoU
    diou_val = iou_val - center_dist_sq / np.clip(enclose_diag_sq, 1e-7, None)
    
    return diou_val


def ciou(boxes1: Union[Tensor, np.ndarray],
         boxes2: Union[Tensor, np.ndarray],
         eps: float = 1e-7) -> np.ndarray:
    """Compute pairwise CIoU (Complete Intersection over Union).
    
    CIoU = IoU - R_DIoU - alpha * v
    
    Where:
    - R_DIoU = rho^2 / c^2 (center distance term from DIoU)
    - v = (4 / pi^2) * (arctan(w2/h2) - arctan(w1/h1))^2 (aspect ratio consistency)
    - alpha = v / (1 - IoU + v) (trade-off parameter)
    
    CIoU adds aspect ratio consistency to DIoU, providing better regression
    for boxes with similar centers but different aspect ratios.
    
    Args:
        boxes1: (N, 4) array in [x1, y1, x2, y2] format
        boxes2: (M, 4) array in [x1, y1, x2, y2] format
        eps: Small constant for numerical stability
    
    Returns:
        (N, M) array of CIoU values in range [-1, 1]
    """
    boxes1 = _to_numpy(boxes1)
    boxes2 = _to_numpy(boxes2)
    
    # Compute IoU
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    inter = box_intersection(boxes1, boxes2)
    union = area1[:, np.newaxis] + area2[np.newaxis, :] - inter
    iou_val = inter / np.clip(union, eps, None)
    
    # Box centers and dimensions
    cx1 = (boxes1[..., 0] + boxes1[..., 2]) / 2
    cy1 = (boxes1[..., 1] + boxes1[..., 3]) / 2
    w1 = boxes1[..., 2] - boxes1[..., 0]
    h1 = boxes1[..., 3] - boxes1[..., 1]
    
    cx2 = (boxes2[..., 0] + boxes2[..., 2]) / 2
    cy2 = (boxes2[..., 1] + boxes2[..., 3]) / 2
    w2 = boxes2[..., 2] - boxes2[..., 0]
    h2 = boxes2[..., 3] - boxes2[..., 1]
    
    # Center distance squared
    cx1 = cx1[:, np.newaxis]
    cy1 = cy1[:, np.newaxis]
    cx2 = cx2[np.newaxis, :]
    cy2 = cy2[np.newaxis, :]
    
    center_dist_sq = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
    
    # Enclosing box diagonal
    boxes1_exp = boxes1[:, np.newaxis, :]
    boxes2_exp = boxes2[np.newaxis, :, :]
    
    enclose_x1 = np.minimum(boxes1_exp[..., 0], boxes2_exp[..., 0])
    enclose_y1 = np.minimum(boxes1_exp[..., 1], boxes2_exp[..., 1])
    enclose_x2 = np.maximum(boxes1_exp[..., 2], boxes2_exp[..., 2])
    enclose_y2 = np.maximum(boxes1_exp[..., 3], boxes2_exp[..., 3])
    
    enclose_diag_sq = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2
    
    # Aspect ratio consistency term (v)
    w1 = w1[:, np.newaxis]
    h1 = h1[:, np.newaxis]
    w2 = w2[np.newaxis, :]
    h2 = h2[np.newaxis, :]
    
    # v = (4 / pi^2) * (arctan(gt_w/gt_h) - arctan(pred_w/pred_h))^2
    v = (4.0 / (np.pi ** 2)) * (np.arctan(w2 / np.clip(h2, eps, None)) - 
                                 np.arctan(w1 / np.clip(h1, eps, None))) ** 2
    
    # alpha = v / (1 - IoU + v)
    alpha = v / np.clip(1 - iou_val + v, eps, None)
    
    # CIoU = IoU - (rho^2 / c^2) - alpha * v
    ciou_val = iou_val - center_dist_sq / np.clip(enclose_diag_sq, eps, None) - alpha * v
    
    return ciou_val


def siou(boxes1: Union[Tensor, np.ndarray],
         boxes2: Union[Tensor, np.ndarray],
         eps: float = 1e-7) -> np.ndarray:
    """Compute pairwise SIoU (SCYLLA-IoU).
    
    SIoU adds angle cost to CIoU, considering the angle between the
    line connecting box centers and the axes.
    
    SIoU = IoU - (theta * R_DIoU + gamma * alpha * v)
    
    Where:
    - theta: Angle cost term
    - gamma: Distance cost weight
    - R_DIoU: Distance penalty from DIoU
    - alpha * v: Aspect ratio consistency from CIoU
    
    Reference: https://arxiv.org/abs/2205.12740
    
    Args:
        boxes1: (N, 4) array in [x1, y1, x2, y2] format
        boxes2: (M, 4) array in [x1, y1, x2, y2] format
        eps: Small constant for numerical stability
    
    Returns:
        (N, M) array of SIoU values
    """
    boxes1 = _to_numpy(boxes1)
    boxes2 = _to_numpy(boxes2)
    
    # Compute IoU
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    inter = box_intersection(boxes1, boxes2)
    union = area1[:, np.newaxis] + area2[np.newaxis, :] - inter
    iou_val = inter / np.clip(union, eps, None)
    
    # Box centers
    cx1 = (boxes1[..., 0] + boxes1[..., 2]) / 2
    cy1 = (boxes1[..., 1] + boxes1[..., 3]) / 2
    w1 = boxes1[..., 2] - boxes1[..., 0]
    h1 = boxes1[..., 3] - boxes1[..., 1]
    
    cx2 = (boxes2[..., 0] + boxes2[..., 2]) / 2
    cy2 = (boxes2[..., 1] + boxes2[..., 3]) / 2
    w2 = boxes2[..., 2] - boxes2[..., 0]
    h2 = boxes2[..., 3] - boxes2[..., 1]
    
    # Expand for pairwise computation
    cx1 = cx1[:, np.newaxis]
    cy1 = cy1[:, np.newaxis]
    w1 = w1[:, np.newaxis]
    h1 = h1[:, np.newaxis]
    
    cx2 = cx2[np.newaxis, :]
    cy2 = cy2[np.newaxis, :]
    w2 = w2[np.newaxis, :]
    h2 = h2[np.newaxis, :]
    
    # Center offset
    offset_x = np.abs(cx1 - cx2)
    offset_y = np.abs(cy1 - cy2)
    
    # Enclosing box
    boxes1_exp = boxes1[:, np.newaxis, :]
    boxes2_exp = boxes2[np.newaxis, :, :]
    
    enclose_x1 = np.minimum(boxes1_exp[..., 0], boxes2_exp[..., 0])
    enclose_y1 = np.minimum(boxes1_exp[..., 1], boxes2_exp[..., 1])
    enclose_x2 = np.maximum(boxes1_exp[..., 2], boxes2_exp[..., 2])
    enclose_y2 = np.maximum(boxes1_exp[..., 3], boxes2_exp[..., 3])
    
    enclose_w = enclose_x2 - enclose_x1
    enclose_h = enclose_y2 - enclose_y1
    
    # Angle cost
    sigma = offset_x ** 2 + offset_y ** 2  # center dist squared
    sin_alpha = offset_y / np.sqrt(sigma + eps)
    sin_beta = offset_x / np.sqrt(sigma + eps)
    
    threshold = np.power(2, 0.5) / 2
    sin_alpha = np.where(sin_alpha < threshold, sin_alpha, sin_beta)
    angle_cost = np.cos(np.arcsin(sin_alpha) * 2 - np.pi / 2)
    
    # Distance cost
    gamma = 2 - angle_cost
    rho_x = (offset_x / enclose_w) ** 2
    rho_y = (offset_y / enclose_h) ** 2
    distance_cost = gamma * rho_x + gamma * rho_y

    # Shape cost (aspect ratio)
    omega_w = np.abs(w1 - w2) / np.clip(np.maximum(w1, w2), eps, None)
    omega_h = np.abs(h1 - h2) / np.clip(np.maximum(h1, h2), eps, None)
    shape_cost = (1 - iou_val) * np.exp(-omega_w) + (1 - iou_val) * np.exp(-omega_h)
    
    # SIoU
    siou_val = iou_val - 0.5 * (distance_cost + shape_cost)
    
    return siou_val


def compute_iou_loss(pred_boxes: Union[Tensor, np.ndarray],
                     target_boxes: Union[Tensor, np.ndarray],
                     iou_type: str = "ciou",
                     reduction: str = "mean") -> Union[float, np.ndarray]:
    """Compute IoU-based regression loss.
    
    Loss = 1 - IoU (where IoU is the selected variant)
    
    Args:
        pred_boxes: (N, 4) predicted boxes in [x1, y1, x2, y2] format
        target_boxes: (N, 4) target boxes in [x1, y1, x2, y2] format
        iou_type: Type of IoU to use ("iou", "giou", "diou", "ciou", "siou")
        reduction: "mean", "sum", or "none"
    
    Returns:
        Loss value(s). If reduction="none", returns (N,) array.
    """
    pred_boxes = _to_numpy(pred_boxes)
    target_boxes = _to_numpy(target_boxes)
    
    # Select IoU function
    iou_funcs = {
        "iou": iou,
        "giou": giou,
        "diou": diou,
        "ciou": ciou,
        "siou": siou
    }
    
    if iou_type not in iou_funcs:
        raise ValueError(f"Unknown IoU type: {iou_type}. "
                        f"Supported: {list(iou_funcs.keys())}")
    
    iou_func = iou_funcs[iou_type]
    
    # Compute pairwise IoU and extract diagonal (matching pairs)
    # pred_boxes[i] vs target_boxes[i]
    n = pred_boxes.shape[0]
    
    # For efficiency, compute element-wise instead of full pairwise matrix
    iou_vals = np.zeros(n)
    for i in range(n):
        iou_matrix = iou_func(pred_boxes[i:i+1], target_boxes[i:i+1])
        iou_vals[i] = iou_matrix[0, 0]
    
    loss = 1.0 - iou_vals
    
    if reduction == "mean":
        return float(np.mean(loss))
    elif reduction == "sum":
        return float(np.sum(loss))
    else:
        return loss


# Vectorized version for efficiency
def compute_iou_loss_vectorized(pred_boxes: Union[Tensor, np.ndarray],
                                target_boxes: Union[Tensor, np.ndarray],
                                iou_type: str = "ciou",
                                eps: float = 1e-7) -> np.ndarray:
    """Vectorized IoU loss computation for matching box pairs.
    
    This is an optimized version that computes IoU for pred[i] vs target[i]
    without computing the full pairwise matrix.
    
    Args:
        pred_boxes: (N, 4) predicted boxes in [x1, y1, x2, y2] format
        target_boxes: (N, 4) target boxes in [x1, y1, x2, y2] format
        iou_type: Type of IoU to use ("iou", "giou", "diou", "ciou")
        eps: Small constant for numerical stability
    
    Returns:
        (N,) array of losses = 1 - IoU
    """
    pred_boxes = _to_numpy(pred_boxes).astype(np.float32)
    target_boxes = _to_numpy(target_boxes).astype(np.float32)
    
    # Extract coordinates
    px1, py1, px2, py2 = pred_boxes[..., 0], pred_boxes[..., 1], pred_boxes[..., 2], pred_boxes[..., 3]
    tx1, ty1, tx2, ty2 = target_boxes[..., 0], target_boxes[..., 1], target_boxes[..., 2], target_boxes[..., 3]
    
    # Areas
    pred_area = (px2 - px1) * (py2 - py1)
    target_area = (tx2 - tx1) * (ty2 - ty1)
    
    # Intersection
    inter_x1 = np.maximum(px1, tx1)
    inter_y1 = np.maximum(py1, ty1)
    inter_x2 = np.minimum(px2, tx2)
    inter_y2 = np.minimum(py2, ty2)
    
    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter_area = inter_w * inter_h
    
    # Union
    union_area = pred_area + target_area - inter_area
    
    # IoU
    iou_val = inter_area / np.clip(union_area, eps, None)
    
    if iou_type == "iou":
        return 1.0 - iou_val
    
    # Enclosing box
    enclose_x1 = np.minimum(px1, tx1)
    enclose_y1 = np.minimum(py1, ty1)
    enclose_x2 = np.maximum(px2, tx2)
    enclose_y2 = np.maximum(py2, ty2)
    
    if iou_type == "giou":
        enclose_area = (enclose_x2 - enclose_x1) * (enclose_y2 - enclose_y1)
        giou_val = iou_val - (enclose_area - union_area) / np.clip(enclose_area, eps, None)
        return 1.0 - giou_val
    
    # Center distance
    pcx = (px1 + px2) / 2
    pcy = (py1 + py2) / 2
    tcx = (tx1 + tx2) / 2
    tcy = (ty1 + ty2) / 2
    
    center_dist_sq = (pcx - tcx) ** 2 + (pcy - tcy) ** 2
    enclose_diag_sq = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2
    
    if iou_type == "diou":
        diou_val = iou_val - center_dist_sq / np.clip(enclose_diag_sq, eps, None)
        return 1.0 - diou_val
    
    if iou_type == "ciou":
        # Aspect ratio consistency
        pw = px2 - px1
        ph = py2 - py1
        tw = tx2 - tx1
        th = ty2 - ty1
        
        v = (4.0 / (np.pi ** 2)) * (np.arctan(tw / np.clip(th, eps, None)) - 
                                      np.arctan(pw / np.clip(ph, eps, None))) ** 2
        alpha = v / np.clip(1 - iou_val + v, eps, None)
        
        ciou_val = iou_val - center_dist_sq / np.clip(enclose_diag_sq, eps, None) - alpha * v
        return 1.0 - ciou_val
    
    raise ValueError(f"Unknown IoU type: {iou_type}")
