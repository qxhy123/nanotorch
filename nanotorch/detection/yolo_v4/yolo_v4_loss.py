"""
YOLO v4 Loss Function

YOLO v4 uses a similar loss function to YOLO v3 with:
1. CIoU loss for bounding box regression (improved over MSE)
2. Binary cross-entropy for objectness
3. Binary cross-entropy for classification (multi-label)

Key improvements:
- CIoU considers overlap, center distance, and aspect ratio
- Focal loss for handling class imbalance
"""

import numpy as np
from typing import Tuple, Dict, List, Optional

from nanotorch.tensor import Tensor
from nanotorch.nn.module import Module


class YOLOv4Loss(Module):
    """YOLO v4 Loss Function.
    
    Combines:
    - CIoU loss for coordinates (improved bounding box regression)
    - BCE loss for objectness
    - BCE loss for classification
    
    Args:
        num_classes: Number of object classes
        anchors: Anchor box sizes [(w1, h1), (w2, h2), ...]
        ignore_threshold: IoU threshold for ignoring predictions
        lambda_coord: Weight for coordinate loss
        lambda_obj: Weight for objectness loss
        lambda_noobj: Weight for no-object loss
        lambda_class: Weight for classification loss
    """
    
    def __init__(
        self,
        num_classes: int = 80,
        anchors: Optional[List[Tuple[int, int]]] = None,
        ignore_threshold: float = 0.5,
        lambda_coord: float = 5.0,
        lambda_obj: float = 1.0,
        lambda_noobj: float = 0.5,
        lambda_class: float = 1.0
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.ignore_threshold = ignore_threshold
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class
        
        if anchors is None:
            # YOLO v4 anchors (optimized from v3)
            self.anchors = [
                (12, 16), (19, 36), (40, 28),
                (36, 75), (76, 55), (72, 146),
                (142, 110), (192, 243), (459, 401)
            ]
        else:
            self.anchors = anchors
    
    def forward(
        self,
        predictions: Dict[str, Tensor],
        targets: List[Dict]
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute YOLO v4 loss.
        
        Args:
            predictions: Dict with 'small', 'medium', 'large' predictions
            targets: List of target dicts with 'boxes', 'labels' keys
        
        Returns:
            total_loss: Total loss value
            loss_dict: Dictionary with individual loss components
        """
        total_coord_loss = 0.0
        total_obj_loss = 0.0
        total_noobj_loss = 0.0
        total_class_loss = 0.0
        
        for scale_name, pred in predictions.items():
            coord_loss, obj_loss, noobj_loss, class_loss = self._compute_scale_loss(
                pred, targets, scale_name
            )
            total_coord_loss += coord_loss
            total_obj_loss += obj_loss
            total_noobj_loss += noobj_loss
            total_class_loss += class_loss
        
        total_loss = (
            self.lambda_coord * total_coord_loss +
            self.lambda_obj * total_obj_loss +
            self.lambda_noobj * total_noobj_loss +
            self.lambda_class * total_class_loss
        )
        
        loss_dict = {
            'coord_loss': total_coord_loss,
            'obj_loss': total_obj_loss,
            'noobj_loss': total_noobj_loss,
            'class_loss': total_class_loss,
            'total_loss': total_loss
        }
        
        return Tensor(total_loss), loss_dict
    
    def _compute_scale_loss(
        self,
        pred: Tensor,
        targets: List[Dict],
        scale_name: str
    ) -> Tuple[float, float, float, float]:
        """Compute loss for a single scale using CIoU."""
        pred_data = pred.data
        N, C, H, W = pred_data.shape
        
        num_anchors = 3
        num_values = 5 + self.num_classes
        
        pred_data = pred_data.reshape(N, num_anchors, num_values, H, W)
        pred_data = pred_data.transpose(0, 3, 4, 1, 2)
        
        coord_loss = 0.0
        obj_loss = 0.0
        noobj_loss = 0.0
        class_loss = 0.0
        
        coord_count = 0
        obj_count = 0
        noobj_count = 0
        class_count = 0
        
        for n in range(N):
            target = targets[n] if n < len(targets) else {'boxes': np.array([]), 'labels': np.array([])}
            boxes = target.get('boxes', np.array([]))
            labels = target.get('labels', np.array([]))
            
            if len(boxes) == 0:
                for i in range(H):
                    for j in range(W):
                        for a in range(num_anchors):
                            conf = pred_data[n, i, j, a, 4]
                            noobj_loss += (conf - 0) ** 2
                            noobj_count += 1
                continue
            
            for i in range(H):
                for j in range(W):
                    for a in range(num_anchors):
                        cell_x = (j + 0.5) / W
                        cell_y = (i + 0.5) / H
                        
                        pred_tx = pred_data[n, i, j, a, 0]
                        pred_ty = pred_data[n, i, j, a, 1]
                        pred_tw = pred_data[n, i, j, a, 2]
                        pred_th = pred_data[n, i, j, a, 3]
                        pred_conf = pred_data[n, i, j, a, 4]
                        
                        anchor_idx = 0 if scale_name == 'large' else (3 if scale_name == 'medium' else 6)
                        anchor_w, anchor_h = self.anchors[anchor_idx + a]
                        
                        best_iou = 0.0
                        best_target_idx = -1
                        
                        for t_idx, box in enumerate(boxes):
                            x1, y1, x2, y2 = box
                            gt_cx = (x1 + x2) / 2
                            gt_cy = (y1 + y2) / 2
                            gt_w = x2 - x1
                            gt_h = y2 - y1
                            
                            iou = self._compute_ciou(
                                cell_x, cell_y, 
                                anchor_w / 416, anchor_h / 416,
                                gt_cx / 416, gt_cy / 416, gt_w / 416, gt_h / 416
                            )
                            
                            if iou > best_iou:
                                best_iou = iou
                                best_target_idx = t_idx
                        
                        if best_iou > self.ignore_threshold and best_target_idx >= 0:
                            box = boxes[best_target_idx]
                            label = labels[best_target_idx]
                            
                            x1, y1, x2, y2 = box
                            gt_cx = (x1 + x2) / 2
                            gt_cy = (y1 + y2) / 2
                            gt_w = x2 - x1
                            gt_h = y2 - y1
                            
                            target_tx = gt_cx * W - j
                            target_ty = gt_cy * H - i
                            target_tw = np.log(gt_w / anchor_w + 1e-7)
                            target_th = np.log(gt_h / anchor_h + 1e-7)
                            
                            # CIoU loss components
                            ciou = self._compute_ciou(
                                (j + pred_tx) / W, (i + pred_ty) / H,
                                np.exp(pred_tw) * anchor_w / 416, 
                                np.exp(pred_th) * anchor_h / 416,
                                gt_cx / 416, gt_cy / 416, gt_w / 416, gt_h / 416
                            )
                            coord_loss += 1 - ciou
                            coord_count += 1
                            
                            obj_loss += (pred_conf - 1) ** 2
                            obj_count += 1
                            
                            pred_class = pred_data[n, i, j, a, 5:]
                            target_class = np.zeros(self.num_classes, dtype=np.float32)
                            if label < self.num_classes:
                                target_class[int(label)] = 1.0
                            
                            for c in range(self.num_classes):
                                class_loss += (pred_class[c] - target_class[c]) ** 2
                            class_count += self.num_classes
                        else:
                            noobj_loss += (pred_conf - 0) ** 2
                            noobj_count += 1
        
        if coord_count > 0:
            coord_loss /= coord_count
        if obj_count > 0:
            obj_loss /= obj_count
        if noobj_count > 0:
            noobj_loss /= noobj_count
        if class_count > 0:
            class_loss /= class_count
        
        return coord_loss, obj_loss, noobj_loss, class_loss
    
    def _compute_ciou(
        self,
        cx1: float, cy1: float, w1: float, h1: float,
        cx2: float, cy2: float, w2: float, h2: float
    ) -> float:
        """Compute Complete IoU (CIoU).
        
        CIoU = IoU - (d^2 / c^2) - alpha * v
        
        Where:
        - d: distance between box centers
        - c: diagonal of smallest enclosing box
        - v: consistency of aspect ratios
        - alpha: trade-off parameter
        """
        # Convert to corners
        x1_min = cx1 - w1 / 2
        y1_min = cy1 - h1 / 2
        x1_max = cx1 + w1 / 2
        y1_max = cy1 + h1 / 2
        
        x2_min = cx2 - w2 / 2
        y2_min = cy2 - h2 / 2
        x2_max = cx2 + w2 / 2
        y2_max = cy2 + h2 / 2
        
        # Intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        inter_w = max(0, inter_x_max - inter_x_min)
        inter_h = max(0, inter_y_max - inter_y_min)
        inter_area = inter_w * inter_h
        
        # Areas
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        
        if union_area <= 0:
            return 0.0
        
        iou = inter_area / union_area
        
        # Center distance
        center_dist_sq = (cx1 - cx2) ** 2 + (cy1 - cy2) ** 2
        
        # Smallest enclosing box diagonal
        enclose_x_min = min(x1_min, x2_min)
        enclose_y_min = min(y1_min, y2_min)
        enclose_x_max = max(x1_max, x2_max)
        enclose_y_max = max(y1_max, y2_max)
        
        enclose_w = enclose_x_max - enclose_x_min
        enclose_h = enclose_y_max - enclose_y_min
        enclose_diag_sq = enclose_w ** 2 + enclose_h ** 2
        
        if enclose_diag_sq <= 0:
            return iou
        
        # Aspect ratio consistency
        v = (4 / (np.pi ** 2)) * (np.arctan(w2 / (h2 + 1e-7)) - np.arctan(w1 / (h1 + 1e-7))) ** 2
        alpha = v / (1 - iou + v + 1e-7)
        
        # CIoU
        ciou = iou - center_dist_sq / enclose_diag_sq - alpha * v
        
        return max(0, ciou)


class YOLOv4LossSimple(Module):
    """Simplified YOLO v4 Loss for testing."""
    
    def __init__(self, num_classes: int = 80):
        super().__init__()
        self.num_classes = num_classes
    
    def forward(
        self,
        predictions: Dict[str, Tensor],
        targets: Dict[str, Tensor]
    ) -> Tuple[float, Dict[str, float]]:
        """Compute simplified MSE loss."""
        total_loss = 0.0
        count = 0
        
        for scale_name, pred in predictions.items():
            if scale_name in targets:
                target = targets[scale_name]
                diff = pred.data - target.data
                total_loss += np.mean(diff ** 2)
                count += 1
        
        if count > 0:
            total_loss /= count
        
        return float(total_loss), {'total_loss': float(total_loss)}


def encode_targets_v4(
    boxes: np.ndarray,
    labels: np.ndarray,
    anchors: List[Tuple[int, int]],
    grid_sizes: List[int] = [13, 26, 52],
    num_classes: int = 80,
    image_size: int = 416
) -> Dict[str, np.ndarray]:
    """Encode ground truth to YOLO v4 format.
    
    Args:
        boxes: (N_obj, 4) boxes in [x1, y1, x2, y2] format
        labels: (N_obj,) class labels
        anchors: List of anchor box sizes
        grid_sizes: Grid sizes for each scale
        num_classes: Number of classes
        image_size: Input image size
    
    Returns:
        Dict with targets for each scale
    """
    targets = {}
    
    for scale_idx, grid_size in enumerate(grid_sizes):
        scale_anchors = anchors[scale_idx * 3:(scale_idx + 1) * 3]
        target = np.zeros((3, 5 + num_classes, grid_size, grid_size), dtype=np.float32)
        
        for box, label in zip(boxes, labels):
            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            w = x2 - x1
            h = y2 - y1
            
            gx = int(cx / image_size * grid_size)
            gy = int(cy / image_size * grid_size)
            
            gx = min(max(gx, 0), grid_size - 1)
            gy = min(max(gy, 0), grid_size - 1)
            
            best_anchor_idx = 0
            best_iou = 0
            
            for a_idx, (aw, ah) in enumerate(scale_anchors):
                iou = _compute_anchor_iou(w, h, aw, ah)
                if iou > best_iou:
                    best_iou = iou
                    best_anchor_idx = a_idx
            
            tx = cx / image_size * grid_size - gx
            ty = cy / image_size * grid_size - gy
            tw = np.log(w / scale_anchors[best_anchor_idx][0] + 1e-7)
            th = np.log(h / scale_anchors[best_anchor_idx][1] + 1e-7)
            
            target[best_anchor_idx, 0, gy, gx] = tx
            target[best_anchor_idx, 1, gy, gx] = ty
            target[best_anchor_idx, 2, gy, gx] = tw
            target[best_anchor_idx, 3, gy, gx] = th
            target[best_anchor_idx, 4, gy, gx] = 1.0
            
            if label < num_classes:
                target[best_anchor_idx, 5 + int(label), gy, gx] = 1.0
        
        targets[f'scale_{scale_idx}'] = target
    
    return targets


def _compute_anchor_iou(w1: float, h1: float, w2: float, h2: float) -> float:
    """Compute IoU between box and anchor (centered at origin)."""
    inter_w = min(w1, w2)
    inter_h = min(h1, h2)
    inter_area = inter_w * inter_h
    
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0:
        return 0.0
    
    return inter_area / union_area


def decode_predictions_v4(
    predictions: np.ndarray,
    anchors: List[Tuple[int, int]],
    conf_threshold: float = 0.5,
    num_classes: int = 80,
    image_size: int = 416
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode YOLO v4 predictions to bounding boxes.
    
    Args:
        predictions: (3, 5+num_classes, H, W) predictions for one scale
        anchors: Anchor box sizes for this scale
        conf_threshold: Confidence threshold
        num_classes: Number of classes
        image_size: Input image size
    
    Returns:
        boxes: (N, 4) boxes in [x1, y1, x2, y2] format
        scores: (N,) confidence scores
        class_ids: (N,) class indices
    """
    num_anchors = 3
    H, W = predictions.shape[2], predictions.shape[3]
    
    boxes = []
    scores = []
    class_ids = []
    
    for a in range(num_anchors):
        for i in range(H):
            for j in range(W):
                tx = predictions[a, 0, i, j]
                ty = predictions[a, 1, i, j]
                tw = predictions[a, 2, i, j]
                th = predictions[a, 3, i, j]
                conf = predictions[a, 4, i, j]
                
                if conf < conf_threshold:
                    continue
                
                anchor_w, anchor_h = anchors[a]
                
                cx = (j + tx) / W * image_size
                cy = (i + ty) / H * image_size
                w = np.exp(tw) * anchor_w
                h = np.exp(th) * anchor_h
                
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                
                class_probs = predictions[a, 5:5+num_classes, i, j]
                class_id = int(np.argmax(class_probs))
                class_prob = class_probs[class_id]
                
                final_score = float(conf * class_prob)
                
                boxes.append([x1, y1, x2, y2])
                scores.append(final_score)
                class_ids.append(class_id)
    
    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])
    
    return np.array(boxes, dtype=np.float32), np.array(scores, dtype=np.float32), np.array(class_ids, dtype=np.int64)
