"""
YOLO v2 Loss Function

YOLO v2 uses a multi-part loss function:
1. Coordinate loss (x, y, w, h)
2. Objectness loss (confidence)
3. No-objectness loss
4. Classification loss

Key features:
- Direct location prediction with sigmoid
- Anchor-based detection
- Sum-squared error for all components
"""

import numpy as np
from typing import Tuple, Dict, List, Optional

from nanotorch.tensor import Tensor
from nanotorch.nn.module import Module


class YOLOv2Loss(Module):
    """YOLO v2 Loss Function.
    
    The loss function consists of:
    - Coordinate loss: MSE for bounding box predictions
    - Objectness loss: BCE for object confidence
    - No-objectness loss: BCE for background confidence
    - Classification loss: BCE for class predictions
    
    Args:
        num_classes: Number of object classes
        anchors: List of anchor box sizes (width, height)
        lambda_coord: Weight for coordinate loss
        lambda_noobj: Weight for no-objectness loss
        lambda_obj: Weight for objectness loss
        lambda_class: Weight for classification loss
        ignore_threshold: IoU threshold for ignoring predictions
    """
    
    VOC_ANCHORS = [
        (1.3221, 1.73145),
        (3.19275, 4.00944),
        (5.05587, 8.09892),
        (9.47112, 4.84053),
        (11.2364, 10.0071)
    ]
    
    def __init__(
        self,
        num_classes: int = 20,
        anchors: Optional[List[Tuple[float, float]]] = None,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
        lambda_obj: float = 1.0,
        lambda_class: float = 1.0,
        ignore_threshold: float = 0.6
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.anchors = anchors or self.VOC_ANCHORS
        self.num_anchors = len(self.anchors)
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_obj = lambda_obj
        self.lambda_class = lambda_class
        self.ignore_threshold = ignore_threshold
    
    def forward(
        self,
        predictions: Dict[str, Tensor],
        targets: List[Dict]
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute YOLO v2 loss.
        
        Args:
            predictions: Dictionary containing 'output' tensor
                Shape: (N, num_anchors * (5 + num_classes), H, W)
            targets: List of target dictionaries, each containing:
                - 'boxes': numpy array of shape (num_objects, 4)
                - 'labels': numpy array of shape (num_objects,)
        
        Returns:
            Tuple of (total_loss, loss_dict)
        """
        pred = predictions['output']
        pred_data = pred.data
        
        N, C, H, W = pred_data.shape
        num_values = 5 + self.num_classes
        
        pred_reshaped = pred_data.reshape(N, self.num_anchors, num_values, H, W)
        pred_reshaped = pred_reshaped.transpose(0, 3, 4, 1, 2)
        
        total_coord_loss = 0.0
        total_obj_loss = 0.0
        total_noobj_loss = 0.0
        total_class_loss = 0.0
        
        coord_count = 0
        obj_count = 0
        noobj_count = 0
        class_count = 0
        
        for n in range(N):
            target = targets[n] if n < len(targets) else {'boxes': np.array([]), 'labels': np.array([])}
            boxes = target.get('boxes', np.array([]))
            labels = target.get('labels', np.array([]))
            
            for i in range(H):
                for j in range(W):
                    for a in range(self.num_anchors):
                        pred_tx = pred_reshaped[n, i, j, a, 0]
                        pred_ty = pred_reshaped[n, i, j, a, 1]
                        pred_tw = pred_reshaped[n, i, j, a, 2]
                        pred_th = pred_reshaped[n, i, j, a, 3]
                        pred_conf = pred_reshaped[n, i, j, a, 4]
                        pred_class = pred_reshaped[n, i, j, a, 5:]
                        
                        cell_cx = (j + pred_tx) / W
                        cell_cy = (i + pred_ty) / H
                        
                        anchor_w, anchor_h = self.anchors[a]
                        pred_w = anchor_w * pred_tw
                        pred_h = anchor_h * pred_th
                        
                        if len(boxes) == 0:
                            total_noobj_loss += pred_conf ** 2
                            noobj_count += 1
                            continue
                        
                        best_iou = 0.0
                        best_target = None
                        
                        for b_idx, box in enumerate(boxes):
                            target_cx = ((box[0] + box[2]) / 2) / 416
                            target_cy = ((box[1] + box[3]) / 2) / 416
                            target_w = (box[2] - box[0]) / 416
                            target_h = (box[3] - box[1]) / 416
                            
                            iou = self._compute_iou(
                                cell_cx, cell_cy, pred_w / 416, pred_h / 416,
                                target_cx, target_cy, target_w, target_h
                            )
                            
                            if iou > best_iou:
                                best_iou = iou
                                best_target = (target_cx, target_cy, target_w, target_h, labels[b_idx])
                        
                        if best_iou > self.ignore_threshold and best_target is not None:
                            target_cx, target_cy, target_w, target_h, target_class = best_target
                            
                            target_tx = target_cx * W - j
                            target_ty = target_cy * H - i
                            
                            total_coord_loss += (pred_tx - target_tx) ** 2
                            total_coord_loss += (pred_ty - target_ty) ** 2
                            coord_count += 2
                            
                            anchor_w, anchor_h = self.anchors[a]
                            if target_w > 0 and target_h > 0:
                                target_tw = target_w * 416 / anchor_w
                                target_th = target_h * 416 / anchor_h
                                total_coord_loss += (pred_tw - target_tw) ** 2
                                total_coord_loss += (pred_th - target_th) ** 2
                                coord_count += 2
                            
                            total_obj_loss += (pred_conf - 1) ** 2
                            obj_count += 1
                            
                            class_probs = np.zeros(self.num_classes)
                            if target_class < self.num_classes:
                                class_probs[int(target_class)] = 1.0
                            
                            for c in range(self.num_classes):
                                total_class_loss += (pred_class[c] - class_probs[c]) ** 2
                            class_count += self.num_classes
                        else:
                            total_noobj_loss += pred_conf ** 2
                            noobj_count += 1
        
        coord_loss = total_coord_loss / max(coord_count, 1)
        obj_loss = total_obj_loss / max(obj_count, 1)
        noobj_loss = total_noobj_loss / max(noobj_count, 1)
        class_loss = total_class_loss / max(class_count, 1)
        
        total_loss = (
            self.lambda_coord * coord_loss +
            self.lambda_obj * obj_loss +
            self.lambda_noobj * noobj_loss +
            self.lambda_class * class_loss
        )
        
        return Tensor(np.array(total_loss)), {
            'coord_loss': coord_loss,
            'obj_loss': obj_loss,
            'noobj_loss': noobj_loss,
            'class_loss': class_loss,
            'total_loss': total_loss
        }
    
    def _compute_iou(
        self,
        cx1: float, cy1: float, w1: float, h1: float,
        cx2: float, cy2: float, w2: float, h2: float
    ) -> float:
        """Compute IoU between two boxes.
        
        Args:
            cx1, cy1, w1, h1: Center x, y, width, height of box 1
            cx2, cy2, w2, h2: Center x, y, width, height of box 2
        
        Returns:
            IoU value
        """
        x1_min = cx1 - w1 / 2
        y1_min = cy1 - h1 / 2
        x1_max = cx1 + w1 / 2
        y1_max = cy1 + h1 / 2
        
        x2_min = cx2 - w2 / 2
        y2_min = cy2 - h2 / 2
        x2_max = cx2 + w2 / 2
        y2_max = cy2 + h2 / 2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        inter_w = max(0, inter_x_max - inter_x_min)
        inter_h = max(0, inter_y_max - inter_y_min)
        inter_area = inter_w * inter_h
        
        area1 = w1 * h1
        area2 = w2 * h2
        union_area = area1 + area2 - inter_area
        
        if union_area <= 0:
            return 0.0
        
        return inter_area / union_area


class YOLOv2LossSimple(Module):
    """Simplified YOLO v2 Loss for testing.
    
    Uses simple MSE loss for testing purposes.
    """
    
    def __init__(self, num_classes: int = 20):
        super().__init__()
        self.num_classes = num_classes
    
    def forward(
        self,
        predictions: Dict[str, Tensor],
        targets: Dict[str, Tensor]
    ) -> Tuple[float, Dict[str, float]]:
        """Compute simple MSE loss."""
        pred = predictions['output']
        if 'output' in targets:
            target = targets['output']
            loss = np.mean((pred.data - target.data) ** 2)
        else:
            loss = np.mean(pred.data ** 2)
        
        return float(loss), {'total_loss': float(loss)}


def encode_targets_v2(
    boxes: np.ndarray,
    labels: np.ndarray,
    grid_size: int = 13,
    num_anchors: int = 5,
    num_classes: int = 20,
    image_size: int = 416,
    anchors: Optional[List[Tuple[float, float]]] = None
) -> np.ndarray:
    """Encode ground truth boxes for YOLO v2 training.
    
    Args:
        boxes: numpy array of shape (num_objects, 4) in [x1, y1, x2, y2] format
        labels: numpy array of shape (num_objects,)
        grid_size: Size of the output grid
        num_anchors: Number of anchor boxes
        num_classes: Number of object classes
        image_size: Size of the input image
        anchors: List of anchor sizes
    
    Returns:
        Target tensor of shape (num_anchors * (5 + num_classes), grid_size, grid_size)
    """
    if anchors is None:
        anchors = YOLOv2Loss.VOC_ANCHORS
    
    num_values = 5 + num_classes
    target = np.zeros((num_anchors * num_values, grid_size, grid_size), dtype=np.float32)
    
    for box, label in zip(boxes, labels):
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        w = box[2] - box[0]
        h = box[3] - box[1]
        
        grid_x = int(cx / image_size * grid_size)
        grid_y = int(cy / image_size * grid_size)
        
        grid_x = min(max(grid_x, 0), grid_size - 1)
        grid_y = min(max(grid_y, 0), grid_size - 1)
        
        best_anchor = 0
        best_iou = 0.0
        
        for a_idx, (anchor_w, anchor_h) in enumerate(anchors):
            inter_w = min(w, anchor_w * 32)
            inter_h = min(h, anchor_h * 32)
            inter_area = inter_w * inter_h
            
            union_area = w * h + anchor_w * anchor_h * 32 * 32 - inter_area
            iou = inter_area / union_area if union_area > 0 else 0
            
            if iou > best_iou:
                best_iou = iou
                best_anchor = a_idx
        
        tx = cx / image_size * grid_size - grid_x
        ty = cy / image_size * grid_size - grid_y
        
        anchor_w, anchor_h = anchors[best_anchor]
        tw = w / (anchor_w * 32)
        th = h / (anchor_h * 32)
        
        offset = best_anchor * num_values
        target[offset + 0, grid_y, grid_x] = tx
        target[offset + 1, grid_y, grid_x] = ty
        target[offset + 2, grid_y, grid_x] = tw
        target[offset + 3, grid_y, grid_x] = th
        target[offset + 4, grid_y, grid_x] = 1.0
        
        if label < num_classes:
            target[offset + 5 + int(label), grid_y, grid_x] = 1.0
    
    return target


def decode_predictions_v2(
    predictions: np.ndarray,
    conf_threshold: float = 0.5,
    num_anchors: int = 5,
    num_classes: int = 20,
    image_size: int = 416,
    anchors: Optional[List[Tuple[float, float]]] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode YOLO v2 predictions to bounding boxes.
    
    Args:
        predictions: Model output of shape (N, num_anchors * (5 + num_classes), H, W)
        conf_threshold: Confidence threshold for filtering
        num_anchors: Number of anchor boxes
        num_classes: Number of object classes
        image_size: Size of the input image
        anchors: List of anchor sizes
    
    Returns:
        Tuple of (boxes, scores, class_ids)
    """
    if anchors is None:
        anchors = YOLOv2Loss.VOC_ANCHORS
    
    if predictions.ndim == 4:
        predictions = predictions[0]
    
    num_values = 5 + num_classes
    C, H, W = predictions.shape
    
    pred_reshaped = predictions.reshape(num_anchors, num_values, H, W)
    pred_reshaped = pred_reshaped.transpose(0, 2, 3, 1)
    
    boxes = []
    scores = []
    class_ids = []
    
    for i in range(H):
        for j in range(W):
            for a in range(num_anchors):
                tx = pred_reshaped[a, i, j, 0]
                ty = pred_reshaped[a, i, j, 1]
                tw = pred_reshaped[a, i, j, 2]
                th = pred_reshaped[a, i, j, 3]
                conf = pred_reshaped[a, i, j, 4]
                class_probs = pred_reshaped[a, i, j, 5:]
                
                if conf < conf_threshold:
                    continue
                
                cx = (j + tx) / W * image_size
                cy = (i + ty) / H * image_size
                
                anchor_w, anchor_h = anchors[a]
                w = tw * anchor_w * 32
                h = th * anchor_h * 32
                
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                
                class_id = int(np.argmax(class_probs))
                score = float(conf * class_probs[class_id])
                
                if score >= conf_threshold:
                    boxes.append([x1, y1, x2, y2])
                    scores.append(score)
                    class_ids.append(class_id)
    
    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])
    
    return (
        np.array(boxes, dtype=np.float32),
        np.array(scores, dtype=np.float32),
        np.array(class_ids, dtype=np.int64)
    )
