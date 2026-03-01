"""
YOLO v1 Loss Function

The YOLO v1 loss function is a sum-squared error loss that combines:
1. Bounding box coordinate loss (x, y, w, h)
2. Confidence loss (objectness)
3. Classification loss (class probabilities)

The loss uses different weights for different components:
- λ_coord = 5.0 for bounding box coordinate loss
- λ_noobj = 0.5 for confidence loss when no object is present

Reference:
    "You Only Look Once: Unified, Real-Time Object Detection"
    Equation 1-3 in Section 2.2
"""

import numpy as np
from typing import Tuple, Dict
from nanotorch.tensor import Tensor
from nanotorch.nn.module import Module


class YOLOv1Loss(Module):
    """YOLO v1 Loss Function.
    
    The loss is computed as:
    
    Loss = λ_coord * (coordinate_loss) + 
           (confidence_loss_with_object) + 
           λ_noobj * (confidence_loss_no_object) + 
           (class_loss)
    
    Where:
    - λ_coord = 5.0: Weight for bounding box coordinate loss
    - λ_noobj = 0.5: Weight for confidence loss when no object present
    - coordinate_loss: Sum-squared error for x, y, sqrt(w), sqrt(h)
    - confidence_loss: Sum-squared error for confidence scores
    - class_loss: Sum-squared error for class probabilities
    
    Args:
        S: Grid size (default: 7)
        B: Number of bounding boxes per grid cell (default: 2)
        C: Number of classes (default: 20)
        coord_weight: λ_coord, weight for coordinate loss (default: 5.0)
        noobj_weight: λ_noobj, weight for no-object confidence loss (default: 0.5)
    """
    
    def __init__(
        self,
        S: int = 7,
        B: int = 2,
        C: int = 20,
        coord_weight: float = 5.0,
        noobj_weight: float = 0.5
    ) -> None:
        super().__init__()
        
        self.S = S
        self.B = B
        self.C = C
        self.coord_weight = coord_weight
        self.noobj_weight = noobj_weight
    
    def forward(
        self,
        predictions: Tensor,
        targets: Tensor
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute YOLO v1 loss.
        
        Args:
            predictions: (N, S, S, B*5+C) or (N, S*S*(B*5+C)) predictions
            targets: (N, S, S, B*5+C) or (N, S*S*(B*5+C)) targets
        
        Returns:
            total_loss: Total loss value
            loss_dict: Dictionary with individual loss components
        """
        pred_data = predictions.data
        target_data = targets.data
        
        if pred_data.ndim == 2:
            pred_data = pred_data.reshape(-1, self.S, self.S, self.B * 5 + self.C)
        if target_data.ndim == 2:
            target_data = target_data.reshape(-1, self.S, self.S, self.B * 5 + self.C)
        
        N = pred_data.shape[0]
        
        coord_loss = 0.0
        obj_conf_loss = 0.0
        noobj_conf_loss = 0.0
        class_loss = 0.0
        
        coord_count = 0
        obj_count = 0
        noobj_count = 0
        class_count = 0
        
        for n in range(N):
            for i in range(self.S):
                for j in range(self.S):
                    target_box1 = target_data[n, i, j, :5]
                    target_box2 = target_data[n, i, j, 5:10] if self.B > 1 else None
                    target_class = target_data[n, i, j, self.B*5:]
                    
                    has_object = target_box1[4] > 0.5
                    
                    if has_object:
                        for b in range(self.B):
                            pred_box = pred_data[n, i, j, b*5:(b+1)*5]
                            target_box = target_box1 if b == 0 else (target_box2 if target_box2 is not None else target_box1)
                            
                            # Coordinate loss (x, y, sqrt(w), sqrt(h))
                            coord_loss += (pred_box[0] - target_box[0]) ** 2  # x
                            coord_loss += (pred_box[1] - target_box[1]) ** 2  # y
                            coord_loss += (np.sqrt(pred_box[2] + 1e-7) - np.sqrt(target_box[2] + 1e-7)) ** 2  # sqrt(w)
                            coord_loss += (np.sqrt(pred_box[3] + 1e-7) - np.sqrt(target_box[3] + 1e-7)) ** 2  # sqrt(h)
                            coord_count += 4
                            
                            # Confidence loss (object present)
                            obj_conf_loss += (pred_box[4] - target_box[4]) ** 2
                            obj_count += 1
                        
                        # Class loss
                        pred_class = pred_data[n, i, j, self.B*5:]
                        for c in range(self.C):
                            class_loss += (pred_class[c] - target_class[c]) ** 2
                        class_count += self.C
                    else:
                        for b in range(self.B):
                            pred_box = pred_data[n, i, j, b*5:(b+1)*5]
                            noobj_conf_loss += (pred_box[4] - 0) ** 2
                            noobj_count += 1
        
        if coord_count > 0:
            coord_loss /= coord_count
        if obj_count > 0:
            obj_conf_loss /= obj_count
        if noobj_count > 0:
            noobj_conf_loss /= noobj_count
        if class_count > 0:
            class_loss /= class_count
        
        total_loss = (
            self.coord_weight * coord_loss +
            obj_conf_loss +
            self.noobj_weight * noobj_conf_loss +
            class_loss
        )
        
        loss_dict = {
            'coord_loss': coord_loss,
            'obj_conf_loss': obj_conf_loss,
            'noobj_conf_loss': noobj_conf_loss,
            'class_loss': class_loss,
            'total_loss': total_loss
        }
        
        return Tensor(total_loss), loss_dict


class YOLOv1LossSimple(Module):
    """Simplified YOLO v1 Loss for testing.
    
    A simpler version that assumes targets are already in the correct format
    and computes loss directly without complex matching.
    
    Args:
        coord_weight: Weight for coordinate loss
        conf_weight: Weight for confidence loss
        class_weight: Weight for class loss
    """
    
    def __init__(
        self,
        coord_weight: float = 5.0,
        conf_weight: float = 1.0,
        class_weight: float = 1.0
    ) -> None:
        super().__init__()
        
        self.coord_weight = coord_weight
        self.conf_weight = conf_weight
        self.class_weight = class_weight
    
    def forward(
        self,
        predictions: Tensor,
        targets: Tensor
    ) -> Tuple[float, Dict[str, float]]:
        """Compute simplified loss.
        
        Args:
            predictions: (N, D) flattened predictions
            targets: (N, D) flattened targets
        
        Returns:
            total_loss: Total loss value
            loss_dict: Dictionary with individual loss components
        """
        pred_data = predictions.data
        target_data = targets.data
        
        diff = pred_data - target_data
        loss = np.mean(diff ** 2)
        
        return float(loss), {'total_loss': float(loss)}


def encode_targets(
    boxes: np.ndarray,
    labels: np.ndarray,
    S: int = 7,
    B: int = 2,
    C: int = 20,
    image_size: int = 448
) -> np.ndarray:
    """Encode ground truth boxes and labels to YOLO v1 target format.
    
    Args:
        boxes: (N_obj, 4) boxes in [x1, y1, x2, y2] format (pixel coordinates)
        labels: (N_obj,) class labels
        S: Grid size (default: 7)
        B: Number of bounding boxes per cell (default: 2)
        C: Number of classes (default: 20)
        image_size: Input image size (default: 448)
    
    Returns:
        target: (S, S, B*5+C) target tensor
    """
    target = np.zeros((S, S, B * 5 + C), dtype=np.float32)
    
    cell_size = image_size / S
    
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        
        grid_x = int(cx / cell_size)
        grid_y = int(cy / cell_size)
        
        grid_x = min(max(grid_x, 0), S - 1)
        grid_y = min(max(grid_y, 0), S - 1)
        
        local_x = (cx / cell_size) - grid_x
        local_y = (cy / cell_size) - grid_y
        local_w = w / image_size
        local_h = h / image_size
        
        for b in range(B):
            target[grid_y, grid_x, b*5 + 0] = local_x
            target[grid_y, grid_x, b*5 + 1] = local_y
            target[grid_y, grid_x, b*5 + 2] = local_w
            target[grid_y, grid_x, b*5 + 3] = local_h
            target[grid_y, grid_x, b*5 + 4] = 1.0
        
        if label < C:
            target[grid_y, grid_x, B*5 + int(label)] = 1.0
    
    return target


def decode_predictions(
    predictions: np.ndarray,
    S: int = 7,
    B: int = 2,
    C: int = 20,
    conf_threshold: float = 0.5,
    image_size: int = 448
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode YOLO v1 predictions to bounding boxes.
    
    Args:
        predictions: (S, S, B*5+C) or (1, S, S, B*5+C) predictions
        S: Grid size
        B: Number of bounding boxes per cell
        C: Number of classes
        conf_threshold: Confidence threshold
        image_size: Input image size
    
    Returns:
        boxes: (N, 4) boxes in [x1, y1, x2, y2] format
        scores: (N,) confidence scores
        class_ids: (N,) class indices
    """
    if predictions.ndim == 4:
        predictions = predictions[0]
    
    cell_size = image_size / S
    
    boxes = []
    scores = []
    class_ids = []
    
    for i in range(S):
        for j in range(S):
            for b in range(B):
                conf = predictions[i, j, b*5 + 4]
                
                if conf < conf_threshold:
                    continue
                
                local_x = predictions[i, j, b*5 + 0]
                local_y = predictions[i, j, b*5 + 1]
                local_w = predictions[i, j, b*5 + 2]
                local_h = predictions[i, j, b*5 + 3]
                
                cx = (j + local_x) * cell_size
                cy = (i + local_y) * cell_size
                w = local_w * image_size
                h = local_h * image_size
                
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                
                class_probs = predictions[i, j, B*5:]
                class_id = int(np.argmax(class_probs))
                class_prob = class_probs[class_id]
                
                final_score = conf * class_prob
                
                boxes.append([x1, y1, x2, y2])
                scores.append(final_score)
                class_ids.append(class_id)
    
    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])
    
    boxes = np.array(boxes, dtype=np.float32)
    scores = np.array(scores, dtype=np.float32)
    class_ids = np.array(class_ids, dtype=np.int64)
    
    return boxes, scores, class_ids
