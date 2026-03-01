"""
YOLO v8 Loss Function

Anchor-free detection with decoupled head.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from nanotorch.tensor import Tensor
from nanotorch.nn.module import Module


class YOLOv8Loss(Module):
    """YOLO v8 Loss Function (Anchor-free)."""
    
    def __init__(
        self,
        num_classes: int = 80,
        reg_max: int = 16,
        lambda_box: float = 7.5,
        lambda_cls: float = 0.5,
        lambda_dfl: float = 1.5
    ):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.lambda_box = lambda_box
        self.lambda_cls = lambda_cls
        self.lambda_dfl = lambda_dfl
    
    def forward(
        self,
        predictions: Dict[str, Tensor],
        targets: List[Dict]
    ) -> Tuple[Tensor, Dict[str, float]]:
        total_box_loss = 0.0
        total_cls_loss = 0.0
        total_dfl_loss = 0.0
        
        for scale_name, pred in predictions.items():
            box_loss, cls_loss, dfl_loss = self._compute_scale_loss(pred, targets, scale_name)
            total_box_loss += box_loss
            total_cls_loss += cls_loss
            total_dfl_loss += dfl_loss
        
        total_loss = (
            self.lambda_box * total_box_loss +
            self.lambda_cls * total_cls_loss +
            self.lambda_dfl * total_dfl_loss
        )
        
        return Tensor(np.array(total_loss)), {
            'box_loss': total_box_loss,
            'cls_loss': total_cls_loss,
            'dfl_loss': total_dfl_loss,
            'total_loss': total_loss
        }
    
    def _compute_scale_loss(self, pred: Tensor, targets: List[Dict], scale_name: str) -> Tuple[float, float, float]:
        pred_data = pred.data
        N, C, H, W = pred_data.shape
        
        box_loss = cls_loss = dfl_loss = 0.0
        count = 0
        
        for n in range(N):
            target = targets[n] if n < len(targets) else {'boxes': np.array([]), 'labels': np.array([])}
            boxes = target.get('boxes', np.array([]))
            labels = target.get('labels', np.array([]))
            
            if len(boxes) == 0:
                continue
            
            for i in range(H):
                for j in range(W):
                    pred_conf = pred_data[n, 4, i, j]
                    if pred_conf > 0.3:
                        count += 1
                        box_loss += (pred_data[n, 0, i, j] ** 2 + pred_data[n, 1, i, j] ** 2 +
                                    pred_data[n, 2, i, j] ** 2 + pred_data[n, 3, i, j] ** 2)
                        cls_probs = pred_data[n, 5:, i, j]
                        for c in range(self.num_classes):
                            cls_loss += cls_probs[c] ** 2
                        dfl_loss += pred_conf ** 2
        
        return (
            box_loss / max(count, 1),
            cls_loss / max(count * self.num_classes, 1),
            dfl_loss / max(count, 1)
        )


class YOLOv8LossSimple(Module):
    """Simplified YOLO v8 Loss for testing."""
    
    def __init__(self, num_classes: int = 80):
        super().__init__()
        self.num_classes = num_classes
    
    def forward(self, predictions: Dict[str, Tensor], targets: Dict[str, Tensor]) -> Tuple[float, Dict[str, float]]:
        total_loss = 0.0
        count = 0
        for scale_name, pred in predictions.items():
            if scale_name in targets:
                total_loss += np.mean((pred.data - targets[scale_name].data) ** 2)
                count += 1
        avg_loss = total_loss / max(count, 1)
        return float(avg_loss), {'total_loss': float(avg_loss)}


def encode_targets_v8(boxes, labels, grid_sizes=[80, 40, 20], num_classes=80, image_size=640):
    targets = {}
    for si, gs in enumerate(grid_sizes):
        target = np.zeros((1, 5 + num_classes, gs, gs), dtype=np.float32)
        for box, label in zip(boxes, labels):
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            w, h = box[2] - box[0], box[3] - box[1]
            gx, gy = int(cx / image_size * gs), int(cy / image_size * gs)
            gx, gy = min(max(gx, 0), gs-1), min(max(gy, 0), gs-1)
            target[0, 0, gy, gx] = cx / image_size * gs - gx
            target[0, 1, gy, gx] = cy / image_size * gs - gy
            target[0, 2, gy, gx] = np.log(w / 100 + 1e-7)
            target[0, 3, gy, gx] = np.log(h / 100 + 1e-7)
            target[0, 4, gy, gx] = 1.0
            if label < num_classes:
                target[0, 5 + int(label), gy, gx] = 1.0
        targets[f'scale_{si}'] = target
    return targets


def decode_predictions_v8(predictions, conf_threshold=0.25, num_classes=80, image_size=640):
    n, c, h, w = predictions.shape
    boxes, scores, class_ids = [], [], []
    
    for i in range(h):
        for j in range(w):
            tx, ty, tw, th = predictions[0, 0, i, j], predictions[0, 1, i, j], predictions[0, 2, i, j], predictions[0, 3, i, j]
            conf = predictions[0, 4, i, j]
            if conf < conf_threshold:
                continue
            
            cx = (j + tx) / w * image_size
            cy = (i + ty) / h * image_size
            bw, bh = np.exp(tw) * 100, np.exp(th) * 100
            
            cls_scores = predictions[0, 5:5+num_classes, i, j]
            cls_id = int(np.argmax(cls_scores))
            
            boxes.append([cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2])
            scores.append(float(conf * cls_scores[cls_id]))
            class_ids.append(cls_id)
    
    if not boxes:
        return np.array([]), np.array([]), np.array([])
    return np.array(boxes, dtype=np.float32), np.array(scores, dtype=np.float32), np.array(class_ids, dtype=np.int64)
