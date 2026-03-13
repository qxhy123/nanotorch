"""
YOLO v11 Loss Function

Latest Ultralytics YOLO loss.
"""

import numpy as np
from typing import Tuple, Dict, List
from nanotorch.tensor import Tensor
from nanotorch.nn.module import Module


class YOLOv11Loss(Module):
    """Lightweight placeholder loss used for examples and smoke tests."""
    def __init__(self, num_classes: int = 80):
        super().__init__()
        self.num_classes = num_classes
    
    def forward(self, predictions: Dict[str, Tensor], targets: List[Dict]) -> Tuple[Tensor, Dict[str, float]]:
        total_loss = 0.0
        for pred in predictions.values():
            total_loss += np.mean(pred.data ** 2)
        return Tensor(np.array(total_loss)), {'total_loss': float(total_loss)}


class YOLOv11LossSimple(Module):
    """Simplified detached MSE loss for testing utilities."""
    def __init__(self, num_classes: int = 80):
        super().__init__()
        self.num_classes = num_classes
    
    def forward(self, predictions: Dict[str, Tensor], targets: Dict[str, Tensor]) -> Tuple[float, Dict[str, float]]:
        total_loss = 0.0
        count = 0
        for k, pred in predictions.items():
            if k in targets:
                total_loss += np.mean((pred.data - targets[k].data) ** 2)
                count += 1
        avg_loss = total_loss / max(count, 1)
        return float(avg_loss), {'total_loss': float(avg_loss)}


def encode_targets_v11(boxes, labels, grid_sizes=[80, 40, 20], num_classes=80, image_size=640):
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


def decode_predictions_v11(predictions, conf_threshold=0.25, num_classes=80, image_size=640):
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
