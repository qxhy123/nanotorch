"""
YOLO v7 Loss Function

Similar to previous YOLO versions with improvements for E-ELAN training.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from nanotorch.tensor import Tensor
from nanotorch.nn.module import Module


class YOLOv7Loss(Module):
    """YOLO v7 Loss Function."""
    
    def __init__(
        self,
        num_classes: int = 80,
        anchors: Optional[List[Tuple[int, int]]] = None,
        ignore_threshold: float = 0.5,
        lambda_box: float = 5.0,
        lambda_obj: float = 1.0,
        lambda_noobj: float = 0.5,
        lambda_class: float = 1.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_threshold = ignore_threshold
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class
        self.anchors = anchors or [
            (12, 16), (19, 36), (40, 28),
            (36, 75), (76, 55), (72, 146),
            (142, 110), (192, 243), (459, 401)
        ]
    
    def forward(
        self,
        predictions: Dict[str, Tensor],
        targets: List[Dict]
    ) -> Tuple[Tensor, Dict[str, float]]:
        total_box_loss = 0.0
        total_obj_loss = 0.0
        total_noobj_loss = 0.0
        total_class_loss = 0.0
        
        for scale_name, pred in predictions.items():
            box_loss, obj_loss, noobj_loss, class_loss = self._compute_scale_loss(pred, targets, scale_name)
            total_box_loss += box_loss
            total_obj_loss += obj_loss
            total_noobj_loss += noobj_loss
            total_class_loss += class_loss
        
        total_loss = (
            self.lambda_box * total_box_loss +
            self.lambda_obj * total_obj_loss +
            self.lambda_noobj * total_noobj_loss +
            self.lambda_class * total_class_loss
        )
        
        return Tensor(np.array(total_loss)), {
            'box_loss': total_box_loss,
            'obj_loss': total_obj_loss,
            'noobj_loss': total_noobj_loss,
            'class_loss': total_class_loss,
            'total_loss': total_loss
        }
    
    def _compute_scale_loss(self, pred: Tensor, targets: List[Dict], scale_name: str) -> Tuple[float, float, float, float]:
        pred_data = pred.data
        N, C, H, W = pred_data.shape
        # YOLOv7 format: (N, C, H, W) where C = 5 + num_classes

        box_loss = obj_loss = noobj_loss = class_loss = 0.0
        box_count = obj_count = noobj_count = class_count = 0

        for n in range(N):
            target = targets[n] if n < len(targets) else {'boxes': np.array([]), 'labels': np.array([])}
            boxes = target.get('boxes', np.array([]))
            labels = target.get('labels', np.array([]))

            if len(boxes) == 0:
                for i in range(H):
                    for j in range(W):
                        noobj_loss += pred_data[n, 4, i, j] ** 2
                        noobj_count += 1
                continue

            # Assign each GT to its corresponding grid cell
            gt_assigned = set()

            for t_idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                gt_cx = (x1 + x2) / 2
                gt_cy = (y1 + y2) / 2
                gt_w = x2 - x1
                gt_h = y2 - y1

                # Find grid cell
                gx = int(gt_cx / 640 * W)
                gy = int(gt_cy / 640 * H)
                gx = min(max(gx, 0), W - 1)
                gy = min(max(gy, 0), H - 1)

                gt_assigned.add((gy, gx))

                # Get predictions
                pred_tx = pred_data[n, 0, gy, gx]
                pred_ty = pred_data[n, 1, gy, gx]
                pred_tw = pred_data[n, 2, gy, gx]
                pred_th = pred_data[n, 3, gy, gx]
                pred_conf = pred_data[n, 4, gy, gx]
                class_probs = pred_data[n, 5:, gy, gx]

                # Target in grid coordinates
                target_tx = gt_cx / 640 * W - gx
                target_ty = gt_cy / 640 * H - gy
                target_tw = gt_w / 640 * W
                target_th = gt_h / 640 * H

                # Box loss
                box_loss += (pred_tx - target_tx) ** 2
                box_loss += (pred_ty - target_ty) ** 2
                box_loss += (pred_tw - target_tw) ** 2
                box_loss += (pred_th - target_th) ** 2
                box_count += 4

                # Objectness loss
                obj_loss += (pred_conf - 1) ** 2
                obj_count += 1

                # Classification loss
                label = labels[t_idx]
                target_class = np.zeros(self.num_classes, dtype=np.float32)
                if label < self.num_classes:
                    target_class[int(label)] = 1.0
                for c in range(self.num_classes):
                    class_loss += (class_probs[c] - target_class[c]) ** 2
                class_count += self.num_classes

            # All other cells are negative samples
            for i in range(H):
                for j in range(W):
                    if (i, j) not in gt_assigned:
                        pred_conf = pred_data[n, 4, i, j]
                        noobj_loss += pred_conf ** 2
                        noobj_count += 1

        return (
            box_loss / max(box_count, 1),
            obj_loss / max(obj_count, 1),
            noobj_loss / max(noobj_count, 1),
            class_loss / max(class_count, 1)
        )
    
    def _compute_iou(self, cx1, cy1, w1, h1, cx2, cy2, w2, h2):
        x1_min, y1_min = cx1 - w1/2, cy1 - h1/2
        x1_max, y1_max = cx1 + w1/2, cy1 + h1/2
        x2_min, y2_min = cx2 - w2/2, cy2 - h2/2
        x2_max, y2_max = cx2 + w2/2, cy2 + h2/2
        
        inter_w = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
        inter_h = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
        inter_area = inter_w * inter_h
        union_area = w1*h1 + w2*h2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0


class YOLOv7LossSimple(Module):
    """Simplified YOLO v7 Loss for testing."""
    
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


def encode_targets_v7(boxes, labels, grid_sizes=[80, 40, 20], num_classes=80, image_size=640):
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


def decode_predictions_v7(predictions, conf_threshold=0.25, num_classes=80, image_size=640):
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
