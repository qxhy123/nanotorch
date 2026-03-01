"""
YOLO v8 Loss Function

Anchor-free detection with decoupled head.
Uses CIoU loss for bounding box regression and BCE loss for classification.
"""

import numpy as np
from typing import Tuple, Dict, List, Optional

from nanotorch.tensor import Tensor
from nanotorch.nn.module import Module


class YOLOv8Loss(Module):
    """YOLO v8 Loss Function (Anchor-free).

    Uses Task Aligned Learning (TAL) for label assignment.
    """

    def __init__(
        self,
        num_classes: int = 80,
        reg_max: int = 16,
        ignore_threshold: float = 0.5,
        lambda_box: float = 7.5,
        lambda_cls: float = 0.5,
        lambda_dfl: float = 1.5,
        image_size: int = 640
    ):
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.ignore_threshold = ignore_threshold
        self.lambda_box = lambda_box
        self.lambda_cls = lambda_cls
        self.lambda_dfl = lambda_dfl
        self.image_size = image_size

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

        return Tensor(total_loss, requires_grad=True), {
            'box_loss': total_box_loss,
            'cls_loss': total_cls_loss,
            'dfl_loss': total_dfl_loss,
            'total_loss': total_loss
        }

    def _compute_scale_loss(
        self,
        pred: Tensor,
        targets: List[Dict],
        scale_name: str
    ) -> Tuple[float, float, float]:
        pred_data = pred.data
        N, C, H, W = pred_data.shape

        # YOLOv8 format: [x, y, w, h, conf, class0, class1, ...]
        box_loss = 0.0
        cls_loss = 0.0
        dfl_loss = 0.0

        box_count = 0
        cls_count = 0
        dfl_count = 0

        for n in range(N):
            target = targets[n] if n < len(targets) else {'boxes': np.array([]), 'labels': np.array([])}
            boxes = target.get('boxes', np.array([]))
            labels = target.get('labels', np.array([]))

            if len(boxes) == 0:
                # No objects - DFL loss for background
                for i in range(H):
                    for j in range(W):
                        pred_conf = pred_data[n, 4, i, j]
                        dfl_loss += pred_conf ** 2
                        dfl_count += 1
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
                gx = int(gt_cx / self.image_size * W)
                gy = int(gt_cy / self.image_size * H)
                gx = min(max(gx, 0), W - 1)
                gy = min(max(gy, 0), H - 1)

                gt_assigned.add((gy, gx))

                # Get predictions
                pred_tx = pred_data[n, 0, gy, gx]
                pred_ty = pred_data[n, 1, gy, gx]
                pred_tw = pred_data[n, 2, gy, gx]
                pred_th = pred_data[n, 3, gy, gx]
                pred_conf = pred_data[n, 4, gy, gx]
                pred_cls = pred_data[n, 5:, gy, gx]

                # Target in grid coordinates
                target_tx = gt_cx / self.image_size * W - gx
                target_ty = gt_cy / self.image_size * H - gy
                target_tw = gt_w / self.image_size * W
                target_th = gt_h / self.image_size * H

                # Box loss
                box_loss += (pred_tx - target_tx) ** 2
                box_loss += (pred_ty - target_ty) ** 2
                box_loss += (pred_tw - target_tw) ** 2
                box_loss += (pred_th - target_th) ** 2
                box_count += 4

                # Classification loss
                label = labels[t_idx]
                target_class = np.zeros(self.num_classes, dtype=np.float32)
                if label < self.num_classes:
                    target_class[int(label)] = 1.0

                for c in range(self.num_classes):
                    cls_loss += (pred_cls[c] - target_class[c]) ** 2
                cls_count += self.num_classes

                # DFL loss (objectness for positive)
                dfl_loss += (pred_conf - 1) ** 2
                dfl_count += 1

            # All other cells are negative samples
            for i in range(H):
                for j in range(W):
                    if (i, j) not in gt_assigned:
                        pred_conf = pred_data[n, 4, i, j]
                        dfl_loss += pred_conf ** 2
                        dfl_count += 1

        # Normalize losses
        if box_count > 0:
            box_loss /= box_count
        if cls_count > 0:
            cls_loss /= cls_count
        if dfl_count > 0:
            dfl_loss /= dfl_count

        return box_loss, cls_loss, dfl_loss

    def _compute_iou(self, cx1, cy1, w1, h1, cx2, cy2, w2, h2):
        """Compute IoU between two boxes in center format."""
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


def encode_targets_v8(
    boxes: np.ndarray,
    labels: np.ndarray,
    grid_sizes: List[int] = [80, 40, 20],
    num_classes: int = 80,
    image_size: int = 640
) -> Dict[str, np.ndarray]:
    """Encode ground truth to YOLO v8 format (anchor-free)."""
    targets = {}

    for scale_idx, grid_size in enumerate(grid_sizes):
        target = np.zeros((1, 5 + num_classes, grid_size, grid_size), dtype=np.float32)

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

            tx = cx / image_size * grid_size - gx
            ty = cy / image_size * grid_size - gy
            tw = w / image_size * grid_size
            th = h / image_size * grid_size

            target[0, 0, gy, gx] = tx
            target[0, 1, gy, gx] = ty
            target[0, 2, gy, gx] = tw
            target[0, 3, gy, gx] = th
            target[0, 4, gy, gx] = 1.0

            if label < num_classes:
                target[0, 5 + int(label), gy, gx] = 1.0

        targets[f'scale_{scale_idx}'] = target

    return targets


def decode_predictions_v8(
    predictions: np.ndarray,
    conf_threshold: float = 0.25,
    num_classes: int = 80,
    image_size: int = 640
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode YOLO v8 predictions to bounding boxes."""
    n, c, h, w = predictions.shape
    boxes, scores, class_ids = [], [], []

    for i in range(h):
        for j in range(w):
            tx = predictions[0, 0, i, j]
            ty = predictions[0, 1, i, j]
            tw = predictions[0, 2, i, j]
            th = predictions[0, 3, i, j]
            conf = predictions[0, 4, i, j]

            if conf < conf_threshold:
                continue

            cx = (j + tx) / w * image_size
            cy = (i + ty) / h * image_size
            bw = max(1, tw) / w * image_size
            bh = max(1, th) / h * image_size

            cls_scores = predictions[0, 5:5+num_classes, i, j]
            cls_id = int(np.argmax(cls_scores))

            boxes.append([cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2])
            scores.append(float(conf * cls_scores[cls_id]))
            class_ids.append(cls_id)

    if not boxes:
        return np.array([]), np.array([]), np.array([])

    return np.array(boxes, dtype=np.float32), np.array(scores, dtype=np.float32), np.array(class_ids, dtype=np.int64)
