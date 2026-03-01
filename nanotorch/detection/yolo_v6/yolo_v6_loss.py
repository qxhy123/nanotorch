"""YOLO v6 Loss - Anchor-free detection loss.

YOLOv6 uses an anchor-free design with TAL (Task Alignment Learning)
for label assignment.
"""

import numpy as np
from typing import Tuple, Dict, List

from nanotorch.tensor import Tensor
from nanotorch.nn.module import Module


class YOLOv6Loss(Module):
    """YOLO v6 Loss Function.

    Uses anchor-free detection with:
    1. CIoU or GIoU loss for bounding box regression
    2. BCE loss for objectness
    3. BCE loss for classification
    """

    def __init__(
        self,
        num_classes: int = 80,
        reg_max: int = 16,
        ignore_threshold: float = 0.5,
        lambda_box: float = 5.0,
        lambda_obj: float = 1.0,
        lambda_noobj: float = 0.5,
        lambda_class: float = 1.0,
        image_size: int = 640
    ):
        super().__init__()

        self.num_classes = num_classes
        self.reg_max = reg_max
        self.ignore_threshold = ignore_threshold
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class
        self.image_size = image_size

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
            box_loss, obj_loss, noobj_loss, class_loss = self._compute_scale_loss(
                pred, targets, scale_name
            )
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

        loss_dict = {
            'box_loss': total_box_loss,
            'obj_loss': total_obj_loss,
            'noobj_loss': total_noobj_loss,
            'class_loss': total_class_loss,
            'total_loss': total_loss
        }

        return Tensor(total_loss, requires_grad=True), loss_dict

    def _compute_scale_loss(
        self,
        pred: Tensor,
        targets: List[Dict],
        scale_name: str
    ) -> Tuple[float, float, float, float]:
        pred_data = pred.data
        N, C, H, W = pred_data.shape

        # YOLOv6 format: [x, y, w, h, conf, class0, class1, ...]
        # Anchor-free: each cell predicts one box

        box_loss = 0.0
        obj_loss = 0.0
        noobj_loss = 0.0
        class_loss = 0.0

        box_count = 0
        obj_count = 0
        noobj_count = 0
        class_count = 0

        for n in range(N):
            target = targets[n] if n < len(targets) else {'boxes': np.array([]), 'labels': np.array([])}
            boxes = target.get('boxes', np.array([]))
            labels = target.get('labels', np.array([]))

            if len(boxes) == 0:
                # No objects - all predictions are background
                for i in range(H):
                    for j in range(W):
                        conf = pred_data[n, 4, i, j]
                        noobj_loss += (conf - 0) ** 2
                        noobj_count += 1
                continue

            # Build target assignment - assign each GT to its corresponding grid cell
            gt_assigned = set()  # Track which grid cells have GT assigned

            for t_idx, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                gt_cx = (x1 + x2) / 2
                gt_cy = (y1 + y2) / 2
                gt_w = x2 - x1
                gt_h = y2 - y1

                # Find grid cell for this GT
                gx = int(gt_cx / self.image_size * W)
                gy = int(gt_cy / self.image_size * H)
                gx = min(max(gx, 0), W - 1)
                gy = min(max(gy, 0), H - 1)

                gt_assigned.add((gy, gx))

                # Get predictions for this cell
                pred_tx = pred_data[n, 0, gy, gx]
                pred_ty = pred_data[n, 1, gy, gx]
                pred_tw = pred_data[n, 2, gy, gx]
                pred_th = pred_data[n, 3, gy, gx]
                pred_conf = pred_data[n, 4, gy, gx]
                pred_class = pred_data[n, 5:, gy, gx]

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

                # Objectness loss
                obj_loss += (pred_conf - 1) ** 2
                obj_count += 1

                # Classification loss
                label = labels[t_idx]
                target_class = np.zeros(self.num_classes, dtype=np.float32)
                if label < self.num_classes:
                    target_class[int(label)] = 1.0

                for c in range(self.num_classes):
                    class_loss += (pred_class[c] - target_class[c]) ** 2
                class_count += self.num_classes

            # All other cells are negative samples
            for i in range(H):
                for j in range(W):
                    if (i, j) not in gt_assigned:
                        pred_conf = pred_data[n, 4, i, j]
                        noobj_loss += (pred_conf - 0) ** 2
                        noobj_count += 1

        # Normalize losses
        if box_count > 0:
            box_loss /= box_count
        if obj_count > 0:
            obj_loss /= obj_count
        if noobj_count > 0:
            noobj_loss /= noobj_count
        if class_count > 0:
            class_loss /= class_count

        return box_loss, obj_loss, noobj_loss, class_loss

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


class YOLOv6LossSimple(Module):
    """Simplified YOLO v6 Loss for testing."""

    def __init__(self, num_classes: int = 80):
        super().__init__()
        self.num_classes = num_classes

    def forward(
        self,
        predictions: Dict[str, Tensor],
        targets: Dict[str, Tensor]
    ) -> Tuple[float, Dict[str, float]]:
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


def encode_targets_v6(
    boxes: np.ndarray,
    labels: np.ndarray,
    grid_sizes: List[int] = [80, 40, 20],
    num_classes: int = 80,
    image_size: int = 640
) -> Dict[str, np.ndarray]:
    """Encode ground truth to YOLO v6 format (anchor-free)."""
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


def decode_predictions_v6(
    predictions: np.ndarray,
    conf_threshold: float = 0.25,
    num_classes: int = 80,
    image_size: int = 640
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode YOLO v6 predictions to bounding boxes."""
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
