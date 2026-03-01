"""
Loss functions for YOLO v12 object detection.

This module implements the composite loss used in YOLO models:
- CIoU Loss: Bounding box regression loss with complete IoU
- DFL Loss: Distribution Focal Loss for box regression
- VFL Loss: Varifocal Loss for classification
- BCE Loss: Binary cross-entropy for objectness

The total loss is a weighted combination:
    total_loss = box_weight * box_loss + cls_weight * cls_loss + dfl_weight * dfl_loss

Reference:
- CIoU: https://arxiv.org/abs/1911.08287
- DFL: https://arxiv.org/abs/2006.04388
- VFL: https://arxiv.org/abs/2008.13367
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union
from nanotorch.tensor import Tensor
from nanotorch.nn.module import Module
from nanotorch.nn.loss import BCEWithLogitsLoss, bce_with_logits_loss
from nanotorch.detection.iou import compute_iou_loss_vectorized, ciou
from nanotorch.detection.bbox import _to_numpy


class CIoULoss(Module):
    """Complete IoU Loss for bounding box regression.
    
    CIoU considers:
    1. Overlap area (IoU)
    2. Center distance
    3. Aspect ratio consistency
    
    Loss = 1 - CIoU
    
    Args:
        reduction: 'mean', 'sum', or 'none'
        eps: Small constant for numerical stability
    """
    
    def __init__(self, reduction: str = 'mean', eps: float = 1e-7) -> None:
        super().__init__()
        self.reduction = reduction
        self.eps = eps
    
    def forward(self, pred_boxes: Tensor, target_boxes: Tensor) -> Union[Tensor, float]:
        """Compute CIoU loss.
        
        Args:
            pred_boxes: (N, 4) predicted boxes [x1, y1, x2, y2]
            target_boxes: (N, 4) target boxes [x1, y1, x2, y2]
        
        Returns:
            CIoU loss value
        """
        losses = compute_iou_loss_vectorized(
            pred_boxes, target_boxes,
            iou_type='ciou',
            eps=self.eps
        )
        
        if self.reduction == 'mean':
            return float(np.mean(losses))
        elif self.reduction == 'sum':
            return float(np.sum(losses))
        else:
            return Tensor(losses)


class DFLoss(Module):
    """Distribution Focal Loss for bounding box regression.
    
    Instead of directly regressing box coordinates, DFL predicts a
    distribution over discrete values. The loss encourages the
    distribution to be sharp (low entropy).
    
    DFL = -((y - floor(y)) * log(p_floor) + (ceil(y) - y) * log(p_ceil))
    
    Args:
        reg_max: Maximum regression range (default: 16)
        reduction: 'mean' or 'sum'
    """
    
    def __init__(self, reg_max: int = 16, reduction: str = 'mean') -> None:
        super().__init__()
        self.reg_max = reg_max
        self.reduction = reduction
    
    def forward(
        self,
        pred_dist: Tensor,
        target_boxes: Tensor,
        anchor_points: Tensor,
        stride: int
    ) -> Union[Tensor, float]:
        """Compute DFL loss.
        
        Args:
            pred_dist: (N, 4 * reg_max) predicted distributions
            target_boxes: (N, 4) target boxes in [x1, y1, x2, y2] format
            anchor_points: (N, 2) anchor center points
            stride: Current stride (8, 16, or 32)
        
        Returns:
            DFL loss value
        """
        pred_dist = _to_numpy(pred_dist).astype(np.float32)
        target_boxes = _to_numpy(target_boxes).astype(np.float32)
        anchor_points = _to_numpy(anchor_points).astype(np.float32)
        
        n = pred_dist.shape[0]
        if n == 0:
            return 0.0
        
        # Compute target distances from anchor center
        # target format: [x1, y1, x2, y2]
        # DFL format: [left, top, right, bottom] distances from anchor
        target_lt = anchor_points - target_boxes[:, :2]  # left, top
        target_rb = target_boxes[:, 2:] - anchor_points  # right, bottom
        target = np.concatenate([target_lt, target_rb], axis=1)  # (N, 4)
        
        # Normalize by stride
        target = target / stride
        
        # Reshape pred_dist: (N, 4 * reg_max) -> (N, 4, reg_max)
        pred_dist = pred_dist.reshape(n, 4, self.reg_max)
        
        # Compute DFL loss for each of 4 distances
        loss = 0.0
        count = 0
        
        for i in range(4):
            # Target value for this distance
            t = target[:, i]  # (N,)
            
            # Floor and ceil indices
            t_floor = np.floor(t).astype(np.int32)
            t_ceil = np.minimum(t_floor + 1, self.reg_max - 1)
            
            # Weights
            w_ceil = t - t_floor
            w_floor = 1.0 - w_ceil
            
            # Get predicted probabilities
            # pred_dist[:, i, :] is (N, reg_max)
            pred_i = pred_dist[:, i, :]  # (N, reg_max)
            
            # Softmax
            pred_i = pred_i - pred_i.max(axis=1, keepdims=True)
            exp_pred = np.exp(pred_i)
            pred_softmax = exp_pred / (exp_pred.sum(axis=1, keepdims=True) + 1e-7)
            
            # Gather probabilities at floor and ceil positions
            n_idx = np.arange(n)
            p_floor = pred_softmax[n_idx, np.clip(t_floor, 0, self.reg_max - 1)]
            p_ceil = pred_softmax[n_idx, np.clip(t_ceil, 0, self.reg_max - 1)]
            
            # Cross entropy loss
            loss += -np.mean(w_floor * np.log(p_floor + 1e-7) + 
                            w_ceil * np.log(p_ceil + 1e-7))
            count += 1
        
        return loss / count if count > 0 else 0.0


class VarifocalLoss(Module):
    """Varifocal Loss for classification.
    
    VFL handles class imbalance by using a focal-like weighting:
    - Positive samples: Use target score (IoU) as weight
    - Negative samples: Use predicted score for weighting
    
    VFL = {
        -q * log(p)                    if q > 0 (positive)
        -(1 - q)^gamma * q * log(1-p)  if q = 0 (negative)
    }
    
    Args:
        gamma: Focusing parameter (default: 2.0)
        alpha: Class weight (default: 0.75)
        reduction: 'mean' or 'sum'
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.75,
        reduction: str = 'mean'
    ) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
    
    def forward(
        self,
        pred_logits: Tensor,
        target_labels: Tensor,
        target_scores: Optional[Tensor] = None
    ) -> Union[Tensor, float]:
        """Compute Varifocal loss.
        
        Args:
            pred_logits: (N, num_classes) predicted class logits
            target_labels: (N,) target class indices
            target_scores: (N,) IoU scores for positive samples (optional)
        
        Returns:
            VFL loss value
        """
        pred_logits = _to_numpy(pred_logits).astype(np.float32)
        target_labels = _to_numpy(target_labels).astype(np.int64)
        
        n, num_classes = pred_logits.shape
        
        # Convert to probabilities
        pred_probs = 1.0 / (1.0 + np.exp(-pred_logits))  # sigmoid
        
        # Create one-hot targets
        target_onehot = np.zeros((n, num_classes), dtype=np.float32)
        target_onehot[np.arange(n), target_labels] = 1.0
        
        # If target_scores provided, use them for positive sample weighting
        if target_scores is not None:
            target_scores = _to_numpy(target_scores).astype(np.float32)
            # Scale positive targets by IoU score
            target_onehot = target_onehot * target_scores[:, np.newaxis]
        
        # Varifocal loss computation
        # Positive: -q * log(p)
        # Negative: -(1-q)^gamma * q * log(1-p)
        
        loss_pos = -target_onehot * np.log(pred_probs + 1e-7)
        loss_neg = -(1 - target_onehot) ** self.gamma * target_onehot * np.log(1 - pred_probs + 1e-7)
        
        # Actually for negative samples, we want: -p^gamma * log(1-p)
        loss_neg = -pred_probs ** self.gamma * (1 - target_onehot) * np.log(1 - pred_probs + 1e-7)
        
        loss = self.alpha * loss_pos + (1 - self.alpha) * loss_neg
        
        if self.reduction == 'mean':
            return float(np.mean(loss))
        else:
            return float(np.sum(loss))


class BCELoss(Module):
    """Binary Cross Entropy Loss with logits.
    
    Standard BCE loss for objectness/classification.
    
    Args:
        reduction: 'mean', 'sum', or 'none'
        pos_weight: Weight for positive samples
    """
    
    def __init__(
        self,
        reduction: str = 'mean',
        pos_weight: Optional[float] = None
    ) -> None:
        super().__init__()
        self.reduction = reduction
        self.pos_weight = pos_weight
    
    def forward(self, pred: Tensor, target: Tensor) -> Union[Tensor, float]:
        """Compute BCE loss."""
        pred = _to_numpy(pred).astype(np.float32)
        target = _to_numpy(target).astype(np.float32)
        
        # Sigmoid
        pred_sigmoid = 1.0 / (1.0 + np.exp(-pred))
        
        # BCE
        loss = -target * np.log(pred_sigmoid + 1e-7) - (1 - target) * np.log(1 - pred_sigmoid + 1e-7)
        
        if self.pos_weight is not None:
            loss = np.where(target == 1, loss * self.pos_weight, loss)
        
        if self.reduction == 'mean':
            return float(np.mean(loss))
        elif self.reduction == 'sum':
            return float(np.sum(loss))
        else:
            return Tensor(loss)


class YOLOLoss(Module):
    """Composite YOLO Loss.
    
    Combines multiple loss components:
    1. Box Loss (CIoU): Bounding box regression
    2. Class Loss (BCE/VFL): Classification
    3. DFL Loss: Distribution focal loss for box regression
    
    Args:
        num_classes: Number of object classes
        reg_max: Maximum regression range for DFL (default: 16)
        box_weight: Weight for box loss (default: 7.5)
        cls_weight: Weight for class loss (default: 0.5)
        dfl_weight: Weight for DFL loss (default: 1.5)
        use_vfl: Use Varifocal Loss instead of BCE (default: False)
    """
    
    def __init__(
        self,
        num_classes: int,
        reg_max: int = 16,
        box_weight: float = 7.5,
        cls_weight: float = 0.5,
        dfl_weight: float = 1.5,
        use_vfl: bool = False
    ) -> None:
        super().__init__()
        
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.box_weight = box_weight
        self.cls_weight = cls_weight
        self.dfl_weight = dfl_weight
        
        # Loss functions
        self.box_loss = CIoULoss(reduction='mean')
        self.cls_loss = VarifocalLoss() if use_vfl else BCELoss(reduction='mean')
        self.dfl_loss = DFLoss(reg_max=reg_max)
    
    def forward(
        self,
        predictions: Dict[str, Tuple[Tensor, Tensor]],
        targets: Dict[str, Dict],
        input_size: Tuple[int, int]
    ) -> Tuple[float, Dict[str, float]]:
        """Compute total YOLO loss.
        
        Args:
            predictions: Dict with predictions at each scale
                'p3': (box_pred, cls_pred)
                'p4': (box_pred, cls_pred)
                'p5': (box_pred, cls_pred)
            targets: Dict with target information
                'boxes': (N, 4) target boxes
                'labels': (N,) target labels
                'assigned_{scale}': Dict with assigned targets per scale
            input_size: (H, W) of input image
        
        Returns:
            total_loss: Combined weighted loss
            loss_dict: Dict with individual loss components
        """
        total_box_loss = 0.0
        total_cls_loss = 0.0
        total_dfl_loss = 0.0
        num_samples = 0
        
        strides = {'p3': 8, 'p4': 16, 'p5': 32}
        
        for scale_name, (box_pred, cls_pred) in predictions.items():
            stride = strides[scale_name]
            
            # Get assigned targets for this scale
            assigned_key = f'assigned_{scale_name}'
            if assigned_key not in targets:
                continue
            
            assigned = targets[assigned_key]
            
            # Extract assignment info
            if 'box_targets' not in assigned or len(assigned['box_targets']) == 0:
                continue
            
            target_boxes = assigned['box_targets']  # (num_pos, 4)
            target_labels = assigned['labels']      # (num_pos,)
            anchor_points = assigned['anchor_points']  # (num_pos, 2)
            mask_pos = assigned.get('mask_pos', None)   # (H*W,) or None
            
            num_pos = len(target_boxes)
            if num_pos == 0:
                continue
            
            # Get predictions at positive locations
            # box_pred: (N, 4*reg_max, H, W)
            # cls_pred: (N, num_classes, H, W)
            
            # For simplicity, assume batch size 1
            N, _, H, W = box_pred.shape
            
            # Reshape to (N, H*W, ...)
            box_pred_flat = box_pred.transpose(0, 2, 3, 1).reshape(-1, 4 * self.reg_max)
            cls_pred_flat = cls_pred.transpose(0, 2, 3, 1).reshape(-1, self.num_classes)
            
            # Get positive indices
            pos_idx = assigned.get('pos_indices', None)  # (num_pos,) indices into flat array
            if pos_idx is None:
                continue
            
            # Box loss (CIoU)
            # Decode predictions first
            pred_boxes = self._decode_dfl(box_pred_flat[pos_idx], anchor_points, stride)
            box_loss = self.box_loss(pred_boxes, target_boxes)
            total_box_loss += box_loss * num_pos
            
            # Class loss
            cls_loss = self.cls_loss(
                cls_pred_flat[pos_idx],
                target_labels
            )
            total_cls_loss += cls_loss * num_pos
            
            # DFL loss
            dfl_loss = self.dfl_loss(
                box_pred_flat[pos_idx],
                target_boxes,
                anchor_points,
                stride
            )
            total_dfl_loss += dfl_loss * num_pos
            
            num_samples += num_pos
        
        # Normalize
        if num_samples > 0:
            total_box_loss /= num_samples
            total_cls_loss /= num_samples
            total_dfl_loss /= num_samples
        
        # Combine with weights
        total_loss = (
            self.box_weight * total_box_loss +
            self.cls_weight * total_cls_loss +
            self.dfl_weight * total_dfl_loss
        )
        
        loss_dict = {
            'box_loss': total_box_loss,
            'cls_loss': total_cls_loss,
            'dfl_loss': total_dfl_loss,
            'total_loss': total_loss
        }
        
        return total_loss, loss_dict
    
    def _decode_dfl(
        self,
        pred_dist: np.ndarray,
        anchor_points: np.ndarray,
        stride: int
    ) -> np.ndarray:
        """Decode DFL predictions to bounding boxes.
        
        Args:
            pred_dist: (N, 4 * reg_max) distribution predictions
            anchor_points: (N, 2) anchor center points
            stride: Current stride
        
        Returns:
            (N, 4) boxes in [x1, y1, x2, y2] format
        """
        n = pred_dist.shape[0]
        
        # Reshape and softmax
        pred_dist = pred_dist.reshape(n, 4, self.reg_max)
        pred_dist = pred_dist - pred_dist.max(axis=2, keepdims=True)
        exp_dist = np.exp(pred_dist)
        pred_dist = exp_dist / (exp_dist.sum(axis=2, keepdims=True) + 1e-7)
        
        # Integrate
        arange = np.arange(self.reg_max, dtype=np.float32)
        distances = (pred_dist * arange).sum(axis=2) * stride  # (N, 4)
        
        # Convert to boxes
        # [left, top, right, bottom] -> [x1, y1, x2, y2]
        x1 = anchor_points[:, 0] - distances[:, 0]
        y1 = anchor_points[:, 1] - distances[:, 1]
        x2 = anchor_points[:, 0] + distances[:, 2]
        y2 = anchor_points[:, 1] + distances[:, 3]
        
        return np.stack([x1, y1, x2, y2], axis=1)


class SimpleYOLOLoss(Module):
    """Simplified YOLO Loss for training without complex assignment.
    
    This is a simpler version that can be used for testing or
    basic training without sophisticated target assignment.
    
    Args:
        num_classes: Number of object classes
        reg_max: Maximum regression range
    """
    
    def __init__(
        self,
        num_classes: int,
        reg_max: int = 16
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.box_loss = CIoULoss()
        self.cls_loss = BCELoss()
    
    def forward(
        self,
        pred_boxes: Tensor,
        pred_classes: Tensor,
        target_boxes: Tensor,
        target_classes: Tensor
    ) -> Tuple[float, Dict[str, float]]:
        """Simple forward with direct predictions and targets.
        
        Args:
            pred_boxes: (N, 4) predicted boxes
            pred_classes: (N, num_classes) predicted class logits
            target_boxes: (N, 4) target boxes
            target_classes: (N,) target class indices
        
        Returns:
            total_loss and loss_dict
        """
        # Box loss
        box_loss = self.box_loss(pred_boxes, target_boxes)
        
        # Class loss - convert to one-hot
        target_classes = _to_numpy(target_classes).astype(np.int64)
        n = pred_classes.shape[0]
        target_onehot = np.zeros((n, self.num_classes), dtype=np.float32)
        target_onehot[np.arange(n), target_classes] = 1.0
        
        cls_loss = self.cls_loss(pred_classes, Tensor(target_onehot))
        
        total_loss = 7.5 * box_loss + 0.5 * cls_loss
        
        return total_loss, {
            'box_loss': box_loss,
            'cls_loss': cls_loss,
            'total_loss': total_loss
        }
