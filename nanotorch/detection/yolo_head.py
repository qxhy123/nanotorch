"""
YOLO v12 Detection Head.

This module implements the anchor-free detection head for YOLO v12:
- Decoupled head for classification and regression
- DFL (Distribution Focal Loss) for bounding box regression
- Anchor-free detection with center-based prediction

Key concepts:
- Anchor-free: No predefined anchor boxes, predicts from grid centers
- Decoupled: Separate branches for box and class predictions
- DFL: Distribution-based regression for better localization
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from nanotorch.tensor import Tensor
from nanotorch.nn.module import Module, Sequential
from nanotorch.nn.conv import Conv2D
from nanotorch.nn.batchnorm import BatchNorm2d
from nanotorch.nn.activation import SiLU, Sigmoid
from nanotorch.detection.layers import Conv, C2f, make_divisible
from nanotorch.detection.bbox import xywh_to_xyxy


class DFLHead(Module):
    """Distribution Focal Loss (DFL) head for bounding box regression.
    
    Instead of directly predicting box coordinates, DFL predicts a
    distribution over discrete values, providing better localization
    for small objects and uncertain boundaries.
    
    Args:
        in_channels: Number of input channels
        reg_max: Maximum regression range (default: 16)
    
    Shape:
        - Input: (N, C, H, W)
        - Output: (N, 4 * reg_max, H, W)
    """
    
    def __init__(self, in_channels: int, reg_max: int = 16) -> None:
        super().__init__()
        
        self.reg_max = reg_max
        
        # DFL convolution: predict distribution for each of 4 box params
        self.conv = Conv2D(in_channels, 4 * reg_max, kernel_size=1, stride=1)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass returning distribution predictions."""
        return self.conv(x)
    
    def integrate(self, pred: Tensor) -> Tensor:
        """Integrate distribution to get box regression values.
        
        Args:
            pred: (N, 4 * reg_max, H, W) distribution predictions
        
        Returns:
            (N, 4, H, W) integrated regression values
        """
        N, _, H, W = pred.shape
        
        # Reshape to (N, 4, reg_max, H, W)
        pred = pred.reshape(N, 4, self.reg_max, H, W)
        
        # Softmax over reg_max dimension
        pred = pred.softmax(dim=2)
        
        # Integrate: sum(pred * [0, 1, 2, ..., reg_max-1])
        arange = np.arange(self.reg_max, dtype=np.float32)
        arange = Tensor(arange.reshape(1, 1, self.reg_max, 1, 1))
        
        integrated = (pred * arange).sum(dim=2)  # (N, 4, H, W)
        
        return integrated


class ClassHead(Module):
    """Classification head for object detection.
    
    Predicts class probabilities for each spatial location.
    
    Args:
        in_channels: Number of input channels
        num_classes: Number of object classes
    
    Shape:
        - Input: (N, C, H, W)
        - Output: (N, num_classes, H, W)
    """
    
    def __init__(self, in_channels: int, num_classes: int) -> None:
        super().__init__()
        
        self.conv = Conv2D(in_channels, num_classes, kernel_size=1, stride=1)
    
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass returning class logits."""
        return self.conv(x)


class DecoupledHead(Module):
    """Decoupled detection head with separate box and class branches.
    
    Separates regression and classification into independent branches
    for better feature specialization.
    
    Structure:
    - Shared stem conv
    - Box branch: Conv -> DFL head -> box prediction
    - Class branch: Conv -> Class head -> class prediction
    
    Args:
        in_channels: Number of input channels
        num_classes: Number of object classes
        reg_max: Maximum regression range for DFL (default: 16)
        hidden_channels: Hidden channels in branches (default: None)
    """
    
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        reg_max: int = 16,
        hidden_channels: Optional[int] = None
    ) -> None:
        super().__init__()
        
        if hidden_channels is None:
            hidden_channels = in_channels
        
        # Shared stem
        self.stem = Conv(in_channels, hidden_channels, kernel_size=3)
        
        # Box regression branch
        self.box_conv = Conv(hidden_channels, hidden_channels, kernel_size=3)
        self.box_dfl = DFLHead(hidden_channels, reg_max)
        
        # Classification branch
        self.cls_conv = Conv(hidden_channels, hidden_channels, kernel_size=3)
        self.cls_head = ClassHead(hidden_channels, num_classes)
        
        self.reg_max = reg_max
        self.num_classes = num_classes
    
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass returning box and class predictions.
        
        Returns:
            box_pred: (N, 4 * reg_max, H, W) distribution predictions
            cls_pred: (N, num_classes, H, W) class logits
        """
        # Shared stem
        x = self.stem(x)
        
        # Box branch
        box_feat = self.box_conv(x)
        box_pred = self.box_dfl(box_feat)
        
        # Class branch
        cls_feat = self.cls_conv(x)
        cls_pred = self.cls_head(cls_feat)
        
        return box_pred, cls_pred


class YOLOHead(Module):
    """YOLO v12 Detection Head.
    
    Multi-scale detection head with decoupled predictions at each scale.
    This is an anchor-free head that predicts from grid centers.
    
    Outputs at three scales (P3, P4, P5) with strides 8, 16, 32.
    
    Args:
        in_channels: Dict of input channels from neck {'p3': c3, 'p4': c4, 'p5': c5}
        num_classes: Number of object classes
        reg_max: Maximum regression range for DFL (default: 16)
        num_layers: Number of conv layers before head (default: 2)
    
    Shape:
        - Input: Dict with 'p3', 'p4', 'p5' feature maps from neck
        - Output: Dict with predictions at each scale
            - 'p3': ((N, 4*reg_max, H/8, W/8), (N, num_classes, H/8, W/8))
            - 'p4': ((N, 4*reg_max, H/16, W/16), (N, num_classes, H/16, W/16))
            - 'p5': ((N, 4*reg_max, H/32, W/32), (N, num_classes, H/32, W/32))
    """
    
    def __init__(
        self,
        in_channels: Dict[str, int],
        num_classes: int,
        reg_max: int = 16,
        num_layers: int = 2
    ) -> None:
        super().__init__()
        
        self.num_classes = num_classes
        self.reg_max = reg_max
        self.num_layers = num_layers
        
        # Decoupled heads for each scale
        self.heads = {
            'p3': DecoupledHead(in_channels['p3'], num_classes, reg_max),
            'p4': DecoupledHead(in_channels['p4'], num_classes, reg_max),
            'p5': DecoupledHead(in_channels['p5'], num_classes, reg_max),
        }
        
        # Strides for each scale
        self.strides = {'p3': 8, 'p4': 16, 'p5': 32}
        
        # Grid caches (regenerated for each input size)
        self._grids: Dict[str, Optional[Tensor]] = {
            'p3': None, 'p4': None, 'p5': None
        }
    
    def _make_grid(self, nx: int, ny: int, stride: int) -> Tensor:
        """Generate grid coordinates for anchor-free detection.
        
        Args:
            nx: Grid width
            ny: Grid height
            stride: Stride relative to input image
        
        Returns:
            (1, 2, ny, nx) tensor with (x, y) grid coordinates
        """
        yv, xv = np.meshgrid(
            np.arange(ny, dtype=np.float32),
            np.arange(nx, dtype=np.float32),
            indexing='ij'
        )
        grid = np.stack([xv, yv], axis=0)  # (2, ny, nx)
        grid = grid[np.newaxis, :, :, :]  # (1, 2, ny, nx)
        
        return Tensor(grid + 0.5)  # Center of grid cell
    
    def forward(self, features: Dict[str, Tensor]) -> Dict[str, Tuple[Tensor, Tensor]]:
        """Forward pass returning predictions at each scale."""
        predictions = {}
        
        for name, feat in features.items():
            box_pred, cls_pred = self.heads[name](feat)
            predictions[name] = (box_pred, cls_pred)
        
        return predictions
    
    def decode_predictions(
        self,
        predictions: Dict[str, Tuple[Tensor, Tensor]],
        input_size: Tuple[int, int],
        conf_threshold: float = 0.25
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Decode raw predictions to bounding boxes.
        
        Converts DFL distribution to box coordinates and applies
        confidence thresholding.
        
        Args:
            predictions: Raw predictions from forward()
            input_size: (H, W) of input image
            conf_threshold: Minimum confidence threshold
        
        Returns:
            boxes: (N, 4) array of [x1, y1, x2, y2] boxes
            scores: (N,) array of confidence scores
            class_ids: (N,) array of class indices
        """
        all_boxes = []
        all_scores = []
        all_class_ids = []
        
        for name, (box_pred, cls_pred) in predictions.items():
            stride = self.strides[name]
            
            N, _, ny, nx = box_pred.shape
            
            # Integrate DFL distribution to get regression values
            # box_pred: (N, 4*reg_max, ny, nx) -> (N, 4, ny, nx)
            box_reg = self.heads[name].box_dfl.integrate(box_pred)
            
            # Generate grid
            grid = self._make_grid(nx, ny, stride)  # (1, 2, ny, nx)
            
            # Decode boxes (anchor-free format)
            # box_reg format: [left, top, right, bottom] distances from grid center
            # Final box: [cx - left, cy - top, cx + right, cy + bottom]
            
            # Reshape for broadcasting
            box_reg = box_reg.data  # (N, 4, ny, nx)
            grid_x = grid.data[0, 0, :, :]  # (ny, nx)
            grid_y = grid.data[0, 1, :, :]  # (ny, nx)
            
            # Decode: grid center + regression * stride
            # DFL predicts distances from center
            cx = grid_x * stride  # (ny, nx)
            cy = grid_y * stride  # (ny, nx)
            
            # box_reg: [dl, dt, dr, db] (left, top, right, bottom distances)
            dl = box_reg[:, 0, :, :] * stride  # (N, ny, nx)
            dt = box_reg[:, 1, :, :] * stride
            dr = box_reg[:, 2, :, :] * stride
            db = box_reg[:, 3, :, :] * stride
            
            # Compute final box coordinates
            x1 = cx[np.newaxis, :, :] - dl
            y1 = cy[np.newaxis, :, :] - dt
            x2 = cx[np.newaxis, :, :] + dr
            y2 = cy[np.newaxis, :, :] + db
            
            # Flatten spatial dimensions
            x1 = x1.reshape(N, -1)
            y1 = y1.reshape(N, -1)
            x2 = x2.reshape(N, -1)
            y2 = y2.reshape(N, -1)
            
            # Apply sigmoid to class predictions and get max
            cls_scores = 1.0 / (1.0 + np.exp(-cls_pred.data))  # sigmoid
            scores = cls_scores.max(axis=1)  # (N, ny*nx)
            class_ids = cls_scores.argmax(axis=1)  # (N, ny*nx)
            
            # Apply confidence threshold
            for n in range(N):
                mask = scores[n] >= conf_threshold
                if mask.any():
                    batch_boxes = np.stack([
                        x1[n, mask], y1[n, mask],
                        x2[n, mask], y2[n, mask]
                    ], axis=1)
                    all_boxes.append(batch_boxes)
                    all_scores.append(scores[n, mask])
                    all_class_ids.append(class_ids[n, mask])
        
        if len(all_boxes) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Concatenate all predictions
        boxes = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        class_ids = np.concatenate(all_class_ids, axis=0)
        
        return boxes, scores, class_ids


class AnchorHead(Module):
    """Anchor-based detection head (for comparison/legacy support).
    
    Traditional anchor-based detection with predefined anchor boxes.
    
    Args:
        in_channels: Dict of input channels
        num_classes: Number of object classes
        num_anchors: Number of anchors per location (default: 3)
    """
    
    def __init__(
        self,
        in_channels: Dict[str, int],
        num_classes: int,
        num_anchors: int = 3
    ) -> None:
        super().__init__()
        
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Output: 4 (bbox) + 1 (obj) + num_classes
        self.output_channels = 4 + 1 + num_classes
        
        # Conv heads for each scale
        self.heads = {}
        for name, channels in in_channels.items():
            self.heads[name] = Conv2D(
                channels,
                num_anchors * self.output_channels,
                kernel_size=1
            )
        
        self.strides = {'p3': 8, 'p4': 16, 'p5': 32}
    
    def forward(self, features: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Forward pass returning raw predictions."""
        predictions = {}
        for name, feat in features.items():
            predictions[name] = self.heads[name](feat)
        return predictions


def build_head(
    head_type: str = 'decoupled',
    in_channels: Optional[Dict[str, int]] = None,
    num_classes: int = 80,
    reg_max: int = 16
) -> Union[YOLOHead, AnchorHead]:
    """Build YOLO detection head.
    
    Args:
        head_type: Type of head ('decoupled' or 'anchor')
        in_channels: Input channels from neck
        num_classes: Number of object classes
        reg_max: Maximum regression range for DFL
    
    Returns:
        Configured detection head
    """
    if in_channels is None:
        # Default channels for YOLO-s
        in_channels = {'p3': 128, 'p4': 256, 'p5': 512}
    
    if head_type == 'decoupled':
        return YOLOHead(in_channels, num_classes, reg_max)
    elif head_type == 'anchor':
        return AnchorHead(in_channels, num_classes)
    else:
        raise ValueError(f"Unknown head type: {head_type}")
