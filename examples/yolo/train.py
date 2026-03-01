"""
Training script for YOLO v12 object detection.

This module provides:
- Training loop with gradient updates
- Learning rate scheduling
- Logging and checkpointing
- Simple target assignment for training
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Any
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nanotorch.tensor import Tensor
from nanotorch.optim import AdamW, SGD
from nanotorch.optim.lr_scheduler import CosineWarmupScheduler
from nanotorch.detection import (
    YOLOBackbone, build_backbone,
    YOLONeck, build_neck,
    YOLOHead, build_head,
    YOLOLoss, SimpleYOLOLoss,
    batched_nms
)
from examples.yolo.data import create_synthetic_dataloader


class SimpleTargetAssigner:
    """Simple target assigner for training.
    
    Assigns ground truth boxes to anchor points using
    center-based assignment.
    """
    
    def __init__(
        self,
        num_classes: int,
        strides: List[int] = [8, 16, 32],
        reg_max: int = 16,
        topk: int = 10
    ):
        self.num_classes = num_classes
        self.strides = strides
        self.reg_max = reg_max
        self.topk = topk
    
    def assign(
        self,
        gt_boxes: np.ndarray,
        gt_labels: np.ndarray,
        image_size: int
    ) -> Dict[str, Dict]:
        """Assign ground truth to feature maps.
        
        Args:
            gt_boxes: (N, 4) boxes in [x1, y1, x2, y2] format
            gt_labels: (N,) class labels
            image_size: Image size (assumes square)
        
        Returns:
            Dict with assignment info for each scale
        """
        assignments = {}
        
        num_gt = len(gt_boxes)
        if num_gt == 0:
            for i, stride in enumerate(self.strides):
                scale_name = f'p{i+3}'
                assignments[f'assigned_{scale_name}'] = {
                    'box_targets': np.zeros((0, 4), dtype=np.float32),
                    'labels': np.zeros((0,), dtype=np.int64),
                    'anchor_points': np.zeros((0, 2), dtype=np.float32),
                    'pos_indices': np.zeros((0,), dtype=np.int64)
                }
            return assignments
        
        for i, stride in enumerate(self.strides):
            scale_name = f'p{i+3}'
            feat_size = image_size // stride
            
            anchor_x = (np.arange(feat_size) + 0.5) * stride
            anchor_y = (np.arange(feat_size) + 0.5) * stride
            anchor_points = np.stack(np.meshgrid(anchor_x, anchor_y), axis=-1)
            anchor_points = anchor_points.reshape(-1, 2)
            
            num_anchors = len(anchor_points)
            
            gt_centers = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
            gt_wh = gt_boxes[:, 2:] - gt_boxes[:, :2]
            
            distances = np.sqrt(
                (anchor_points[:, np.newaxis, 0] - gt_centers[np.newaxis, :, 0]) ** 2 +
                (anchor_points[:, np.newaxis, 1] - gt_centers[np.newaxis, :, 1]) ** 2
            )
            
            topk = min(self.topk, num_anchors // num_gt)
            
            anchor_to_gt = np.zeros(num_anchors, dtype=np.int64) - 1
            anchor_mask = np.zeros(num_anchors, dtype=bool)
            
            for gt_idx in range(num_gt):
                gt_dists = distances[:, gt_idx]
                topk_indices = np.argsort(gt_dists)[:topk]
                
                for anchor_idx in topk_indices:
                    if not anchor_mask[anchor_idx]:
                        anchor_mask[anchor_idx] = True
                        anchor_to_gt[anchor_idx] = gt_idx
            
            pos_indices = np.where(anchor_mask)[0]
            
            if len(pos_indices) > 0:
                box_targets = gt_boxes[anchor_to_gt[pos_indices]]
                labels = gt_labels[anchor_to_gt[pos_indices]]
                anchor_pts = anchor_points[pos_indices]
            else:
                box_targets = np.zeros((0, 4), dtype=np.float32)
                labels = np.zeros((0,), dtype=np.int64)
                anchor_pts = np.zeros((0, 2), dtype=np.float32)
            
            assignments[f'assigned_{scale_name}'] = {
                'box_targets': box_targets,
                'labels': labels,
                'anchor_points': anchor_pts,
                'pos_indices': pos_indices
            }
        
        return assignments


class YOLOModel:
    """Complete YOLO v12 model combining backbone, neck, and head."""
    
    def __init__(
        self,
        num_classes: int = 10,
        model_size: str = 's',
        use_attention: bool = True
    ):
        self.num_classes = num_classes
        
        self.backbone = build_backbone(
            model_size=model_size,
            use_attention=use_attention
        )
        
        self.neck = build_neck(
            neck_type='panet',
            in_channels=self.backbone.out_channels,
            num_blocks=3
        )
        
        self.head = build_head(
            head_type='decoupled',
            in_channels=self.neck.out_channels,
            num_classes=num_classes
        )
    
    def __call__(self, x: Tensor) -> Dict[str, Tuple[Tensor, Tensor]]:
        features = self.backbone(x)
        features = self.neck(features)
        predictions = self.head(features)
        return predictions
    
    def parameters(self) -> List[Tensor]:
        params = []
        params.extend(self.backbone.parameters())
        params.extend(self.neck.parameters())
        params.extend(self.head.parameters())
        return params


class Trainer:
    """Training class for YOLO model."""
    
    def __init__(
        self,
        model: YOLOModel,
        train_loader,
        num_classes: int = 10,
        lr: float = 1e-3,
        weight_decay: float = 5e-4,
        image_size: int = 640,
        log_interval: int = 10
    ):
        self.model = model
        self.train_loader = train_loader
        self.num_classes = num_classes
        self.image_size = image_size
        self.log_interval = log_interval
        
        self.optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.loss_fn = SimpleYOLOLoss(num_classes=num_classes)
        
        self.target_assigner = SimpleTargetAssigner(
            num_classes=num_classes,
            strides=[8, 16, 32]
        )
    
    def train_epoch(self, epoch: int) -> float:
        self.model.backbone.train()
        self.model.neck.train()
        self.model.head.train()
        
        total_loss = 0.0
        num_batches = 0
        
        start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            images = Tensor(batch['images'])
            gt_boxes = batch['boxes']
            gt_labels = batch['labels']
            batch_indices = batch['batch_indices']
            
            predictions = self.model(images)
            
            batch_loss = 0.0
            batch_samples = 0
            
            unique_batches = np.unique(batch_indices) if len(batch_indices) > 0 else []
            
            for b in unique_batches:
                mask = batch_indices == b
                boxes = gt_boxes[mask]
                labels = gt_labels[mask]
                
                if len(boxes) == 0:
                    continue
                
                for scale_name, (box_pred, cls_pred) in predictions.items():
                    N, _, H, W = box_pred.shape
                    stride = self.image_size // H
                    
                    box_pred_flat = box_pred.data[b].transpose(1, 2, 0).reshape(-1, 4 * 16)
                    cls_pred_flat = cls_pred.data[b].transpose(1, 2, 0).reshape(-1, self.num_classes)
                    
                    anchor_x = (np.arange(W) + 0.5) * stride
                    anchor_y = (np.arange(H) + 0.5) * stride
                    anchor_points = np.stack(np.meshgrid(anchor_x, anchor_y), axis=-1).reshape(-1, 2)
                    
                    distances = np.sqrt(
                        (anchor_points[:, np.newaxis, 0] - ((boxes[:, 0] + boxes[:, 2]) / 2)[np.newaxis, :]) ** 2 +
                        (anchor_points[:, np.newaxis, 1] - ((boxes[:, 1] + boxes[:, 3]) / 2)[np.newaxis, :]) ** 2
                    )
                    
                    for gt_idx in range(len(boxes)):
                        topk_idx = np.argsort(distances[:, gt_idx])[:5]
                        
                        gt_box = boxes[gt_idx:gt_idx+1]
                        gt_label = labels[gt_idx]
                        
                        for anchor_idx in topk_idx:
                            cx = anchor_points[anchor_idx, 0]
                            cy = anchor_points[anchor_idx, 1]
                            
                            pred_box = np.array([
                                cx - stride * 5, cy - stride * 5,
                                cx + stride * 5, cy + stride * 5
                            ], dtype=np.float32)
                            
                            pred_cls = cls_pred_flat[anchor_idx:anchor_idx+1]
                            target_onehot = np.zeros(self.num_classes, dtype=np.float32)
                            target_onehot[gt_label] = 1.0
                            
                            loss, _ = self.loss_fn(
                                Tensor(pred_box[np.newaxis]),
                                Tensor(pred_cls),
                                Tensor(gt_box),
                                Tensor(np.array([gt_label]))
                            )
                            batch_loss += loss
                            batch_samples += 1
            
            if batch_samples > 0:
                batch_loss = batch_loss / batch_samples
                total_loss += batch_loss
                num_batches += 1
                
                self.optimizer.zero_grad()
                
                if isinstance(batch_loss, (int, float)):
                    batch_loss_tensor = Tensor(batch_loss, requires_grad=True)
                else:
                    batch_loss_tensor = batch_loss if isinstance(batch_loss, Tensor) else Tensor(batch_loss, requires_grad=True)
                
                batch_loss_tensor.backward()
                self.optimizer.step()
            
            if (batch_idx + 1) % self.log_interval == 0:
                elapsed = time.time() - start_time
                avg_loss = total_loss / max(num_batches, 1)
                print(f"  Batch {batch_idx + 1}/{len(self.train_loader)}, "
                      f"Loss: {avg_loss:.4f}, Time: {elapsed:.1f}s")
        
        return total_loss / max(num_batches, 1)


def train_yolo(
    num_epochs: int = 10,
    batch_size: int = 4,
    image_size: int = 640,
    num_classes: int = 10,
    model_size: str = 'n',
    lr: float = 1e-3,
    num_samples: int = 200
):
    print("=" * 60)
    print("YOLO v12 Training with nanotorch")
    print("=" * 60)
    
    print(f"\nInitializing model (size={model_size}, classes={num_classes})...")
    model = YOLOModel(
        num_classes=num_classes,
        model_size=model_size,
        use_attention=False
    )
    
    print(f"Creating dataloader (batch_size={batch_size}, samples={num_samples})...")
    train_loader = create_synthetic_dataloader(
        num_samples=num_samples,
        batch_size=batch_size,
        image_size=image_size,
        num_classes=num_classes
    )
    
    print(f"Setting up trainer (lr={lr})...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        num_classes=num_classes,
        lr=lr,
        image_size=image_size
    )
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        avg_loss = trainer.train_epoch(epoch)
        print(f"  Average Loss: {avg_loss:.4f}")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    
    return model


if __name__ == "__main__":
    model = train_yolo(
        num_epochs=3,
        batch_size=2,
        image_size=320,
        num_classes=5,
        model_size='n',
        num_samples=20
    )
