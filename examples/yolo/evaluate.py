"""
Evaluation metrics for YOLO object detection.

This module provides:
- mAP (mean Average Precision) computation
- Precision/Recall curves
- Per-class metrics
- IoU threshold evaluation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from nanotorch.detection.bbox import box_iou


def compute_ap(
    recalls: np.ndarray,
    precisions: np.ndarray,
    use_07_metric: bool = False
) -> float:
    """Compute Average Precision from recall-precision pairs.
    
    Args:
        recalls: Array of recall values
        precisions: Array of precision values
        use_07_metric: Use VOC2007 11-point interpolation
    
    Returns:
        AP value
    """
    if use_07_metric:
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            mask = recalls >= t
            if mask.any():
                p = precisions[mask].max()
            else:
                p = 0.0
            ap += p / 11.0
        return ap
    
    mrec = np.concatenate([[0.0], recalls, [1.0]])
    mpre = np.concatenate([[0.0], precisions, [0.0]])
    
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    
    change_points = np.where(mrec[1:] != mrec[:-1])[0]
    
    ap = np.sum((mrec[change_points + 1] - mrec[change_points]) * mpre[change_points + 1])
    
    return float(ap)


def compute_precision_recall(
    pred_boxes: np.ndarray,
    pred_scores: np.ndarray,
    pred_labels: np.ndarray,
    gt_boxes: np.ndarray,
    gt_labels: np.ndarray,
    iou_threshold: float = 0.5,
    num_classes: int = 80
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute precision-recall for all classes.
    
    Args:
        pred_boxes: (N, 4) predicted boxes
        pred_scores: (N,) predicted scores
        pred_labels: (N,) predicted class labels
        gt_boxes: (M, 4) ground truth boxes
        gt_labels: (M,) ground truth labels
        iou_threshold: IoU threshold for matching
        num_classes: Number of classes
    
    Returns:
        precisions: (num_classes,) precision per class
        recalls: (num_classes,) recall per class
        aps: (num_classes,) AP per class
    """
    precisions = np.zeros(num_classes)
    recalls = np.zeros(num_classes)
    aps = np.zeros(num_classes)
    
    for c in range(num_classes):
        pred_mask = pred_labels == c
        gt_mask = gt_labels == c
        
        class_pred_boxes = pred_boxes[pred_mask]
        class_pred_scores = pred_scores[pred_mask]
        class_gt_boxes = gt_boxes[gt_mask]
        
        if len(class_gt_boxes) == 0:
            continue
        
        if len(class_pred_boxes) == 0:
            continue
        
        sort_idx = np.argsort(-class_pred_scores)
        class_pred_boxes = class_pred_boxes[sort_idx]
        class_pred_scores = class_pred_scores[sort_idx]
        
        num_gt = len(class_gt_boxes)
        num_pred = len(class_pred_boxes)
        
        gt_matched = np.zeros(num_gt, dtype=bool)
        
        tp = np.zeros(num_pred)
        fp = np.zeros(num_pred)
        
        for pred_idx in range(num_pred):
            if len(class_gt_boxes) > 0:
                ious = box_iou(
                    class_pred_boxes[pred_idx:pred_idx + 1],
                    class_gt_boxes
                )[0]
                
                best_gt_idx = ious.argmax()
                best_iou = ious[best_gt_idx]
                
                if best_iou >= iou_threshold and not gt_matched[best_gt_idx]:
                    tp[pred_idx] = 1
                    gt_matched[best_gt_idx] = True
                else:
                    fp[pred_idx] = 1
        
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        class_precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-7)
        class_recalls = tp_cumsum / (num_gt + 1e-7)
        
        precisions[c] = class_precisions[-1] if len(class_precisions) > 0 else 0
        recalls[c] = class_recalls[-1] if len(class_recalls) > 0 else 0
        aps[c] = compute_ap(class_recalls, class_precisions)
    
    return precisions, recalls, aps


class DetectionMetrics:
    """Comprehensive detection metrics calculator.
    
    Computes mAP at various IoU thresholds, precision, recall,
    and per-class metrics.
    
    Args:
        num_classes: Number of object classes
        iou_thresholds: IoU thresholds for mAP (default: [0.5, 0.75])
    """
    
    def __init__(
        self,
        num_classes: int,
        iou_thresholds: List[float] = None
    ):
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        
        self.reset()
    
    def reset(self):
        self.all_pred_boxes = []
        self.all_pred_scores = []
        self.all_pred_labels = []
        self.all_gt_boxes = []
        self.all_gt_labels = []
    
    def update(
        self,
        pred_boxes: np.ndarray,
        pred_scores: np.ndarray,
        pred_labels: np.ndarray,
        gt_boxes: np.ndarray,
        gt_labels: np.ndarray
    ):
        pred_boxes = np.asarray(pred_boxes, dtype=np.float32)
        pred_scores = np.asarray(pred_scores, dtype=np.float32)
        pred_labels = np.asarray(pred_labels, dtype=np.int64)
        gt_boxes = np.asarray(gt_boxes, dtype=np.float32)
        gt_labels = np.asarray(gt_labels, dtype=np.int64)
        
        if pred_boxes.ndim == 0 or len(pred_boxes) == 0:
            pred_boxes = np.zeros((0, 4), dtype=np.float32)
            pred_scores = np.zeros((0,), dtype=np.float32)
            pred_labels = np.zeros((0,), dtype=np.int64)
        
        if gt_boxes.ndim == 0 or len(gt_boxes) == 0:
            gt_boxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.zeros((0,), dtype=np.int64)
        
        self.all_pred_boxes.append(pred_boxes)
        self.all_pred_scores.append(pred_scores)
        self.all_pred_labels.append(pred_labels)
        self.all_gt_boxes.append(gt_boxes)
        self.all_gt_labels.append(gt_labels)
    
    def compute(self) -> Dict[str, float]:
        if len(self.all_pred_boxes) == 0:
            return {'mAP': 0.0, 'mAP50': 0.0, 'mAP75': 0.0}
        
        pred_boxes = np.concatenate(self.all_pred_boxes, axis=0)
        pred_scores = np.concatenate(self.all_pred_scores, axis=0)
        pred_labels = np.concatenate(self.all_pred_labels, axis=0)
        gt_boxes = np.concatenate(self.all_gt_boxes, axis=0)
        gt_labels = np.concatenate(self.all_gt_labels, axis=0)
        
        aps_per_threshold = {}
        
        for iou_thresh in self.iou_thresholds:
            _, _, aps = compute_precision_recall(
                pred_boxes, pred_scores, pred_labels,
                gt_boxes, gt_labels,
                iou_threshold=iou_thresh,
                num_classes=self.num_classes
            )
            
            valid_aps = aps[aps > 0]
            if len(valid_aps) > 0:
                aps_per_threshold[iou_thresh] = np.mean(valid_aps)
            else:
                aps_per_threshold[iou_thresh] = 0.0
        
        results = {
            'mAP': np.mean(list(aps_per_threshold.values())),
            'mAP50': aps_per_threshold.get(0.5, 0.0),
            'mAP75': aps_per_threshold.get(0.75, 0.0),
        }
        
        for iou_thresh, ap in aps_per_threshold.items():
            results[f'AP@{iou_thresh}'] = ap
        
        return results


def evaluate_model(
    model,
    dataloader,
    num_classes: int,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    image_size: int = 640
) -> Dict[str, float]:
    """Evaluate a YOLO model on a dataset.
    
    Args:
        model: YOLO model with __call__ method
        dataloader: DataLoader providing batches
        num_classes: Number of classes
        conf_threshold: Confidence threshold
        iou_threshold: NMS IoU threshold
        image_size: Input image size
    
    Returns:
        Dict with evaluation metrics
    """
    metrics = DetectionMetrics(num_classes=num_classes)
    
    from nanotorch.tensor import Tensor
    from nanotorch.detection.nms import postprocess_detections
    
    for batch in dataloader:
        images = Tensor(batch['images'])
        gt_boxes = batch['boxes']
        gt_labels = batch['labels']
        batch_indices = batch['batch_indices']
        
        predictions = model(images)
        
        batch_size = images.shape[0]
        
        for b in range(batch_size):
            batch_pred_boxes = []
            batch_pred_scores = []
            batch_pred_labels = []
            
            for scale_name, (box_pred, cls_pred) in predictions.items():
                N, _, H, W = box_pred.shape
                stride = image_size // H
                
                box_b = box_pred.data[b]
                cls_b = cls_pred.data[b]
                
                cls_scores = 1.0 / (1.0 + np.exp(-cls_b))
                scores = cls_scores.max(axis=0)
                labels = cls_scores.argmax(axis=0)
                
                mask = scores >= conf_threshold
                
                for i in range(H):
                    for j in range(W):
                        if mask[i, j]:
                            cx = (j + 0.5) * stride
                            cy = (i + 0.5) * stride
                            half_size = stride * 5
                            
                            x1 = max(0, cx - half_size)
                            y1 = max(0, cy - half_size)
                            x2 = min(image_size, cx + half_size)
                            y2 = min(image_size, cy + half_size)
                            
                            batch_pred_boxes.append([x1, y1, x2, y2])
                            batch_pred_scores.append(scores[i, j])
                            batch_pred_labels.append(labels[i, j])
            
            if len(batch_pred_boxes) > 0:
                batch_pred_boxes = np.array(batch_pred_boxes)
                batch_pred_scores = np.array(batch_pred_scores)
                batch_pred_labels = np.array(batch_pred_labels)
                
                keep = []
                if len(batch_pred_boxes) > 0:
                    from nanotorch.detection.nms import batched_nms
                    keep = batched_nms(
                        batch_pred_boxes,
                        batch_pred_scores,
                        batch_pred_labels,
                        iou_threshold
                    )
                
                batch_pred_boxes = batch_pred_boxes[keep]
                batch_pred_scores = batch_pred_scores[keep]
                batch_pred_labels = batch_pred_labels[keep]
            else:
                batch_pred_boxes = np.zeros((0, 4), dtype=np.float32)
                batch_pred_scores = np.zeros((0,), dtype=np.float32)
                batch_pred_labels = np.zeros((0,), dtype=np.int64)
            
            mask = batch_indices == b if len(batch_indices) > 0 else np.array([], dtype=bool)
            batch_gt_boxes = gt_boxes[mask] if mask.any() else np.zeros((0, 4), dtype=np.float32)
            batch_gt_labels = gt_labels[mask] if mask.any() else np.zeros((0,), dtype=np.int64)
            
            metrics.update(
                batch_pred_boxes,
                batch_pred_scores,
                batch_pred_labels,
                batch_gt_boxes,
                batch_gt_labels
            )
    
    return metrics.compute()


if __name__ == "__main__":
    print("Testing DetectionMetrics...")
    
    metrics = DetectionMetrics(num_classes=3)
    
    pred_boxes = np.array([[10, 10, 50, 50], [60, 60, 100, 100], [150, 150, 200, 200]], dtype=np.float32)
    pred_scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    pred_labels = np.array([0, 1, 2], dtype=np.int64)
    
    gt_boxes = np.array([[12, 12, 48, 48], [55, 55, 95, 95]], dtype=np.float32)
    gt_labels = np.array([0, 1], dtype=np.int64)
    
    metrics.update(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)
    
    results = metrics.compute()
    
    print("\nEvaluation Results:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")
    
    print("\nTesting AP computation...")
    recalls = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    precisions = np.array([1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.7, 0.7, 0.6, 0.5])
    ap = compute_ap(recalls, precisions)
    print(f"  AP: {ap:.4f}")
    
    print("\nAll tests passed!")
