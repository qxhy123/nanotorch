"""
Data loading and preprocessing utilities for YOLO v1.

This module provides:
- Synthetic dataset for testing
- Data augmentation
- Target encoding utilities
- DataLoader helpers
"""

import numpy as np
from typing import List, Tuple, Dict, Any

from nanotorch.data import Dataset, DataLoader


class SyntheticVOCDataset(Dataset):
    """Synthetic dataset that mimics PASCAL VOC format.
    
    Generates random images with synthetic objects for testing
    YOLO v1 without requiring real data.
    
    Args:
        num_samples: Number of samples to generate
        image_size: Image size (default: 448)
        S: Grid size (default: 7)
        B: Number of boxes per cell (default: 2)
        C: Number of classes (default: 20)
        max_objects: Maximum objects per image (default: 5)
        min_objects: Minimum objects per image (default: 1)
    """
    
    VOC_CLASSES = [
        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
        'bus', 'car', 'cat', 'chair', 'cow',
        'diningtable', 'dog', 'horse', 'motorbike', 'person',
        'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]
    
    def __init__(
        self,
        num_samples: int = 1000,
        image_size: int = 448,
        S: int = 7,
        B: int = 2,
        C: int = 20,
        max_objects: int = 5,
        min_objects: int = 1
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.S = S
        self.B = B
        self.C = C
        self.max_objects = max_objects
        self.min_objects = min_objects
        
        self.class_colors = self._generate_class_colors()
    
    def _generate_class_colors(self) -> np.ndarray:
        colors = np.zeros((self.C, 3), dtype=np.uint8)
        for i in range(self.C):
            np.random.seed(i * 17)
            colors[i] = np.random.randint(80, 200, size=3)
        return colors
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        np.random.seed(idx)
        
        image = np.random.randint(30, 60, (self.image_size, self.image_size, 3), dtype=np.uint8)
        
        num_objects = np.random.randint(self.min_objects, self.max_objects + 1)
        
        boxes = []
        labels = []
        
        for _ in range(num_objects):
            label = np.random.randint(0, self.C)
            
            obj_w = np.random.randint(30, self.image_size // 3)
            obj_h = np.random.randint(30, self.image_size // 3)
            
            x1 = np.random.randint(10, self.image_size - obj_w - 10)
            y1 = np.random.randint(10, self.image_size - obj_h - 10)
            x2 = x1 + obj_w
            y2 = y1 + obj_h
            
            boxes.append([x1, y1, x2, y2])
            labels.append(label)
            
            color = self.class_colors[label]
            noise = np.random.randint(-20, 20, 3)
            obj_color = np.clip(color.astype(np.int32) + noise, 0, 255).astype(np.uint8)
            image[y1:y2, x1:x2] = obj_color
        
        boxes = np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)
        labels = np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)
        
        from nanotorch.detection.yolo_v1.yolo_v1_loss import encode_targets
        target = encode_targets(boxes, labels, self.S, self.B, self.C, self.image_size)
        
        image = image.astype(np.float32) / 255.0
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'target': target,
            'image_id': idx
        }


class YOLOv1Transform:
    """Transform pipeline for YOLO v1 preprocessing."""
    
    def __init__(
        self,
        image_size: int = 448,
        S: int = 7,
        B: int = 2,
        C: int = 20
    ):
        self.image_size = image_size
        self.S = S
        self.B = B
        self.C = C
    
    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        image = sample['image']
        boxes = sample['boxes']
        
        if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            image, boxes = self._resize(image, boxes, self.image_size)
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': sample['labels'],
            'target': sample['target'],
            'image_id': sample.get('image_id', 0)
        }
    
    def _resize(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        target_size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        h, w = image.shape[:2]
        
        scale = target_size / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        
        resized = self._resize_image(image, (new_h, new_w))
        
        pad_h = (target_size - new_h) // 2
        pad_w = (target_size - new_w) // 2
        
        padded = np.zeros((target_size, target_size, 3), dtype=resized.dtype)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        
        if len(boxes) > 0:
            boxes = boxes.copy()
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale + pad_w
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale + pad_h
        
        return padded, boxes
    
    def _resize_image(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        h, w = size
        orig_h, orig_w = image.shape[:2]
        
        y_indices = (np.arange(h) * orig_h / h).astype(np.int32)
        x_indices = (np.arange(w) * orig_w / w).astype(np.int32)
        
        y_indices = np.clip(y_indices, 0, orig_h - 1)
        x_indices = np.clip(x_indices, 0, orig_w - 1)
        
        return image[y_indices][:, x_indices]


class YOLOv1Collate:
    """Collate function for YOLO v1 dataloader."""
    
    def __init__(self, S: int = 7, B: int = 2, C: int = 20):
        self.S = S
        self.B = B
        self.C = C
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        images = []
        targets = []
        all_boxes = []
        all_labels = []
        
        for sample in batch:
            img = sample['image']
            if img.max() <= 1.0:
                pass
            else:
                img = img.astype(np.float32) / 255.0
            
            img = np.transpose(img, (2, 0, 1))
            images.append(img)
            
            targets.append(sample['target'])
            all_boxes.append(sample['boxes'])
            all_labels.append(sample['labels'])
        
        images = np.stack(images, axis=0).astype(np.float32)
        targets = np.stack(targets, axis=0).astype(np.float32)
        
        return {
            'images': images,
            'targets': targets,
            'boxes': all_boxes,
            'labels': all_labels
        }


def create_yolov1_dataloader(
    dataset: Dataset,
    batch_size: int = 8,
    shuffle: bool = True,
    S: int = 7,
    B: int = 2,
    C: int = 20
) -> DataLoader:
    """Create YOLO v1 dataloader.
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle
        S: Grid size
        B: Number of boxes per cell
        C: Number of classes
    
    Returns:
        DataLoader instance
    """
    collate_fn = YOLOv1Collate(S=S, B=B, C=C)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )


def create_synthetic_dataloader(
    num_samples: int = 1000,
    batch_size: int = 8,
    image_size: int = 224,
    S: int = 7,
    B: int = 2,
    C: int = 20,
    shuffle: bool = True
) -> DataLoader:
    """Create synthetic dataloader for testing.
    
    Args:
        num_samples: Number of samples
        batch_size: Batch size
        image_size: Image size
        S: Grid size
        B: Number of boxes per cell
        C: Number of classes
        shuffle: Whether to shuffle
    
    Returns:
        DataLoader instance
    """
    dataset = SyntheticVOCDataset(
        num_samples=num_samples,
        image_size=image_size,
        S=S,
        B=B,
        C=C
    )
    
    return create_yolov1_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        S=S,
        B=B,
        C=C
    )
