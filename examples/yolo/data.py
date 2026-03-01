"""
Data loading and augmentation for YOLO object detection.

This module provides:
- Synthetic dataset generation for testing
- Data augmentation (Mosaic, MixUp, random affine)
- DataLoader for batching
- Target assignment for training
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Callable, Any
import random
from nanotorch.tensor import Tensor
from nanotorch.data import Dataset, DataLoader
from nanotorch.detection.bbox import (
    xyxy_to_xywh, xywh_to_xyxy, clip_boxes, box_iou
)


class DetectionSample:
    """Single detection sample with image and annotations."""
    
    def __init__(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        labels: np.ndarray,
        image_id: int = 0
    ):
        self.image = image
        self.boxes = boxes
        self.labels = labels
        self.image_id = image_id


class SyntheticDetectionDataset(Dataset):
    """Synthetic dataset for testing YOLO models.
    
    Generates random images with synthetic objects for rapid testing
    without requiring real datasets.
    
    Args:
        num_samples: Number of samples to generate
        image_size: (H, W) image size (default: 640)
        num_classes: Number of object classes (default: 10)
        max_objects: Maximum objects per image (default: 10)
        min_objects: Minimum objects per image (default: 1)
    """
    
    def __init__(
        self,
        num_samples: int = 1000,
        image_size: Tuple[int, int] = (640, 640),
        num_classes: int = 10,
        max_objects: int = 10,
        min_objects: int = 1
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self.max_objects = max_objects
        self.min_objects = min_objects
        
        self.class_colors = self._generate_class_colors()
    
    def _generate_class_colors(self) -> np.ndarray:
        colors = np.zeros((self.num_classes, 3), dtype=np.uint8)
        for i in range(self.num_classes):
            np.random.seed(i * 17)
            colors[i] = np.random.randint(50, 200, size=3)
        return colors
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        np.random.seed(idx)
        
        h, w = self.image_size
        
        image = np.random.randint(20, 40, (h, w, 3), dtype=np.uint8)
        
        num_objects = np.random.randint(self.min_objects, self.max_objects + 1)
        
        boxes = []
        labels = []
        
        for _ in range(num_objects):
            label = np.random.randint(0, self.num_classes)
            
            obj_w = np.random.randint(30, min(200, w // 2))
            obj_h = np.random.randint(30, min(200, h // 2))
            
            x1 = np.random.randint(0, w - obj_w)
            y1 = np.random.randint(0, h - obj_h)
            x2 = x1 + obj_w
            y2 = y1 + obj_h
            
            boxes.append([x1, y1, x2, y2])
            labels.append(label)
            
            color = self.class_colors[label]
            noise = np.random.randint(-20, 20, 3)
            obj_color = np.clip(color.astype(np.int32) + noise, 0, 255).astype(np.uint8)
            image[y1:y2, x1:x2] = obj_color
        
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        
        image = image.astype(np.float32) / 255.0
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': labels,
            'image_id': idx
        }


class MosaicAugmentation:
    """Mosaic augmentation for object detection.
    
    Combines 4 images into a single mosaic image, increasing
    data diversity and small object visibility.
    
    Args:
        image_size: Output image size (default: 640)
        p: Probability of applying mosaic (default: 1.0)
    """
    
    def __init__(
        self,
        image_size: int = 640,
        p: float = 1.0
    ):
        self.image_size = image_size
        self.p = p
    
    def __call__(
        self,
        samples: List[Dict[str, np.ndarray]]
    ) -> Dict[str, np.ndarray]:
        if len(samples) != 4:
            raise ValueError("Mosaic requires exactly 4 samples")
        
        if random.random() > self.p:
            return samples[0]
        
        h, w = self.image_size, self.image_size
        
        mosaic_image = np.zeros((h, w, 3), dtype=np.float32)
        mosaic_boxes = []
        mosaic_labels = []
        
        cx = np.random.randint(w // 4, 3 * w // 4)
        cy = np.random.randint(h // 4, 3 * h // 4)
        
        positions = [
            (0, 0, cx, cy),
            (cx, 0, w, cy),
            (0, cy, cx, h),
            (cx, cy, w, h)
        ]
        
        for i, (x1, y1, x2, y2) in enumerate(positions):
            sample = samples[i]
            img = sample['image']
            boxes = sample['boxes'].copy()
            labels = sample['labels']
            
            patch_h = y2 - y1
            patch_w = x2 - x1
            
            img_h, img_w = img.shape[:2]
            scale_x = patch_w / img_w
            scale_y = patch_h / img_h
            
            mosaic_image[y1:y2, x1:x2] = np.clip(
                img * 255 if img.max() <= 1 else img,
                0, 255
            ).astype(np.float32) / 255.0
            
            if len(boxes) > 0:
                boxes[:, 0] = boxes[:, 0] * scale_x + x1
                boxes[:, 1] = boxes[:, 1] * scale_y + y1
                boxes[:, 2] = boxes[:, 2] * scale_x + x1
                boxes[:, 3] = boxes[:, 3] * scale_y + y1
                
                valid_mask = (
                    (boxes[:, 2] > boxes[:, 0]) &
                    (boxes[:, 3] > boxes[:, 1]) &
                    (boxes[:, 0] < w) & (boxes[:, 1] < h) &
                    (boxes[:, 2] > 0) & (boxes[:, 3] > 0)
                )
                boxes = boxes[valid_mask]
                labels = labels[valid_mask]
                
                boxes = clip_boxes(boxes, (w, h))
                
                mosaic_boxes.append(boxes)
                mosaic_labels.append(labels)
        
        if len(mosaic_boxes) > 0:
            mosaic_boxes = np.concatenate(mosaic_boxes, axis=0)
            mosaic_labels = np.concatenate(mosaic_labels, axis=0)
        else:
            mosaic_boxes = np.zeros((0, 4), dtype=np.float32)
            mosaic_labels = np.zeros((0,), dtype=np.int64)
        
        return {
            'image': mosaic_image,
            'boxes': mosaic_boxes,
            'labels': mosaic_labels,
            'image_id': samples[0]['image_id']
        }


class MixUpAugmentation:
    """MixUp augmentation for object detection.
    
    Blends two images and their annotations.
    
    Args:
        p: Probability of applying mixup (default: 0.1)
        alpha: Beta distribution parameter (default: 32.0)
    """
    
    def __init__(self, p: float = 0.1, alpha: float = 32.0):
        self.p = p
        self.alpha = alpha
    
    def __call__(
        self,
        sample1: Dict[str, np.ndarray],
        sample2: Dict[str, np.ndarray]
    ) -> Dict[str, np.ndarray]:
        if random.random() > self.p:
            return sample1
        
        lam = np.random.beta(self.alpha, self.alpha)
        lam = max(0.3, min(0.7, lam))
        
        mixed_image = lam * sample1['image'] + (1 - lam) * sample2['image']
        
        mixed_boxes = np.concatenate([
            sample1['boxes'],
            sample2['boxes']
        ], axis=0) if len(sample1['boxes']) > 0 or len(sample2['boxes']) > 0 else np.zeros((0, 4), dtype=np.float32)
        
        mixed_labels = np.concatenate([
            sample1['labels'],
            sample2['labels']
        ], axis=0) if len(sample1['labels']) > 0 or len(sample2['labels']) > 0 else np.zeros((0,), dtype=np.int64)
        
        return {
            'image': mixed_image,
            'boxes': mixed_boxes,
            'labels': mixed_labels,
            'image_id': sample1['image_id']
        }


class RandomHorizontalFlip:
    """Random horizontal flip for object detection."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if random.random() > self.p:
            return sample
        
        image = sample['image'].copy()
        boxes = sample['boxes'].copy()
        
        w = image.shape[1]
        
        image = image[:, ::-1, :]
        
        if len(boxes) > 0:
            boxes[:, 0], boxes[:, 2] = w - boxes[:, 2], w - boxes[:, 0]
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': sample['labels'],
            'image_id': sample['image_id']
        }


class RandomVerticalFlip:
    """Random vertical flip for object detection."""
    
    def __init__(self, p: float = 0.5):
        self.p = p
    
    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if random.random() > self.p:
            return sample
        
        image = sample['image'].copy()
        boxes = sample['boxes'].copy()
        
        h = image.shape[0]
        
        image = image[::-1, :, :]
        
        if len(boxes) > 0:
            boxes[:, 1], boxes[:, 3] = h - boxes[:, 3], h - boxes[:, 1]
        
        return {
            'image': image,
            'boxes': boxes,
            'labels': sample['labels'],
            'image_id': sample['image_id']
        }


class RandomScale:
    """Random scale augmentation."""
    
    def __init__(self, scale_range: Tuple[float, float] = (0.5, 1.5)):
        self.scale_range = scale_range
    
    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        scale = random.uniform(*self.scale_range)
        
        image = sample['image']
        h, w = image.shape[:2]
        
        new_h, new_w = int(h * scale), int(w * scale)
        
        image_scaled = self._resize_image(image, (new_h, new_w))
        
        boxes = sample['boxes'].copy() * scale
        
        return {
            'image': image_scaled,
            'boxes': boxes,
            'labels': sample['labels'],
            'image_id': sample['image_id']
        }
    
    def _resize_image(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        h, w = size
        orig_h, orig_w = image.shape[:2]
        
        y_indices = (np.arange(h) * orig_h / h).astype(np.int32)
        x_indices = (np.arange(w) * orig_w / w).astype(np.int32)
        
        y_indices = np.clip(y_indices, 0, orig_h - 1)
        x_indices = np.clip(x_indices, 0, orig_w - 1)
        
        return image[y_indices][:, x_indices]


class LetterboxResize:
    """Letterbox resize maintaining aspect ratio.
    
    Resizes image while maintaining aspect ratio by adding padding.
    
    Args:
        target_size: Target size (single int or (H, W) tuple)
        fill_value: Padding fill value (default: 114)
    """
    
    def __init__(
        self,
        target_size: int = 640,
        fill_value: int = 114
    ):
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        self.target_size = target_size
        self.fill_value = fill_value
    
    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        image = sample['image']
        boxes = sample['boxes'].copy()
        
        h, w = image.shape[:2]
        target_h, target_w = self.target_size
        
        scale = min(target_w / w, target_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        image_scaled = self._resize_image(image, (new_h, new_w))
        
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        
        if image_scaled.max() <= 1.0:
            fill = self.fill_value / 255.0
        else:
            fill = self.fill_value
        
        padded = np.full((target_h, target_w, 3), fill, dtype=image_scaled.dtype)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = image_scaled
        
        if len(boxes) > 0:
            boxes[:, 0] = boxes[:, 0] * scale + pad_w
            boxes[:, 1] = boxes[:, 1] * scale + pad_h
            boxes[:, 2] = boxes[:, 2] * scale + pad_w
            boxes[:, 3] = boxes[:, 3] * scale + pad_h
        
        return {
            'image': padded,
            'boxes': boxes,
            'labels': sample['labels'],
            'image_id': sample['image_id']
        }
    
    def _resize_image(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        h, w = size
        orig_h, orig_w = image.shape[:2]
        
        y_indices = (np.arange(h) * orig_h / h).astype(np.int32)
        x_indices = (np.arange(w) * orig_w / w).astype(np.int32)
        
        y_indices = np.clip(y_indices, 0, orig_h - 1)
        x_indices = np.clip(x_indices, 0, orig_w - 1)
        
        return image[y_indices][:, x_indices]


class Compose:
    """Compose multiple transforms."""
    
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms
    
    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        for t in self.transforms:
            sample = t(sample)
        return sample


class YOLOCollate:
    """Collate function for YOLO dataloader."""
    
    def __init__(self, image_size: int = 640):
        self.image_size = image_size
    
    def __call__(self, batch: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        images = []
        all_boxes = []
        all_labels = []
        batch_indices = []
        
        for i, sample in enumerate(batch):
            img = sample['image']
            if img.max() <= 1.0:
                img = img * 255
            
            if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
                resize = LetterboxResize(self.image_size)
                sample = resize(sample)
                img = sample['image']
                if img.max() <= 1.0:
                    img = img * 255
            
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            images.append(img)
            
            boxes = sample['boxes']
            labels = sample['labels']
            
            all_boxes.append(boxes)
            all_labels.append(labels)
            batch_indices.append(np.full(len(boxes), i, dtype=np.int64))
        
        images = np.stack(images, axis=0).astype(np.float32)
        
        if len(all_boxes) > 0 and any(len(b) > 0 for b in all_boxes):
            all_boxes = np.concatenate(all_boxes, axis=0).astype(np.float32)
            all_labels = np.concatenate(all_labels, axis=0).astype(np.int64)
            batch_indices = np.concatenate(batch_indices, axis=0).astype(np.int64)
        else:
            all_boxes = np.zeros((0, 4), dtype=np.float32)
            all_labels = np.zeros((0,), dtype=np.int64)
            batch_indices = np.zeros((0,), dtype=np.int64)
        
        return {
            'images': images,
            'boxes': all_boxes,
            'labels': all_labels,
            'batch_indices': batch_indices
        }


def create_yolo_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    shuffle: bool = True,
    image_size: int = 640,
    num_workers: int = 0
) -> DataLoader:
    collate_fn = YOLOCollate(image_size=image_size)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn
    )


def create_synthetic_dataloader(
    num_samples: int = 1000,
    batch_size: int = 8,
    image_size: int = 640,
    num_classes: int = 10,
    shuffle: bool = True
) -> DataLoader:
    dataset = SyntheticDetectionDataset(
        num_samples=num_samples,
        image_size=(image_size, image_size),
        num_classes=num_classes
    )
    
    return create_yolo_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        image_size=image_size
    )
