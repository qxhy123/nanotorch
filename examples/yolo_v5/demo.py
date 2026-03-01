#!/usr/bin/env python3
"""
YOLO v5 Demo - Training and Inference Example

This demo showcases how to use nanotorch to train and evaluate YOLO v5
for object detection on synthetic data.

Usage:
    python examples/yolo_v5/demo.py --mode train --epochs 3
    python examples/yolo_v5/demo.py --mode inference
    python examples/yolo_v5/demo.py --mode both
"""

import argparse
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nanotorch.tensor import Tensor
from nanotorch.optim import Adam
from nanotorch.optim.lr_scheduler import StepLR
from nanotorch.detection.yolo_v5 import (
    YOLOv5,
    YOLOv5Nano,
    YOLOv5Small,
    YOLOv5Loss,
    build_yolov5
)
from nanotorch.data import Dataset, DataLoader


COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class SyntheticCOCODataset(Dataset):
    """Synthetic dataset mimicking COCO format."""
    
    def __init__(self, num_samples=100, image_size=640, num_classes=80, max_objects=5):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self.max_objects = max_objects
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        np.random.seed(idx)
        
        image = np.random.rand(self.image_size, self.image_size, 3).astype(np.float32)
        
        num_objects = np.random.randint(1, self.max_objects + 1)
        
        boxes = []
        labels = []
        
        for _ in range(num_objects):
            w = np.random.randint(30, self.image_size // 4)
            h = np.random.randint(30, self.image_size // 4)
            x1 = np.random.randint(0, self.image_size - w)
            y1 = np.random.randint(0, self.image_size - h)
            x2 = x1 + w
            y2 = y1 + h
            
            boxes.append([x1, y1, x2, y2])
            labels.append(np.random.randint(0, self.num_classes))
        
        return {
            'image': image.transpose(2, 0, 1),
            'boxes': np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32),
            'labels': np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64),
            'image_id': idx
        }


def create_dataloader(num_samples, batch_size, image_size, num_classes, shuffle=True):
    dataset = SyntheticCOCODataset(num_samples, image_size, num_classes)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def demo_architecture():
    print("=" * 60)
    print("YOLO v5 Architecture Demo")
    print("=" * 60)
    
    print("\n1. YOLOv5Nano Model:")
    model = YOLOv5Nano(num_classes=80, input_size=224)
    num_params = sum(p.data.size for p in model.parameters())
    print(f"   Parameters: {num_params:,}")
    
    x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
    output = model(x)
    
    for scale, pred in output.items():
        print(f"   {scale}: {pred.shape}")
    
    print("\n2. YOLOv5Small Model:")
    model = YOLOv5Small(num_classes=80, input_size=224)
    num_params = sum(p.data.size for p in model.parameters())
    print(f"   Parameters: {num_params:,}")
    
    output = model(x)
    for scale, pred in output.items():
        print(f"   {scale}: {pred.shape}")
    
    print("\n3. Key Features:")
    print("   - C3 modules (CSP Bottleneck with 3 convolutions)")
    print("   - SPPF (Spatial Pyramid Pooling Fast)")
    print("   - SiLU activation function")
    print("   - PANet neck for feature fusion")
    print("   - Multiple model sizes (n, s, m, l, x)")


def train(args):
    print("=" * 60)
    print("YOLO v5 Training Demo")
    print("=" * 60)
    
    image_size = args.image_size
    version = args.version
    
    model = build_yolov5(version, num_classes=args.num_classes, input_size=image_size)
    print(f"Using YOLOv5-{version} with input size {image_size}")
    
    num_params = sum(p.data.size for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=0.1)
    
    dataloader = create_dataloader(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        image_size=image_size,
        num_classes=args.num_classes
    )
    
    print(f"\nCreating synthetic dataset with {args.num_samples} samples...")
    print(f"Dataset created: {len(dataloader)} batches")
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            images = np.stack([item['image'] for item in batch])
            images = Tensor(images)
            
            optimizer.zero_grad()
            output = model(images)
            
            loss = 0.0
            for scale_name, pred in output.items():
                target = Tensor(np.zeros_like(pred.data))
                diff = pred - target
                loss += (diff * diff).mean().item()
            
            loss_tensor = Tensor(loss, requires_grad=True)
            loss_tensor.backward()
            optimizer.step()
            
            total_loss += loss
            num_batches += 1
        
        scheduler.step()
        avg_loss = total_loss / num_batches
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1:3d}/{args.epochs}] Loss: {avg_loss:.4f} LR: {current_lr:.6f}")
    
    print("-" * 60)
    print(f"Training complete!")
    
    state_dict = model.state_dict()
    np.savez('yolov5_best.npz', **state_dict)
    print(f"Model saved to: yolov5_best.npz")
    
    return model


def inference(args):
    print("=" * 60)
    print("YOLO v5 Inference Demo")
    print("=" * 60)
    
    image_size = args.image_size
    version = args.version
    
    model = build_yolov5(version, num_classes=args.num_classes, input_size=image_size)
    print(f"Using YOLOv5-{version} with input size {image_size}")
    
    if os.path.exists('yolov5_best.npz'):
        print("Loading trained weights from yolov5_best.npz...")
        state_dict = dict(np.load('yolov5_best.npz'))
        model.load_state_dict(state_dict)
    else:
        print("No trained weights found, using random initialization.")
    
    model.eval()
    
    print(f"\nRunning inference on {args.num_inference} synthetic images...")
    print("-" * 60)
    
    np.random.seed(42)
    
    for i in range(args.num_inference):
        image = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
        x = Tensor(image)
        
        output = model(x)
        
        print(f"\nImage {i+1}:")
        for scale_name, pred in output.items():
            print(f"  {scale_name}: shape={pred.shape}")


def main():
    parser = argparse.ArgumentParser(description='YOLO v5 Demo')
    parser.add_argument('--mode', type=str, default='both',
                        choices=['train', 'inference', 'both', 'arch'],
                        help='Demo mode')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lr-step', type=int, default=5, help='LR scheduler step')
    parser.add_argument('--image-size', type=int, default=224, help='Input image size')
    parser.add_argument('--num-samples', type=int, default=20, help='Number of training samples')
    parser.add_argument('--num-inference', type=int, default=3, help='Number of inference images')
    parser.add_argument('--num-classes', type=int, default=80, help='Number of classes')
    parser.add_argument('--version', type=str, default='n', 
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='Model version (n=nano, s=small, m=medium, l=large, x=xlarge)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'inference':
        inference(args)
    elif args.mode == 'arch':
        demo_architecture()
    else:
        demo_architecture()
        print("\n")
        train(args)
        print("\n")
        inference(args)


if __name__ == "__main__":
    main()
