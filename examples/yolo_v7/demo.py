#!/usr/bin/env python3
import argparse
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nanotorch.tensor import Tensor
from nanotorch.optim import Adam
from nanotorch.optim.lr_scheduler import StepLR
from nanotorch.detection.yolo_v7 import YOLOv7, YOLOv7Loss, build_yolov7
from nanotorch.data import Dataset, DataLoader

COCO_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

class SyntheticCOCODataset(Dataset):
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
        boxes, labels = [], []
        for _ in range(num_objects):
            w = np.random.randint(30, self.image_size // 4)
            h = np.random.randint(30, self.image_size // 4)
            x1 = np.random.randint(0, self.image_size - w)
            y1 = np.random.randint(0, self.image_size - h)
            boxes.append([x1, y1, x1 + w, y1 + h])
            labels.append(np.random.randint(0, self.num_classes))
        return {'image': image.transpose(2, 0, 1), 'boxes': np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32), 'labels': np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)}

def create_dataloader(num_samples, batch_size, image_size, num_classes, shuffle=True):
    return DataLoader(SyntheticCOCODataset(num_samples, image_size, num_classes), batch_size=batch_size, shuffle=shuffle)

def demo_architecture():
    print("=" * 60)
    print("YOLO v7 Architecture Demo (Trainable Bag-of-Freebies)")
    print("=" * 60)
    model = build_yolov7(num_classes=80, input_size=224)
    num_params = sum(p.data.size for p in model.parameters())
    print(f"\nModel Parameters: {num_params:,}")
    x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
    output = model(x)
    for scale, pred in output.items():
        print(f"   {scale}: {pred.shape}")
    print("\nKey Features:")
    print("   - E-ELAN (Extended Efficient Layer Aggregation Network)")
    print("   - Model scaling techniques")
    print("   - Auxiliary training heads")

def train(args):
    print("=" * 60)
    print("YOLO v7 Training Demo")
    print("=" * 60)
    model = build_yolov7(num_classes=args.num_classes, input_size=args.image_size)
    num_params = sum(p.data.size for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=0.1)
    dataloader = create_dataloader(args.num_samples, args.batch_size, args.image_size, args.num_classes)
    print(f"\nStarting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            images = np.stack([item['image'] for item in batch])
            images = Tensor(images)
            optimizer.zero_grad()
            output = model(images)
            loss = sum(np.mean(pred.data ** 2) for pred in output.values())
            Tensor(loss, requires_grad=True).backward()
            optimizer.step()
            total_loss += loss
        scheduler.step()
        print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {total_loss/len(dataloader):.4f}")
    np.savez('yolov7_best.npz', **model.state_dict())
    print(f"Model saved to: yolov7_best.npz")
    return model

def inference(args):
    print("=" * 60)
    print("YOLO v7 Inference Demo")
    print("=" * 60)
    model = build_yolov7(num_classes=args.num_classes, input_size=args.image_size)
    if os.path.exists('yolov7_best.npz'):
        model.load_state_dict(dict(np.load('yolov7_best.npz')))
        print("Loaded trained weights")
    else:
        print("Using random initialization")
    model.eval()
    np.random.seed(42)
    for i in range(args.num_inference):
        image = np.random.randn(1, 3, args.image_size, args.image_size).astype(np.float32)
        output = model(Tensor(image))
        print(f"\nImage {i+1}:")
        for scale_name, pred in output.items():
            print(f"  {scale_name}: shape={pred.shape}")

def main():
    parser = argparse.ArgumentParser(description='YOLO v7 Demo')
    parser.add_argument('--mode', type=str, default='both', choices=['train', 'inference', 'both', 'arch'])
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr-step', type=int, default=5)
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--num-samples', type=int, default=20)
    parser.add_argument('--num-inference', type=int, default=3)
    parser.add_argument('--num-classes', type=int, default=80)
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
