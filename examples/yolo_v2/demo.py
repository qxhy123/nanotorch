"""
YOLO v2 Demo Script

This script demonstrates how to use the YOLO v2 model for:
- Model architecture inspection
- Training on synthetic data
- Inference demonstration
"""

import argparse
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nanotorch.tensor import Tensor
from nanotorch.optim import Adam, SGD
from nanotorch.optim.lr_scheduler import StepLR
from nanotorch.detection.yolo_v2 import (
    build_yolov2,
    YOLOv2Loss,
    YOLOv2LossSimple,
    encode_targets_v2,
    decode_predictions_v2
)


def print_architecture(args):
    model = build_yolov2(args.model_type, num_classes=args.num_classes, input_size=args.image_size)
    
    print("=" * 60)
    print("YOLO v2 Model Architecture")
    print("=" * 60)
    print(f"Model Type: {args.model_type}")
    print(f"Input Size: {args.image_size}x{args.image_size}")
    print(f"Number of Classes: {args.num_classes}")
    print(f"Number of Anchors: 5")
    
    grid_size = args.image_size // 32
    output_channels = 5 * (5 + args.num_classes)
    
    print(f"\nOutput Grid Size: {grid_size}x{grid_size}")
    print(f"Output Channels: {output_channels}")
    print(f"Output Shape: (N, {output_channels}, {grid_size}, {grid_size})")
    
    num_params = sum(p.data.size for p in model.parameters())
    print(f"\nTotal Parameters: {num_params:,}")
    
    print("\n" + "=" * 60)
    print("YOLO v2 Key Features:")
    print("=" * 60)
    print("1. Darknet-19 Backbone (19 conv layers)")
    print("2. Batch Normalization on all conv layers")
    print("3. Anchor-based detection (5 anchors)")
    print("4. Passthrough layer for fine-grained features")
    print("5. Direct location prediction (sigmoid)")


def train_model(args):
    print("=" * 60)
    print("YOLO v2 Training Demo")
    print("=" * 60)
    
    model = build_yolov2(args.model_type, num_classes=args.num_classes, input_size=args.image_size)
    loss_fn = YOLOv2LossSimple(num_classes=args.num_classes)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    
    print(f"Training Configuration:")
    print(f"  Epochs: {args.epochs}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Image Size: {args.image_size}")
    print()
    
    grid_size = args.image_size // 32
    output_channels = 5 * (5 + args.num_classes)
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        
        for batch in range(10):
            images = Tensor(np.random.randn(args.batch_size, 3, args.image_size, args.image_size).astype(np.float32))
            
            target_data = np.zeros((args.batch_size, output_channels, grid_size, grid_size), dtype=np.float32)
            targets = {'output': Tensor(target_data)}
            
            optimizer.zero_grad()
            output = model(images)
            
            loss, _ = loss_fn(output, targets)
            
            loss_tensor = Tensor(loss, requires_grad=True)
            loss_tensor.backward()
            optimizer.step()
            
            total_loss += loss
        
        scheduler.step()
        avg_loss = total_loss / 10
        print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {avg_loss:.6f} LR: {scheduler.get_last_lr()[0]:.6f}")
    
    print("\nTraining completed!")
    return model


def run_inference(args):
    print("=" * 60)
    print("YOLO v2 Inference Demo")
    print("=" * 60)
    
    model = build_yolov2(args.model_type, num_classes=args.num_classes, input_size=args.image_size)
    model.eval()
    
    print(f"Running inference on random input...")
    print(f"  Input shape: (1, 3, {args.image_size}, {args.image_size})")
    
    x = Tensor(np.random.randn(1, 3, args.image_size, args.image_size).astype(np.float32))
    
    output = model(x)
    
    grid_size = args.image_size // 32
    output_channels = 5 * (5 + args.num_classes)
    
    print(f"  Output shape: {output['output'].shape}")
    print(f"  Expected: (1, {output_channels}, {grid_size}, {grid_size})")
    
    boxes, scores, class_ids = decode_predictions_v2(
        output['output'].data,
        conf_threshold=0.3,
        num_anchors=5,
        num_classes=args.num_classes,
        image_size=args.image_size
    )
    
    print(f"\nDetection Results:")
    if len(boxes) > 0:
        print(f"  Detected {len(boxes)} objects")
        for i in range(min(5, len(boxes))):
            print(f"    Box {i+1}: {boxes[i]} Score: {scores[i]:.4f} Class: {class_ids[i]}")
    else:
        print("  No objects detected (random input)")


def main():
    parser = argparse.ArgumentParser(description='YOLO v2 Demo')
    parser.add_argument('--mode', type=str, default='arch', 
                        choices=['arch', 'train', 'inference', 'both'],
                        help='Demo mode')
    parser.add_argument('--model-type', type=str, default='tiny',
                        choices=['full', 'tiny'],
                        help='Model type')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--image-size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--num-classes', type=int, default=20,
                        help='Number of classes')
    
    args = parser.parse_args()
    
    if args.mode == 'arch':
        print_architecture(args)
    elif args.mode == 'train':
        train_model(args)
    elif args.mode == 'inference':
        run_inference(args)
    elif args.mode == 'both':
        print_architecture(args)
        print()
        train_model(args)
        print()
        run_inference(args)


if __name__ == "__main__":
    main()
