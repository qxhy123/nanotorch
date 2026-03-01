#!/usr/bin/env python3
"""
YOLO v1 Demo - Training and Inference Example

This demo showcases how to use nanotorch to train and evaluate YOLO v1
for object detection on synthetic VOC-style data.

Usage:
    python examples/yolo_v1/demo.py --mode train --epochs 10
    python examples/yolo_v1/demo.py --mode inference
    python examples/yolo_v1/demo.py --mode both
"""

import argparse
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nanotorch.tensor import Tensor
from nanotorch.optim import Adam, SGD
from nanotorch.optim.lr_scheduler import StepLR
from nanotorch.detection.yolo_v1 import (
    YOLOv1,
    YOLOv1Tiny,
    YOLOv1Loss,
    decode_predictions,
    build_yolov1
)
from nanotorch.detection.nms import nms
from examples.yolo_v1.data import create_synthetic_dataloader


# VOC 20 class names
VOC_CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]


def train(args):
    """Train YOLO v1 model on synthetic data."""
    print("=" * 60)
    print("YOLO v1 Training Demo")
    print("=" * 60)
    
    S, B, C = 7, 2, 20
    
    tiny_default_size = 224
    full_default_size = 448
    use_tiny = args.tiny
    image_size = args.image_size
    
    if use_tiny and image_size == full_default_size:
        image_size = tiny_default_size
    elif not use_tiny and image_size == tiny_default_size:
        image_size = full_default_size
    
    if use_tiny:
        model = YOLOv1Tiny(input_size=image_size, S=S, B=B, C=C)
        print(f"Using YOLOv1Tiny with input size {image_size}")
    else:
        print("WARNING: Full YOLOv1 requires ~8GB+ memory for 448x448 input")
        print("         If you get SIGKILL, try: --tiny or reduce --batch-size")
        model = YOLOv1(input_size=image_size, S=S, B=B, C=C)
        print(f"Using YOLOv1 with input size {image_size}")
    
    num_params = sum(p.data.size for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=args.lr_step, gamma=0.1)
    
    loss_fn = YOLOv1Loss(S=S, B=B, C=C, coord_weight=5.0, noobj_weight=0.5)
    
    print(f"\nCreating synthetic dataset with {args.num_samples} samples...")
    dataloader = create_synthetic_dataloader(
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        image_size=image_size,
        S=S, B=B, C=C,
        shuffle=True
    )
    print(f"Dataset created: {len(dataloader)} batches")
    
    # Training loop
    print(f"\nStarting training for {args.epochs} epochs...")
    print("-" * 60)
    
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in dataloader:
            images = Tensor(batch['images'])
            targets = Tensor(batch['targets'])
            
            optimizer.zero_grad()
            
            # Forward pass
            output = model(images)
            output_reshaped = output.reshape((images.shape[0], S, S, B*5+C))
            
            # Compute MSE loss for backward (YOLOv1Loss doesn't support autograd)
            diff = output_reshaped - targets
            loss = (diff * diff).mean()
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        avg_loss = total_loss / num_batches
        
        # Print progress
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1:3d}/{args.epochs}] Loss: {avg_loss:.4f} LR: {current_lr:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            state_dict = model.state_dict()
            np.savez('yolov1_best.npz', **state_dict)
    
    print("-" * 60)
    print(f"Training complete! Best loss: {best_loss:.4f}")
    print(f"Model saved to: yolov1_best.npz")
    
    return model


def inference(args):
    """Run inference with trained YOLO v1 model."""
    print("=" * 60)
    print("YOLO v1 Inference Demo")
    print("=" * 60)
    
    S, B, C = 7, 2, 20
    tiny_default_size = 224
    full_default_size = 448
    use_tiny = args.tiny
    image_size = args.image_size
    
    if use_tiny and image_size == full_default_size:
        image_size = tiny_default_size
    elif not use_tiny and image_size == tiny_default_size:
        image_size = full_default_size
    
    if use_tiny:
        model = YOLOv1Tiny(input_size=image_size, S=S, B=B, C=C)
        print(f"Using YOLOv1Tiny with input size {image_size}")
    else:
        model = YOLOv1(input_size=image_size, S=S, B=B, C=C)
        print(f"Using YOLOv1 with input size {image_size}")
    
    if os.path.exists('yolov1_best.npz'):
        print("Loading trained weights from yolov1_best.npz...")
        state_dict = dict(np.load('yolov1_best.npz'))
        model.load_state_dict(state_dict)
    else:
        print("No trained weights found, using random initialization.")
    
    model.eval()
    
    print(f"\nRunning inference on {args.num_inference} synthetic images...")
    print("-" * 60)
    
    np.random.seed(42)
    
    total_detections = 0
    
    for i in range(args.num_inference):
        # Create random image
        image = np.random.randn(1, 3, image_size, image_size).astype(np.float32)
        x = Tensor(image)
        
        # Forward pass
        output = model(x)
        output_reshaped = output.reshape((1, S, S, B*5+C))
        
        # Decode predictions
        predictions = output_reshaped.data[0]
        boxes, scores, class_ids = decode_predictions(
            predictions,
            conf_threshold=args.conf_threshold,
            image_size=image_size
        )
        
        # Apply NMS
        if len(boxes) > 0:
            keep = nms(boxes, scores, iou_threshold=args.nms_threshold)
            boxes = boxes[keep]
            scores = scores[keep]
            class_ids = class_ids[keep]
        
        # Print detections
        print(f"\nImage {i+1}:")
        if len(boxes) == 0:
            print("  No detections")
        else:
            total_detections += len(boxes)
            for j in range(len(boxes)):
                class_name = VOC_CLASSES[class_ids[j]]
                x1, y1, x2, y2 = boxes[j]
                print(f"  {class_name}: ({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), score: {scores[j]:.2f}")
    
    print("-" * 60)
    print(f"Inference complete! Total detections: {total_detections}")
    print(f"Average detections per image: {total_detections / args.num_inference:.1f}")


def demo_encode_decode():
    """Demo the encode/decode pipeline."""
    print("=" * 60)
    print("YOLO v1 Encode/Decode Demo")
    print("=" * 60)
    
    from nanotorch.detection.yolo_v1 import encode_targets, decode_predictions
    
    # Example: Two objects in an image
    boxes = np.array([
        [100, 100, 200, 200],  # Object 1: (x1, y1, x2, y2)
        [300, 300, 400, 400],  # Object 2
    ], dtype=np.float32)
    
    labels = np.array([5, 14], dtype=np.int64)  # Class 5=bus, 14=person
    
    print("\nInput boxes:")
    for i, (box, label) in enumerate(zip(boxes, labels)):
        print(f"  Object {i+1}: {VOC_CLASSES[label]} at {box}")
    
    # Encode to YOLO format
    target = encode_targets(boxes, labels, S=7, B=2, C=20, image_size=448)
    print(f"\nEncoded target shape: {target.shape}")
    print(f"Non-zero elements: {np.count_nonzero(target)}")
    
    # Find grid cells with objects
    object_cells = []
    for i in range(7):
        for j in range(7):
            if target[i, j, 4] > 0.5:  # Check confidence
                object_cells.append((i, j))
    
    print(f"Grid cells with objects: {object_cells}")
    
    # Decode back to boxes
    decoded_boxes, scores, class_ids = decode_predictions(
        target, conf_threshold=0.5, image_size=448
    )
    
    print(f"\nDecoded boxes ({len(decoded_boxes)} total):")
    for i in range(len(decoded_boxes)):
        class_name = VOC_CLASSES[class_ids[i]]
        print(f"  {class_name}: box={decoded_boxes[i]}, score={scores[i]:.2f}")


def demo_model_architecture():
    """Demo the model architecture."""
    print("=" * 60)
    print("YOLO v1 Architecture Demo")
    print("=" * 60)
    
    # Build models
    print("\n1. Full YOLOv1 Model:")
    full_model = build_yolov1('full', input_size=448, S=7, B=2, C=20)
    full_params = sum(p.data.size for p in full_model.parameters())
    print(f"   Parameters: {full_params:,}")
    
    # Test forward pass
    x = Tensor(np.random.randn(1, 3, 448, 448).astype(np.float32))
    output = full_model(x)
    print(f"   Input shape: (1, 3, 448, 448)")
    print(f"   Output shape: {output.shape}")
    
    print("\n2. YOLOv1Tiny Model (for testing):")
    tiny_model = build_yolov1('tiny', input_size=224, S=7, B=2, C=20)
    tiny_params = sum(p.data.size for p in tiny_model.parameters())
    print(f"   Parameters: {tiny_params:,}")
    
    x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
    output = tiny_model(x)
    print(f"   Input shape: (1, 3, 224, 224)")
    print(f"   Output shape: {output.shape}")
    
    # Show output format
    print("\n3. Output Format:")
    print(f"   Grid size: 7×7 = 49 cells")
    print(f"   Boxes per cell: 2")
    print(f"   Classes: 20")
    print(f"   Output per cell: 2×5 + 20 = 30 values")
    print(f"   Total output: 7×7×30 = 1470 values")


def main():
    parser = argparse.ArgumentParser(description='YOLO v1 Demo')
    parser.add_argument('--mode', type=str, default='both', 
                        choices=['train', 'inference', 'both', 'encode', 'arch'],
                        help='Demo mode')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lr-step', type=int, default=5, help='LR scheduler step')
    parser.add_argument('--image-size', type=int, default=224, help='Input image size')
    parser.add_argument('--num-samples', type=int, default=20, help='Number of training samples')
    parser.add_argument('--num-inference', type=int, default=3, help='Number of inference images')
    parser.add_argument('--conf-threshold', type=float, default=0.3, help='Confidence threshold')
    parser.add_argument('--nms-threshold', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--tiny', action='store_true', default=True, help='Use YOLOv1Tiny (default: True)')
    parser.add_argument('--full', action='store_true', help='Use full YOLOv1 (requires more memory)')
    
    args = parser.parse_args()
    
    if args.full:
        args.tiny = False
    
    if args.mode == 'train':
        train(args)
    elif args.mode == 'inference':
        inference(args)
    elif args.mode == 'encode':
        demo_encode_decode()
    elif args.mode == 'arch':
        demo_model_architecture()
    else:  # both
        demo_model_architecture()
        print("\n")
        demo_encode_decode()
        print("\n")
        train(args)
        print("\n")
        inference(args)


if __name__ == "__main__":
    main()
