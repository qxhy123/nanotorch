"""
Real image detection demo with sample images.

This script downloads sample images and runs YOLO v12 detection.
"""

import numpy as np
import sys
import os
from typing import Tuple, Dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nanotorch.tensor import Tensor
from nanotorch.detection import (
    build_backbone,
    build_neck, 
    build_head,
    batched_nms
)

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Warning: PIL not available. Install with: pip install pillow")


class YOLOv12Detector:
    """Complete YOLO v12 detector for inference."""
    
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]
    
    COLORS = [
        '#FF3838', '#FF9D97', '#FF701F', '#FFB21D', '#CFD231', '#48F90A',
        '#92CC17', '#3DDB86', '#1A9334', '#00D4BB', '#2C99A8', '#00C2FF',
        '#344593', '#6473FF', '#0018EC', '#8438FF', '#520085', '#CB38FF',
        '#FF95C8', '#FF37C7',
    ]
    
    def __init__(
        self,
        model_size: str = 'n',
        num_classes: int = 80,
        input_size: int = 640,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        use_attention: bool = False
    ):
        self.num_classes = num_classes
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        print(f"Building YOLO v12-{model_size} model...")
        
        self.backbone = build_backbone(
            model_size=model_size,
            in_channels=3,
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
            num_classes=num_classes,
            reg_max=16
        )
        
        self.strides = {'p3': 8, 'p4': 16, 'p5': 32}
        self.reg_max = 16
        
        print("Model built successfully!")
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple]:
        orig_h, orig_w = image.shape[:2]
        
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
        
        target_size = self.input_size
        scale = min(target_size / orig_w, target_size / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        
        resized = self._resize_image(image, (new_h, new_w))
        
        pad_h = (target_size - new_h) // 2
        pad_w = (target_size - new_w) // 2
        
        padded = np.full((target_size, target_size, 3), 0.5, dtype=np.float32)
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        
        processed = np.transpose(padded, (2, 0, 1))
        processed = processed[np.newaxis, :, :, :]
        
        meta = (orig_h, orig_w, scale, pad_h, pad_w)
        
        return processed, meta
    
    def _resize_image(self, image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        h, w = size
        orig_h, orig_w = image.shape[:2]
        
        y_indices = (np.arange(h) * orig_h / h).astype(np.int32)
        x_indices = (np.arange(w) * orig_w / w).astype(np.int32)
        
        y_indices = np.clip(y_indices, 0, orig_h - 1)
        x_indices = np.clip(x_indices, 0, orig_w - 1)
        
        return image[y_indices][:, x_indices]
    
    def forward(self, x: Tensor) -> Dict[str, Tuple[Tensor, Tensor]]:
        features = self.backbone(x)
        features = self.neck(features)
        predictions = self.head(features)
        return predictions
    
    def decode_predictions(
        self,
        predictions: Dict[str, Tuple[Tensor, Tensor]],
        meta: Tuple
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        orig_h, orig_w, scale, pad_h, pad_w = meta
        
        all_boxes = []
        all_scores = []
        all_class_ids = []
        
        for scale_name, (box_pred, cls_pred) in predictions.items():
            stride = self.strides[scale_name]
            
            N, _, H, W = box_pred.shape
            
            box_pred_np = box_pred.data[0]
            cls_pred_np = cls_pred.data[0]
            
            cls_scores = 1.0 / (1.0 + np.exp(-cls_pred_np))
            
            box_pred_reshaped = box_pred_np.reshape(4, self.reg_max, H, W)
            box_pred_reshaped = box_pred_reshaped.transpose(2, 3, 0, 1)
            
            box_pred_softmax = self._softmax(box_pred_reshaped.reshape(H * W, 4, self.reg_max))
            box_pred_softmax = box_pred_softmax.reshape(H, W, 4, self.reg_max)
            
            arange = np.arange(self.reg_max, dtype=np.float32)
            distances = (box_pred_softmax * arange).sum(axis=-1) * stride
            
            for i in range(H):
                for j in range(W):
                    scores = cls_scores[:, i, j]
                    max_score = scores.max()
                    
                    if max_score < self.conf_threshold:
                        continue
                    
                    class_id = scores.argmax()
                    
                    cx = (j + 0.5) * stride
                    cy = (i + 0.5) * stride
                    
                    dl, dt, dr, db = distances[i, j]
                    
                    x1 = cx - dl
                    y1 = cy - dt
                    x2 = cx + dr
                    y2 = cy + db
                    
                    all_boxes.append([x1, y1, x2, y2])
                    all_scores.append(max_score)
                    all_class_ids.append(class_id)
        
        if len(all_boxes) == 0:
            return np.array([]), np.array([]), np.array([])
        
        all_boxes = np.array(all_boxes, dtype=np.float32)
        all_scores = np.array(all_scores, dtype=np.float32)
        all_class_ids = np.array(all_class_ids, dtype=np.int64)
        
        keep = batched_nms(all_boxes, all_scores, all_class_ids, self.iou_threshold)
        
        all_boxes = all_boxes[keep]
        all_scores = all_scores[keep]
        all_class_ids = all_class_ids[keep]
        
        all_boxes[:, 0] = (all_boxes[:, 0] - pad_w) / scale
        all_boxes[:, 1] = (all_boxes[:, 1] - pad_h) / scale
        all_boxes[:, 2] = (all_boxes[:, 2] - pad_w) / scale
        all_boxes[:, 3] = (all_boxes[:, 3] - pad_h) / scale
        
        all_boxes[:, 0] = np.clip(all_boxes[:, 0], 0, orig_w)
        all_boxes[:, 1] = np.clip(all_boxes[:, 1], 0, orig_h)
        all_boxes[:, 2] = np.clip(all_boxes[:, 2], 0, orig_w)
        all_boxes[:, 3] = np.clip(all_boxes[:, 3], 0, orig_h)
        
        return all_boxes, all_scores, all_class_ids
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        x_max = x.max(axis=axis, keepdims=True)
        exp_x = np.exp(x - x_max)
        return exp_x / exp_x.sum(axis=axis, keepdims=True)
    
    def detect(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        processed, meta = self.preprocess(image)
        x = Tensor(processed)
        predictions = self.forward(x)
        boxes, scores, class_ids = self.decode_predictions(predictions, meta)
        return boxes, scores, class_ids
    
    def visualize_pil(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        output_path: str,
        score_threshold: float = 0.3
    ):
        if not HAS_PIL:
            print("PIL not available. Cannot visualize.")
            return
        
        if image.dtype == np.float32 and image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        except:
            font = ImageFont.load_default()
        
        for i in range(len(boxes)):
            if scores[i] < score_threshold:
                continue
            
            x1, y1, x2, y2 = boxes[i]
            
            class_id = int(class_ids[i])
            color = self.COLORS[class_id % len(self.COLORS)]
            
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            class_name = self.COCO_CLASSES[class_id] if class_id < len(self.COCO_CLASSES) else f"cls_{class_id}"
            label = f"{class_name}: {scores[i]:.2f}"
            
            bbox = draw.textbbox((x1, y1), label, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            draw.rectangle([x1, y1 - text_height - 4, x1 + text_width + 4, y1], fill=color)
            draw.text((x1 + 2, y1 - text_height - 2), label, fill='white', font=font)
        
        pil_image.save(output_path)
        print(f"Saved: {output_path}")
        
        return pil_image


def download_sample_image(url: str, save_path: str):
    import urllib.request
    
    print(f"Downloading: {url}")
    urllib.request.urlretrieve(url, save_path)
    print(f"Saved to: {save_path}")


def create_test_image(size: int = 640):
    np.random.seed(42)
    
    image = np.random.randint(100, 200, (size, size, 3), dtype=np.uint8)
    
    shapes = [
        {'xy': (50, 50, 200, 300), 'color': (255, 100, 100)},
        {'xy': (250, 100, 450, 350), 'color': (100, 255, 100)},
        {'xy': (350, 300, 550, 500), 'color': (100, 100, 255)},
        {'xy': (100, 350, 300, 550), 'color': (255, 255, 100)},
    ]
    
    for shape in shapes:
        x1, y1, x2, y2 = shape['xy']
        color = shape['color']
        
        image[y1:y2, x1:x2] = color
        
        image[y1:y1+3, x1:x2] = (255, 255, 255)
        image[y2-3:y2, x1:x2] = (255, 255, 255)
        image[y1:y2, x1:x1+3] = (255, 255, 255)
        image[y1:y2, x2-3:x2] = (255, 255, 255)
    
    for _ in range(50):
        x = np.random.randint(0, size)
        y = np.random.randint(0, size)
        gray = np.random.randint(50, 150)
        image[max(0,y-5):min(size,y+5), max(0,x-5):min(size,x+5)] = (gray, gray, gray)
    
    return image


def main():
    print("=" * 70)
    print("YOLO v12 Real Image Detection Demo")
    print("=" * 70)
    
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    detector = YOLOv12Detector(
        model_size='n',
        num_classes=80,
        input_size=640,
        conf_threshold=0.25,
        iou_threshold=0.45,
        use_attention=False
    )
    
    print("\n" + "-" * 70)
    print("Test 1: Synthetic test image")
    print("-" * 70)
    
    test_image = create_test_image(640)
    
    test_input_path = os.path.join(output_dir, 'test_input.jpg')
    if HAS_PIL:
        Image.fromarray(test_image).save(test_input_path)
        print(f"Saved test input: {test_input_path}")
    
    print("\nRunning detection...")
    boxes, scores, class_ids = detector.detect(test_image)
    
    print(f"Detected {len(boxes)} objects")
    
    if HAS_PIL:
        output_path = os.path.join(output_dir, 'test_output.jpg')
        detector.visualize_pil(test_image, boxes, scores, class_ids, output_path, score_threshold=0.3)
    
    print("\n" + "-" * 70)
    print("Test 2: Try with a real image (if available)")
    print("-" * 70)
    
    sample_urls = [
        "https://ultralytics.com/images/bus.jpg",
        "https://ultralytics.com/images/zidane.jpg",
    ]
    
    sample_path = os.path.join(output_dir, 'sample.jpg')
    
    if not os.path.exists(sample_path):
        try:
            download_sample_image(sample_urls[0], sample_path)
        except Exception as e:
            print(f"Could not download sample image: {e}")
            print("Creating a simple test image instead...")
            
            sample_image = create_test_image(640)
            if HAS_PIL:
                Image.fromarray(sample_image).save(sample_path)
                print(f"Created sample image: {sample_path}")
    
    if os.path.exists(sample_path) and HAS_PIL:
        print(f"\nLoading: {sample_path}")
        sample_image = np.array(Image.open(sample_path).convert('RGB'))
        
        print(f"Image size: {sample_image.shape}")
        
        print("\nRunning detection...")
        boxes, scores, class_ids = detector.detect(sample_image)
        
        print(f"\nDetected {len(boxes)} objects")
        for i in range(min(len(boxes), 10)):
            class_name = detector.COCO_CLASSES[int(class_ids[i])] if int(class_ids[i]) < len(detector.COCO_CLASSES) else f"cls_{class_ids[i]}"
            print(f"  [{i+1}] {class_name}: {scores[i]:.3f} at [{boxes[i, 0]:.0f}, {boxes[i, 1]:.0f}, {boxes[i, 2]:.0f}, {boxes[i, 3]:.0f}]")
        
        output_path = os.path.join(output_dir, 'sample_output.jpg')
        detector.visualize_pil(sample_image, boxes, scores, class_ids, output_path, score_threshold=0.25)
    
    print("\n" + "=" * 70)
    print("Demo completed!")
    print(f"Output files saved to: {output_dir}")
    print("=" * 70)
    
    print("\n" + "=" * 70)
    print("IMPORTANT: Model uses random weights!")
    print("=" * 70)
    print("""
This demo uses a YOLO v12 model with RANDOMLY INITIALIZED weights.
The detection results are NOT accurate because the model hasn't been trained.

To get meaningful detections, you would need to:
1. Train the model on a labeled dataset (like COCO)
2. Or load pre-trained weights from a checkpoint

The purpose of this demo is to show:
- How to use the YOLO v12 architecture
- The complete inference pipeline (preprocess -> forward -> postprocess)
- Visualization of detection results

The model architecture is complete and functional.
With proper training, it would produce accurate detections.
""")


if __name__ == "__main__":
    main()
