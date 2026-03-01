"""
Integration tests for YOLO v5 training and inference pipeline.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from nanotorch.tensor import Tensor
from nanotorch.optim import Adam
from nanotorch.detection.yolo_v5 import (
    YOLOv5,
    YOLOv5Nano,
    YOLOv5Loss,
    YOLOv5LossSimple,
    decode_predictions_v5,
    encode_targets_v5,
    build_yolov5
)
from nanotorch.data import Dataset, DataLoader


class SyntheticDataset(Dataset):
    def __init__(self, num_samples=100, image_size=224, num_classes=80, max_objects=5):
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
            w = np.random.randint(20, self.image_size // 4)
            h = np.random.randint(20, self.image_size // 4)
            x1 = np.random.randint(0, self.image_size - w)
            y1 = np.random.randint(0, self.image_size - h)
            boxes.append([x1, y1, x1 + w, y1 + h])
            labels.append(np.random.randint(0, self.num_classes))
        
        return {
            'image': image.transpose(2, 0, 1),
            'boxes': np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32),
            'labels': np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64),
        }


def create_dataloader(num_samples=20, batch_size=2, image_size=224, num_classes=80):
    dataset = SyntheticDataset(num_samples, image_size, num_classes)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


class TestModelDataIntegration:
    def test_model_with_dataloader(self):
        dataloader = create_dataloader(num_samples=4, batch_size=2, image_size=224)
        model = YOLOv5Nano(num_classes=80, input_size=224)
        
        for batch in dataloader:
            images = np.stack([item['image'] for item in batch])
            images = Tensor(images)
            output = model(images)
            assert 'small' in output
            break


class TestTrainingIntegration:
    def test_forward_backward_cycle(self):
        model = YOLOv5Nano(num_classes=80, input_size=224)
        x = Tensor(np.random.randn(2, 3, 224, 224).astype(np.float32))
        
        output = model(x)
        
        pred = output['small']
        target = Tensor(np.zeros_like(pred.data))
        diff = pred - target
        loss = (diff * diff).mean()
        
        loss.backward()
        
        params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
        assert params_with_grad > 0
    
    def test_loss_computation(self):
        loss_fn = YOLOv5Loss(num_classes=80)
        
        predictions = {
            'small': Tensor(np.random.rand(1, 255, 7, 7).astype(np.float32)),
            'medium': Tensor(np.random.rand(1, 255, 14, 14).astype(np.float32)),
            'large': Tensor(np.random.rand(1, 255, 28, 28).astype(np.float32))
        }
        
        targets = [{'boxes': np.array([[50, 50, 150, 150]], dtype=np.float32), 'labels': np.array([0])}]
        
        loss, loss_dict = loss_fn(predictions, targets)
        
        assert 'box_loss' in loss_dict
        assert 'obj_loss' in loss_dict


class TestInferenceIntegration:
    def test_single_inference(self):
        model = YOLOv5Nano(num_classes=80, input_size=224)
        x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
        output = model(x)
        assert 'small' in output
    
    def test_batch_inference(self):
        model = YOLOv5Nano(num_classes=80, input_size=224)
        x = Tensor(np.random.randn(4, 3, 224, 224).astype(np.float32))
        output = model(x)
        assert output['small'].shape[0] == 4


class TestTrainingLoopIntegration:
    def test_mini_training_loop(self):
        dataloader = create_dataloader(num_samples=4, batch_size=2, image_size=224)
        model = YOLOv5Nano(num_classes=80, input_size=224)
        optimizer = Adam(model.parameters(), lr=0.01)
        
        losses = []
        
        for batch in dataloader:
            images = np.stack([item['image'] for item in batch])
            images = Tensor(images)
            
            optimizer.zero_grad()
            output = model(images)
            
            pred = output['small']
            target = Tensor(np.zeros_like(pred.data))
            diff = pred - target
            loss = (diff * diff).mean()
            
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        assert all(not np.isnan(l) for l in losses)
    
    def test_model_save_load(self):
        model1 = YOLOv5Nano(num_classes=80, input_size=224)
        x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
        output1 = model1(x)
        
        state = model1.state_dict()
        
        model2 = YOLOv5Nano(num_classes=80, input_size=224)
        model2.load_state_dict(state)
        
        output2 = model2(x)
        
        np.testing.assert_allclose(output1['small'].data, output2['small'].data, rtol=1e-5)


class TestNumericalStability:
    def test_loss_with_zeros(self):
        model = YOLOv5Nano(num_classes=80, input_size=224)
        loss_fn = YOLOv5LossSimple(num_classes=80)
        
        x = Tensor(np.zeros((1, 3, 224, 224), dtype=np.float32))
        output = model(x)
        
        targets = {k: Tensor(np.zeros_like(v.data)) for k, v in output.items()}
        loss, _ = loss_fn(output, targets)
        
        assert not np.isnan(loss)


class TestEncodeDecode:
    def test_encode_targets(self):
        anchors = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]
        boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
        labels = np.array([0], dtype=np.int64)
        
        targets = encode_targets_v5(boxes, labels, anchors, grid_sizes=[28, 14, 7], num_classes=80, image_size=224)
        
        assert targets['scale_0'].shape == (3, 85, 28, 28)
    
    def test_decode_predictions(self):
        predictions = np.zeros((3, 85, 7, 7), dtype=np.float32)
        predictions[0, 4, 3, 3] = 0.9
        predictions[0, 0, 3, 3] = 0.5
        predictions[0, 1, 3, 3] = 0.5
        predictions[0, 5, 3, 3] = 0.9
        
        anchors = [(116, 90), (156, 198), (373, 326)]
        
        boxes, scores, class_ids = decode_predictions_v5(predictions, anchors=anchors, conf_threshold=0.5, num_classes=80, image_size=224)
        
        assert len(boxes) >= 1


class TestBuildModel:
    def test_build_nano(self):
        model = build_yolov5('n', num_classes=80, input_size=224)
        assert isinstance(model, YOLOv5)
    
    def test_build_small(self):
        model = build_yolov5('s', num_classes=80, input_size=224)
        assert isinstance(model, YOLOv5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
