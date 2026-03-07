"""
Integration tests for YOLO v6 training and inference pipeline.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from nanotorch.tensor import Tensor
from nanotorch.optim import SGD, Adam
from nanotorch.detection.yolo_v6 import (
    YOLOv6,
    YOLOv6Nano,
    YOLOv6Small,
    YOLOv6Loss,
    YOLOv6LossSimple,
    decode_predictions_v6,
    encode_targets_v6,
    build_yolov6
)
from nanotorch.data import Dataset, DataLoader


class SyntheticCOCODataset(Dataset):
    
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
        boxes, labels = [], []
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


def create_test_dataloader(num_samples=20, batch_size=2, image_size=224, num_classes=80):
    dataset = SyntheticCOCODataset(num_samples, image_size, num_classes)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


class TestModelDataIntegration:
    
    def test_model_accepts_dataloader_output(self):
        dataloader = create_test_dataloader(num_samples=4, batch_size=2, image_size=224)
        model = YOLOv6Nano(num_classes=80, input_size=224)
        for batch in dataloader:
            images = np.stack([item['image'] for item in batch])
            images = Tensor(images)
            output = model(images)
            assert 'small' in output
            assert 'medium' in output
            assert 'large' in output
            break


class TestTrainingIntegration:
    
    def test_forward_loss_backward_cycle(self):
        model = YOLOv6Nano(num_classes=80, input_size=224)
        x = Tensor(np.random.randn(2, 3, 224, 224).astype(np.float32))
        output = model(x)
        loss = sum(np.mean(pred.data ** 2) for pred in output.values())
        loss_tensor = Tensor(loss, requires_grad=True)
        loss_tensor.backward()
        params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
        assert params_with_grad > 0
    
    def test_optimizer_step(self):
        model = YOLOv6Nano(num_classes=80, input_size=224)
        optimizer = Adam(model.parameters(), lr=0.1)
        x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
        optimizer.zero_grad()
        output = model(x)
        loss = sum(np.mean(pred.data ** 2) for pred in output.values())
        Tensor(loss, requires_grad=True).backward()
        optimizer.step()
        assert loss > 0


class TestInferenceIntegration:
    
    def test_predict(self):
        model = YOLOv6Nano(num_classes=80, input_size=224)
        x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
        output = model(x)
        assert 'small' in output
        assert 'medium' in output
        assert 'large' in output
    
    def test_batch_inference(self):
        model = YOLOv6Nano(num_classes=80, input_size=224)
        x = Tensor(np.random.randn(4, 3, 224, 224).astype(np.float32))
        output = model(x)
        assert output['small'].shape[0] == 4


class TestTrainingLoopIntegration:
    
    def test_mini_training_loop(self):
        dataloader = create_test_dataloader(num_samples=4, batch_size=2, image_size=224)
        model = YOLOv6Nano(num_classes=80, input_size=224)
        optimizer = Adam(model.parameters(), lr=0.01)
        epoch_losses = []
        for batch in dataloader:
            images = np.stack([item['image'] for item in batch])
            images = Tensor(images)
            optimizer.zero_grad()
            output = model(images)
            loss = sum(np.mean(pred.data ** 2) for pred in output.values())
            Tensor(loss, requires_grad=True).backward()
            optimizer.step()
            epoch_losses.append(loss)
        assert all(not np.isnan(l) for l in epoch_losses)
    
    def test_model_save_load_state(self):
        model1 = YOLOv6Nano(num_classes=80, input_size=224)
        x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
        output1 = model1(x)
        state = model1.state_dict()
        model2 = YOLOv6Nano(num_classes=80, input_size=224)
        model2.load_state_dict(state)
        output2 = model2(x)
        np.testing.assert_allclose(output1['small'].data, output2['small'].data, rtol=1e-5)


class TestEncodeDecode:
    
    def test_encode_targets(self):
        boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
        labels = np.array([0], dtype=np.int64)
        targets = encode_targets_v6(
            boxes=boxes,
            labels=labels,
            grid_sizes=[20, 10, 5],
            num_classes=80,
            image_size=224
        )
        assert 'scale_0' in targets
        assert 'scale_1' in targets
        assert 'scale_2' in targets
    
    def test_decode_predictions_empty(self):
        predictions = np.zeros((1, 85, 10, 10), dtype=np.float32)
        boxes, scores, class_ids = decode_predictions_v6(
            predictions, conf_threshold=0.5, num_classes=80, image_size=224
        )
        assert len(boxes) == 0


class TestBuildModel:
    
    def test_build_nano_model(self):
        model = build_yolov6('n', num_classes=80, input_size=224)
        assert isinstance(model, YOLOv6Nano)
    
    def test_build_small_model(self):
        model = build_yolov6('s', num_classes=80, input_size=224)
        assert isinstance(model, YOLOv6Small)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
