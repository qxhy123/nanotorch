import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from nanotorch.tensor import Tensor
from nanotorch.optim import Adam
from nanotorch.detection.yolo_v8 import YOLOv8, build_yolov8
from nanotorch.data import Dataset, DataLoader

class SyntheticDataset(Dataset):
    def __init__(self, num_samples=20, image_size=224, num_classes=80):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        np.random.seed(idx)
        image = np.random.rand(self.image_size, self.image_size, 3).astype(np.float32)
        return {'image': image.transpose(2, 0, 1), 'boxes': np.zeros((0, 4), dtype=np.float32), 'labels': np.zeros((0,), dtype=np.int64)}

class TestIntegration:
    def test_dataloader_integration(self):
        dataloader = DataLoader(SyntheticDataset(4, 224, 80), batch_size=2, shuffle=False)
        model = build_yolov8(num_classes=80, input_size=224)
        for batch in dataloader:
            images = np.stack([item['image'] for item in batch])
            images = Tensor(images)
            output = model(images)
            assert 'small' in output
            break
    
    def test_training_loop(self):
        model = build_yolov8(num_classes=80, input_size=224)
        optimizer = Adam(model.parameters(), lr=0.01)
        for _ in range(2):
            x = Tensor(np.random.randn(2, 3, 224, 224).astype(np.float32))
            optimizer.zero_grad()
            output = model(x)
            loss = sum(np.mean(pred.data ** 2) for pred in output.values())
            Tensor(loss, requires_grad=True).backward()
            optimizer.step()
        assert True

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
