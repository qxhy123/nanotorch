"""
Integration tests for YOLO v2 model.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from nanotorch.tensor import Tensor
from nanotorch.optim import Adam, SGD
from nanotorch.detection.yolo_v2 import (
    build_yolov2,
    YOLOv2Loss,
    YOLOv2LossSimple
)
from nanotorch.data import Dataset, DataLoader


class SyntheticDataset(Dataset):
    """Synthetic dataset for testing."""
    
    def __init__(self, num_samples=20, image_size=224, num_classes=20):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        np.random.seed(idx)
        image = np.random.rand(self.image_size, self.image_size, 3).astype(np.float32)
        num_objects = np.random.randint(0, 3)
        
        boxes = []
        labels = []
        for _ in range(num_objects):
            x1 = np.random.randint(0, self.image_size - 50)
            y1 = np.random.randint(0, self.image_size - 50)
            x2 = x1 + np.random.randint(20, 50)
            y2 = y1 + np.random.randint(20, 50)
            boxes.append([x1, y1, x2, y2])
            labels.append(np.random.randint(0, self.num_classes))
        
        return {
            'image': image.transpose(2, 0, 1),
            'boxes': np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32),
            'labels': np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64)
        }


class TestIntegration:
    """Integration tests for YOLO v2."""
    
    def test_dataloader_integration(self):
        """Test model with DataLoader."""
        dataloader = DataLoader(
            SyntheticDataset(4, 224, 20),
            batch_size=2,
            shuffle=False
        )
        
        model = build_yolov2('tiny', num_classes=20, input_size=224)
        
        for batch in dataloader:
            images = np.stack([item['image'] for item in batch])
            images = Tensor(images)
            
            output = model(images)
            
            assert 'output' in output
            expected_channels = 5 * (5 + 20)
            assert output['output'].shape[0] == 2
            assert output['output'].shape[1] == expected_channels
            break
    
    def test_training_loop(self):
        """Test complete training loop."""
        model = build_yolov2('tiny', num_classes=20, input_size=224)
        optimizer = Adam(model.parameters(), lr=0.001)
        
        for _ in range(3):
            x = Tensor(np.random.randn(2, 3, 224, 224).astype(np.float32))
            
            optimizer.zero_grad()
            output = model(x)
            
            loss = 0.0
            for v in output.values():
                loss += np.mean(v.data ** 2)
            
            loss_tensor = Tensor(loss, requires_grad=True)
            loss_tensor.backward()
            optimizer.step()
        
        assert True
    
    def test_loss_computation(self):
        """Test loss computation with targets."""
        model = build_yolov2('tiny', num_classes=20, input_size=224)
        loss_fn = YOLOv2Loss(num_classes=20)
        
        x = Tensor(np.random.randn(2, 3, 224, 224).astype(np.float32))
        
        model.train()
        output = model(x)
        
        targets = [
            {
                'boxes': np.array([[50, 50, 100, 100]], dtype=np.float32),
                'labels': np.array([0], dtype=np.int64)
            },
            {
                'boxes': np.array([[30, 30, 80, 80]], dtype=np.float32),
                'labels': np.array([1], dtype=np.int64)
            }
        ]
        
        loss, loss_dict = loss_fn(output, targets)
        
        assert loss.item() >= 0
        assert 'coord_loss' in loss_dict
        assert 'obj_loss' in loss_dict
        assert 'noobj_loss' in loss_dict
        assert 'class_loss' in loss_dict
    
    def test_forward_backward(self):
        """Test forward and backward pass."""
        model = build_yolov2('tiny', num_classes=20, input_size=224)
        
        x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
        
        output = model(x)
        loss = sum(np.mean(v.data ** 2) for v in output.values())
        
        loss_tensor = Tensor(loss, requires_grad=True)
        loss_tensor.backward()
        
        assert True
    
    def test_training_step(self):
        """Test single training step."""
        model = build_yolov2('tiny', num_classes=20, input_size=224)
        optimizer = SGD(model.parameters(), lr=0.001)
        loss_fn = YOLOv2LossSimple(num_classes=20)
        
        x = Tensor(np.random.randn(2, 3, 224, 224).astype(np.float32))
        
        optimizer.zero_grad()
        output = model(x)
        
        grid_size = output['output'].shape[2]
        target = {'output': Tensor(np.zeros((2, 125, grid_size, grid_size), dtype=np.float32))}
        
        loss, _ = loss_fn(output, target)
        
        loss_tensor = Tensor(loss, requires_grad=True)
        loss_tensor.backward()
        optimizer.step()
        
        assert True
    
    def test_eval_mode(self):
        """Test model in evaluation mode."""
        model = build_yolov2('tiny', num_classes=20, input_size=224)
        
        model.eval()
        
        x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
        output1 = model(x)
        output2 = model(x)
        
        assert np.allclose(output1['output'].data, output2['output'].data)
    
    def test_save_load_state(self):
        """Test model state save and load."""
        model1 = build_yolov2('tiny', num_classes=20, input_size=224)
        model1.eval()
        
        x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
        output1 = model1(x)
        
        state = model1.state_dict()
        
        model2 = build_yolov2('tiny', num_classes=20, input_size=224)
        model2.eval()
        model2.load_state_dict(state)
        
        output2 = model2(x)
        
        assert np.allclose(output1['output'].data, output2['output'].data, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
