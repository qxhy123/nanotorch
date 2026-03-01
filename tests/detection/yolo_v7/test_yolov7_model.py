import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from nanotorch.tensor import Tensor
from nanotorch.detection.yolo_v7 import YOLOv7, build_yolov7, YOLOv7Loss, YOLOv7LossSimple

class TestYOLOv7:
    def test_model_forward(self):
        model = build_yolov7(num_classes=80, input_size=224)
        x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
        output = model(x)
        assert 'small' in output
        assert 'medium' in output
        assert 'large' in output
    
    def test_model_parameters(self):
        model = YOLOv7(num_classes=80, input_size=224)
        params = list(model.parameters())
        assert len(params) > 0
        total_params = sum(p.data.size for p in params)
        assert total_params > 0
    
    def test_loss_forward(self):
        loss_fn = YOLOv7Loss(num_classes=80)
        predictions = {'small': Tensor(np.random.rand(1, 85, 7, 7).astype(np.float32))}
        targets = [{'boxes': np.array([[50, 50, 150, 150]], dtype=np.float32), 'labels': np.array([0], dtype=np.int64)}]
        loss, loss_dict = loss_fn(predictions, targets)
        assert 'total_loss' in loss_dict
    
    def test_simple_loss(self):
        loss_fn = YOLOv7LossSimple(num_classes=80)
        predictions = {'small': Tensor(np.random.rand(2, 85, 7, 7).astype(np.float32))}
        targets = {'small': Tensor(np.zeros((2, 85, 7, 7), dtype=np.float32))}
        loss, _ = loss_fn(predictions, targets)
        assert loss >= 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
