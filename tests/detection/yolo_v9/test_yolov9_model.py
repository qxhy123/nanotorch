import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from nanotorch.tensor import Tensor
from nanotorch.detection.yolo_v9 import YOLOv9, build_yolov9, YOLOv9Loss

class TestYOLOv9:
    def test_model_forward(self):
        model = build_yolov9(num_classes=80, input_size=224)
        x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
        output = model(x)
        assert 'small' in output
        assert 'medium' in output
        assert 'large' in output
    
    def test_model_parameters(self):
        model = YOLOv9(num_classes=80, input_size=224)
        params = list(model.parameters())
        assert len(params) > 0
    
    def test_loss_forward(self):
        loss_fn = YOLOv9Loss(num_classes=80)
        predictions = {'small': Tensor(np.random.rand(1, 85, 7, 7).astype(np.float32))}
        targets = [{'boxes': np.array([[50, 50, 150, 150]], dtype=np.float32), 'labels': np.array([0], dtype=np.int64)}]
        loss, loss_dict = loss_fn(predictions, targets)
        assert 'total_loss' in loss_dict

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
