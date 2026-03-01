import numpy as np
import pytest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from nanotorch.tensor import Tensor
from nanotorch.optim import Adam
from nanotorch.detection.yolo_v10 import YOLOv10, build_yolov10

class TestIntegration:
    def test_forward_backward(self):
        model = build_yolov10(num_classes=80, input_size=224)
        x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
        output = model(x)
        loss = sum(np.mean(pred.data ** 2) for pred in output.values())
        Tensor(loss, requires_grad=True).backward()
        assert True
    
    def test_training_step(self):
        model = build_yolov10(num_classes=80, input_size=224)
        optimizer = Adam(model.parameters(), lr=0.01)
        x = Tensor(np.random.randn(2, 3, 224, 224).astype(np.float32))
        optimizer.zero_grad()
        output = model(x)
        loss = sum(np.mean(pred.data ** 2) for pred in output.values())
        Tensor(loss, requires_grad=True).backward()
        optimizer.step()
        assert True

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
