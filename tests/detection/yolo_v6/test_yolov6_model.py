"""Unit tests for YOLO v6."""

import numpy as np
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from nanotorch.tensor import Tensor
from nanotorch.detection.yolo_v6 import (
    ConvBN, RepVGGBlock, YOLOv6, YOLOv6Nano, build_yolov6,
    YOLOv6Loss, YOLOv6LossSimple, encode_targets_v6, decode_predictions_v6
)


class TestConvBN:
    def test_forward(self):
        block = ConvBN(64, 128, 3)
        x = Tensor(np.random.randn(2, 64, 16, 16).astype(np.float32))
        y = block(x)
        assert y.shape == (2, 128, 16, 16)


class TestRepVGGBlock:
    def test_forward(self):
        block = RepVGGBlock(64, 64)
        x = Tensor(np.random.randn(2, 64, 16, 16).astype(np.float32))
        y = block(x)
        assert y.shape == (2, 64, 16, 16)


class TestYOLOv6:
    def test_forward(self):
        model = YOLOv6(num_classes=80, input_size=224, variant='s')
        x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
        output = model(x)
        assert 'small' in output
        assert 'medium' in output
        assert 'large' in output


class TestBuildYOLOv6:
    def test_build_nano(self):
        model = build_yolov6('n', num_classes=80, input_size=224)
        assert isinstance(model, YOLOv6)
    
    def test_build_small(self):
        model = build_yolov6('s', num_classes=80, input_size=224)
        assert isinstance(model, YOLOv6)


class TestLoss:
    def test_loss(self):
        loss_fn = YOLOv6Loss(num_classes=80)
        preds = {'small': Tensor(np.random.rand(1, 85, 7, 7).astype(np.float32))}
        targets = [{'boxes': np.array([[100, 100, 200, 200]], dtype=np.float32), 'labels': np.array([0])}]
        loss, _ = loss_fn(preds, targets)
        assert loss.item() >= 0


class TestEncodeDecode:
    def test_encode(self):
        boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
        labels = np.array([0])
        targets = encode_targets_v6(boxes, labels, [], grid_sizes=[28, 14, 7], num_classes=80, image_size=224)
        assert 'scale_0' in targets
    
    def test_decode(self):
        preds = np.zeros((1, 85, 7, 7), dtype=np.float32)
        preds[0, 4, 3, 3] = 0.9
        preds[0, 5, 3, 3] = 0.9
        boxes, scores, ids = decode_predictions_v6(preds, conf_threshold=0.5, num_classes=80, image_size=224)
        assert len(boxes) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
