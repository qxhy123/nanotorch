"""
Unit tests for YOLO v2 model implementation.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from nanotorch.tensor import Tensor
from nanotorch.detection.yolo_v2 import (
    ConvBN,
    Darknet19,
    PassthroughLayer,
    YOLOv2Head,
    YOLOv2,
    YOLOv2Tiny,
    build_yolov2,
    YOLOv2Loss,
    YOLOv2LossSimple,
    encode_targets_v2,
    decode_predictions_v2
)


class TestConvBN:
    """Tests for ConvBN module."""
    
    def test_forward_shape(self):
        conv_bn = ConvBN(3, 32, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
        y = conv_bn(x)
        assert y.shape == (1, 32, 224, 224)
    
    def test_stride_downsample(self):
        conv_bn = ConvBN(32, 64, kernel_size=3, stride=2, padding=1)
        x = Tensor(np.random.randn(1, 32, 224, 224).astype(np.float32))
        y = conv_bn(x)
        assert y.shape == (1, 64, 112, 112)


class TestDarknet19:
    """Tests for Darknet-19 backbone."""
    
    def test_forward_output_shape(self):
        backbone = Darknet19(in_channels=3)
        x = Tensor(np.random.randn(1, 3, 416, 416).astype(np.float32))
        features = backbone(x)
        
        assert 'output' in features
        assert 'route1' in features
        assert 'route2' in features
        
        assert features['output'].shape == (1, 1024, 13, 13)
        assert features['route1'].shape == (1, 256, 52, 52)
        assert features['route2'].shape == (1, 512, 26, 26)
    
    def test_backbone_parameters(self):
        backbone = Darknet19()
        params = list(backbone.parameters())
        assert len(params) > 0


class TestPassthroughLayer:
    """Tests for PassthroughLayer."""
    
    def test_passthrough_shape(self):
        passthrough = PassthroughLayer(stride=2)
        x = Tensor(np.random.randn(1, 512, 26, 26).astype(np.float32))
        y = passthrough(x)
        
        assert y.shape == (1, 2048, 13, 13)
    
    def test_passthrough_different_sizes(self):
        passthrough = PassthroughLayer(stride=2)
        x = Tensor(np.random.randn(1, 64, 32, 32).astype(np.float32))
        y = passthrough(x)
        
        assert y.shape == (1, 256, 16, 16)


class TestYOLOv2Head:
    """Tests for YOLO v2 detection head."""
    
    def test_head_output_shape(self):
        head = YOLOv2Head(
            in_channels=1024,
            passthrough_channels=512,
            num_anchors=5,
            num_classes=20
        )
        
        x = Tensor(np.random.randn(1, 1024, 13, 13).astype(np.float32))
        passthrough = Tensor(np.random.randn(1, 512, 26, 26).astype(np.float32))
        
        output = head(x, passthrough)
        
        num_anchors = 5
        num_classes = 20
        expected_channels = num_anchors * (5 + num_classes)
        
        assert output.shape == (1, expected_channels, 13, 13)
    
    def test_head_output_range(self):
        head = YOLOv2Head(
            in_channels=1024,
            passthrough_channels=512,
            num_anchors=5,
            num_classes=20
        )
        
        x = Tensor(np.random.randn(1, 1024, 13, 13).astype(np.float32))
        passthrough = Tensor(np.random.randn(1, 512, 26, 26).astype(np.float32))
        
        output = head(x, passthrough)
        
        assert np.all(output.data >= 0)
        assert np.all(output.data <= 1)


class TestYOLOv2:
    """Tests for complete YOLO v2 model."""
    
    def test_model_forward_shape(self):
        model = YOLOv2(num_classes=20, input_size=416)
        x = Tensor(np.random.randn(1, 3, 416, 416).astype(np.float32))
        
        output = model(x)
        
        assert 'output' in output
        
        num_anchors = 5
        num_classes = 20
        expected_channels = num_anchors * (5 + num_classes)
        
        assert output['output'].shape == (1, expected_channels, 13, 13)
    
    def test_model_parameters(self):
        model = YOLOv2(num_classes=20)
        params = list(model.parameters())
        assert len(params) > 0
    
    def test_tiny_model_forward_shape(self):
        model = YOLOv2Tiny(num_classes=20, input_size=416)
        x = Tensor(np.random.randn(1, 3, 416, 416).astype(np.float32))
        
        output = model(x)
        
        assert 'output' in output
        
        num_anchors = 5
        num_classes = 20
        expected_channels = num_anchors * (5 + num_classes)
        
        assert output['output'].shape == (1, expected_channels, 12, 12)
    
    def test_build_yolov2_full(self):
        model = build_yolov2('full', num_classes=20)
        assert isinstance(model, YOLOv2)
    
    def test_build_yolov2_tiny(self):
        model = build_yolov2('tiny', num_classes=20)
        assert isinstance(model, YOLOv2Tiny)
    
    def test_different_input_sizes(self):
        model = YOLOv2(num_classes=20, input_size=416)
        x = Tensor(np.random.randn(1, 3, 416, 416).astype(np.float32))
        output = model(x)
        
        expected_channels = 5 * (5 + 20)
        assert output['output'].shape == (1, expected_channels, 13, 13)


class TestYOLOv2Loss:
    """Tests for YOLO v2 loss function."""
    
    def test_loss_forward_no_targets(self):
        loss_fn = YOLOv2Loss(num_classes=20)
        
        predictions = {
            'output': Tensor(np.random.randn(2, 125, 13, 13).astype(np.float32) * 0.1)
        }
        targets = [{'boxes': np.array([]), 'labels': np.array([])}]
        
        loss, loss_dict = loss_fn(predictions, targets)
        
        assert loss.item() >= 0
        assert 'total_loss' in loss_dict
    
    def test_loss_forward_with_targets(self):
        loss_fn = YOLOv2Loss(num_classes=20)
        
        predictions = {
            'output': Tensor(np.random.randn(1, 125, 13, 13).astype(np.float32) * 0.1)
        }
        targets = [{
            'boxes': np.array([[100, 100, 200, 200]], dtype=np.float32),
            'labels': np.array([0], dtype=np.int64)
        }]
        
        loss, loss_dict = loss_fn(predictions, targets)
        
        assert loss.item() >= 0
        assert 'coord_loss' in loss_dict
        assert 'obj_loss' in loss_dict
        assert 'noobj_loss' in loss_dict
        assert 'class_loss' in loss_dict
    
    def test_simple_loss(self):
        loss_fn = YOLOv2LossSimple(num_classes=20)
        
        predictions = {
            'output': Tensor(np.random.randn(1, 125, 13, 13).astype(np.float32) * 0.1)
        }
        targets = {
            'output': Tensor(np.zeros((1, 125, 13, 13), dtype=np.float32))
        }
        
        loss, loss_dict = loss_fn(predictions, targets)
        
        assert loss >= 0
        assert 'total_loss' in loss_dict


class TestEncodeDecode:
    """Tests for target encoding and prediction decoding."""
    
    def test_encode_targets(self):
        boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
        labels = np.array([0], dtype=np.int64)
        
        target = encode_targets_v2(
            boxes, labels,
            grid_size=13,
            num_anchors=5,
            num_classes=20
        )
        
        expected_channels = 5 * (5 + 20)
        assert target.shape == (expected_channels, 13, 13)
    
    def test_decode_predictions(self):
        predictions = np.random.rand(1, 125, 13, 13).astype(np.float32) * 0.1
        
        boxes, scores, class_ids = decode_predictions_v2(
            predictions,
            conf_threshold=0.1,
            num_anchors=5,
            num_classes=20
        )
        
        if len(boxes) > 0:
            assert boxes.shape[1] == 4
            assert len(scores) == len(boxes)
            assert len(class_ids) == len(boxes)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
