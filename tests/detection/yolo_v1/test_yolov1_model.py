"""
Unit tests for YOLO v1 model components.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from nanotorch.tensor import Tensor
from nanotorch.detection.yolo_v1 import (
    ConvBlock,
    Darknet,
    YOLOv1Head,
    YOLOv1,
    YOLOv1Tiny,
    build_yolov1,
    YOLOv1Loss,
    YOLOv1LossSimple,
    encode_targets,
    decode_predictions
)


class TestConvBlock:
    
    def test_conv_block_forward(self):
        conv_block = ConvBlock(3, 16, kernel_size=3, stride=1, padding=1)
        x = Tensor(np.random.randn(1, 3, 32, 32).astype(np.float32))
        
        y = conv_block(x)
        
        assert y.shape == (1, 16, 32, 32)
    
    def test_conv_block_stride_2(self):
        conv_block = ConvBlock(3, 16, kernel_size=3, stride=2, padding=1)
        x = Tensor(np.random.randn(1, 3, 32, 32).astype(np.float32))
        
        y = conv_block(x)
        
        assert y.shape == (1, 16, 16, 16)
    
    def test_conv_block_large_kernel(self):
        conv_block = ConvBlock(3, 64, kernel_size=7, stride=2, padding=3)
        x = Tensor(np.random.randn(1, 3, 448, 448).astype(np.float32))
        
        y = conv_block(x)
        
        assert y.shape == (1, 64, 224, 224)


class TestDarknet:
    
    def test_darknet_forward(self):
        darknet = Darknet(in_channels=3)
        x = Tensor(np.random.randn(1, 3, 448, 448).astype(np.float32))
        
        y = darknet(x)
        
        assert y.shape[0] == 1
        assert y.shape[1] == 1024
        assert y.shape[2] == 7
        assert y.shape[3] == 7
    
    def test_darknet_output_channels(self):
        darknet = Darknet(in_channels=3)
        
        assert darknet.out_channels == 1024


class TestYOLOv1Head:
    
    def test_head_output_shape(self):
        head = YOLOv1Head(
            in_channels=1024,
            hidden_dim=4096,
            S=7,
            B=2,
            C=20
        )
        
        x = Tensor(np.random.randn(1, 1024, 7, 7).astype(np.float32))
        y = head(x)
        
        expected_dim = 7 * 7 * (2 * 5 + 20)
        assert y.shape == (1, expected_dim)
    
    def test_head_reshape(self):
        head = YOLOv1Head(
            in_channels=1024,
            hidden_dim=4096,
            S=7,
            B=2,
            C=20
        )
        
        x = Tensor(np.random.randn(2, 1024, 7, 7).astype(np.float32))
        y = head(x)
        reshaped = head.reshape_output(y)
        
        assert reshaped.shape == (2, 7, 7, 30)


class TestYOLOv1:
    
    def test_yolov1_forward(self):
        model = YOLOv1(input_size=448, S=7, B=2, C=20)
        x = Tensor(np.random.randn(1, 3, 448, 448).astype(np.float32))
        
        y = model(x)
        
        expected_dim = 7 * 7 * 30
        assert y.shape == (1, expected_dim)
    
    def test_yolov1_predict(self):
        model = YOLOv1(input_size=448, S=7, B=2, C=20)
        x = Tensor(np.random.randn(1, 3, 448, 448).astype(np.float32))
        
        result = model.predict(x)
        
        assert 'output' in result
        assert 'reshaped' in result
        assert result['reshaped'].shape == (1, 7, 7, 30)
    
    def test_yolov1_batch_forward(self):
        model = YOLOv1(input_size=448, S=7, B=2, C=20)
        x = Tensor(np.random.randn(4, 3, 448, 448).astype(np.float32))
        
        y = model(x)
        
        expected_dim = 7 * 7 * 30
        assert y.shape == (4, expected_dim)


class TestYOLOv1Tiny:
    
    def test_tiny_forward(self):
        model = YOLOv1Tiny(input_size=224, S=7, B=2, C=20)
        x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
        
        y = model(x)
        
        expected_dim = 7 * 7 * 30
        assert y.shape == (1, expected_dim)
    
    def test_tiny_parameters(self):
        model = YOLOv1Tiny(input_size=224, S=7, B=2, C=20)
        
        params = list(model.parameters())
        
        assert len(params) > 0


class TestBuildYOLOv1:
    
    def test_build_full_model(self):
        model = build_yolov1('full', input_size=448, S=7, B=2, C=20)
        
        assert isinstance(model, YOLOv1)
    
    def test_build_tiny_model(self):
        model = build_yolov1('tiny', input_size=224, S=7, B=2, C=20)
        
        assert isinstance(model, YOLOv1Tiny)


class TestEncodeTargets:
    
    def test_encode_single_object(self):
        boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
        labels = np.array([0], dtype=np.int64)
        
        target = encode_targets(boxes, labels, S=7, B=2, C=20, image_size=448)
        
        assert target.shape == (7, 7, 30)
        assert np.count_nonzero(target) > 0
    
    def test_encode_multiple_objects(self):
        boxes = np.array([
            [100, 100, 200, 200],
            [250, 250, 350, 350]
        ], dtype=np.float32)
        labels = np.array([0, 5], dtype=np.int64)
        
        target = encode_targets(boxes, labels, S=7, B=2, C=20, image_size=448)
        
        assert target.shape == (7, 7, 30)
    
    def test_encode_empty(self):
        boxes = np.array([], dtype=np.float32).reshape(0, 4)
        labels = np.array([], dtype=np.int64)
        
        target = encode_targets(boxes, labels, S=7, B=2, C=20, image_size=448)
        
        assert target.shape == (7, 7, 30)
        assert np.count_nonzero(target) == 0


class TestDecodePredictions:
    
    def test_decode_empty_predictions(self):
        predictions = np.zeros((7, 7, 30), dtype=np.float32)
        
        boxes, scores, class_ids = decode_predictions(predictions, conf_threshold=0.5)
        
        assert len(boxes) == 0
        assert len(scores) == 0
        assert len(class_ids) == 0
    
    def test_decode_with_objects(self):
        predictions = np.zeros((7, 7, 30), dtype=np.float32)
        
        predictions[3, 3, 0] = 0.5
        predictions[3, 3, 1] = 0.5
        predictions[3, 3, 2] = 0.1
        predictions[3, 3, 3] = 0.1
        predictions[3, 3, 4] = 0.8
        
        predictions[3, 3, 10 + 5] = 1.0
        
        boxes, scores, class_ids = decode_predictions(predictions, conf_threshold=0.5)
        
        assert len(boxes) >= 1
    
    def test_decode_confidence_threshold(self):
        predictions = np.zeros((7, 7, 30), dtype=np.float32)
        
        predictions[3, 3, 4] = 0.3
        
        boxes, scores, class_ids = decode_predictions(predictions, conf_threshold=0.5)
        
        assert len(boxes) == 0


class TestYOLOv1Loss:
    
    def test_loss_zero_for_identical(self):
        loss_fn = YOLOv1Loss(S=7, B=2, C=20)
        
        predictions = Tensor(np.zeros((1, 7, 7, 30), dtype=np.float32))
        targets = Tensor(np.zeros((1, 7, 7, 30), dtype=np.float32))
        
        loss, loss_dict = loss_fn(predictions, targets)
        
        assert loss_dict['total_loss'] == 0.0
    
    def test_loss_positive_for_different(self):
        loss_fn = YOLOv1Loss(S=7, B=2, C=20)
        
        predictions = Tensor(np.random.randn(1, 7, 7, 30).astype(np.float32) * 0.1)
        targets = Tensor(np.zeros((1, 7, 7, 30), dtype=np.float32))
        
        loss, loss_dict = loss_fn(predictions, targets)
        
        assert loss_dict['total_loss'] >= 0
    
    def test_loss_dict_keys(self):
        loss_fn = YOLOv1Loss(S=7, B=2, C=20)
        
        predictions = Tensor(np.zeros((1, 7, 7, 30), dtype=np.float32))
        targets = Tensor(np.zeros((1, 7, 7, 30), dtype=np.float32))
        
        loss, loss_dict = loss_fn(predictions, targets)
        
        assert 'coord_loss' in loss_dict
        assert 'obj_conf_loss' in loss_dict
        assert 'noobj_conf_loss' in loss_dict
        assert 'class_loss' in loss_dict
        assert 'total_loss' in loss_dict


class TestYOLOv1LossSimple:
    
    def test_simple_loss(self):
        loss_fn = YOLOv1LossSimple()
        
        predictions = Tensor(np.array([[1.0, 2.0]], dtype=np.float32))
        targets = Tensor(np.array([[1.0, 2.0]], dtype=np.float32))
        
        loss, loss_dict = loss_fn(predictions, targets)
        
        assert loss == 0.0
    
    def test_simple_loss_mse(self):
        loss_fn = YOLOv1LossSimple()
        
        predictions = Tensor(np.array([[1.0, 2.0]], dtype=np.float32))
        targets = Tensor(np.array([[2.0, 3.0]], dtype=np.float32))
        
        loss, loss_dict = loss_fn(predictions, targets)
        
        expected = np.mean(np.array([(1.0 - 2.0) ** 2, (2.0 - 3.0) ** 2]))
        assert abs(loss - expected) < 1e-5


class TestEndToEnd:
    
    def test_tiny_model_loss_pipeline(self):
        model = build_yolov1('tiny', input_size=224, S=7, B=2, C=20)
        loss_fn = YOLOv1LossSimple()
        
        x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
        
        output = model(x)
        
        targets = Tensor(np.zeros((1, 1470), dtype=np.float32))
        
        loss, loss_dict = loss_fn(output, targets)
        
        assert loss >= 0
    
    def test_encode_decode_roundtrip(self):
        boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
        labels = np.array([5], dtype=np.int64)
        
        target = encode_targets(boxes, labels, S=7, B=2, C=20, image_size=448)
        
        decoded_boxes, scores, class_ids = decode_predictions(target, conf_threshold=0.5)
        
        assert len(decoded_boxes) >= 1
        assert len(scores) >= 1
        assert len(class_ids) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
