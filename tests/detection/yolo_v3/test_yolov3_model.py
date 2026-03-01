"""
Unit tests for YOLO v3 model components.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from nanotorch.tensor import Tensor
from nanotorch.detection.yolo_v3 import (
    ConvBN,
    ResidualBlock,
    Darknet53,
    FPN,
    YOLOHead,
    YOLOv3,
    YOLOv3Tiny,
    build_yolov3,
    YOLOv3Loss,
    YOLOv3LossSimple,
    encode_targets_v3,
    decode_predictions_v3
)


class TestConvBN:
    
    def test_conv_bn_forward(self):
        conv_bn = ConvBN(3, 32, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(1, 3, 32, 32).astype(np.float32))
        
        y = conv_bn(x)
        
        assert y.shape == (1, 32, 32, 32)
    
    def test_conv_bn_stride(self):
        conv_bn = ConvBN(32, 64, kernel_size=3, stride=2, padding=1)
        x = Tensor(np.random.randn(1, 32, 32, 32).astype(np.float32))
        
        y = conv_bn(x)
        
        assert y.shape == (1, 64, 16, 16)


class TestResidualBlock:
    
    def test_residual_forward(self):
        block = ResidualBlock(64)
        x = Tensor(np.random.randn(1, 64, 16, 16).astype(np.float32))
        
        y = block(x)
        
        assert y.shape == (1, 64, 16, 16)


class TestDarknet53:
    
    def test_darknet_forward(self):
        backbone = Darknet53(in_channels=3)
        x = Tensor(np.random.randn(1, 3, 416, 416).astype(np.float32))
        
        features = backbone(x)
        
        assert 'scale1' in features
        assert 'scale2' in features
        assert 'scale3' in features
        
        assert features['scale1'].shape[1] == 1024
        assert features['scale2'].shape[1] == 512
        assert features['scale3'].shape[1] == 256
    
    def test_darknet_out_channels(self):
        backbone = Darknet53(in_channels=3)
        
        assert backbone.out_channels == [256, 512, 1024]


class TestFPN:
    
    def test_fpn_forward(self):
        fpn = FPN([1024, 512, 256])
        
        features = {
            'scale1': Tensor(np.random.randn(1, 1024, 13, 13).astype(np.float32)),
            'scale2': Tensor(np.random.randn(1, 512, 26, 26).astype(np.float32)),
            'scale3': Tensor(np.random.randn(1, 256, 52, 52).astype(np.float32))
        }
        
        output = fpn(features)
        
        assert 'p3' in output
        assert 'p4' in output
        assert 'p5' in output


class TestYOLOHead:
    
    def test_head_forward(self):
        head = YOLOHead(128, num_anchors=3, num_classes=80)
        x = Tensor(np.random.randn(1, 128, 52, 52).astype(np.float32))
        
        y = head(x)
        
        expected_channels = 3 * (5 + 80)
        assert y.shape == (1, expected_channels, 52, 52)


class TestYOLOv3:
    
    def test_yolov3_forward(self):
        model = YOLOv3(num_classes=80, input_size=416)
        x = Tensor(np.random.randn(1, 3, 416, 416).astype(np.float32))
        
        output = model(x)
        
        assert 'small' in output
        assert 'medium' in output
        assert 'large' in output
    
    def test_yolov3_parameters(self):
        model = YOLOv3(num_classes=80)
        
        params = list(model.parameters())
        
        assert len(params) > 0


class TestYOLOv3Tiny:
    
    def test_tiny_forward(self):
        model = YOLOv3Tiny(num_classes=80, input_size=416)
        x = Tensor(np.random.randn(1, 3, 416, 416).astype(np.float32))
        
        output = model(x)
        
        assert 'small' in output
    
    def test_tiny_parameters(self):
        model = YOLOv3Tiny(num_classes=80)
        
        params = list(model.parameters())
        
        assert len(params) > 0


class TestBuildYOLOv3:
    
    def test_build_full_model(self):
        model = build_yolov3('full', num_classes=80, input_size=416)
        
        assert isinstance(model, YOLOv3)
    
    def test_build_tiny_model(self):
        model = build_yolov3('tiny', num_classes=80, input_size=416)
        
        assert isinstance(model, YOLOv3Tiny)


class TestYOLOv3Loss:
    
    def test_loss_creation(self):
        loss_fn = YOLOv3Loss(num_classes=80)
        
        assert loss_fn.num_classes == 80
    
    def test_loss_forward(self):
        loss_fn = YOLOv3Loss(num_classes=80)
        
        predictions = {
            'small': Tensor(np.random.randn(1, 255, 13, 13).astype(np.float32) * 0.1),
            'medium': Tensor(np.random.randn(1, 255, 26, 26).astype(np.float32) * 0.1),
            'large': Tensor(np.random.randn(1, 255, 52, 52).astype(np.float32) * 0.1)
        }
        targets = [{'boxes': np.array([[100, 100, 200, 200]], dtype=np.float32), 'labels': np.array([0])}]
        
        loss, loss_dict = loss_fn(predictions, targets)
        
        assert 'total_loss' in loss_dict


class TestYOLOv3LossSimple:
    
    def test_simple_loss(self):
        loss_fn = YOLOv3LossSimple(num_classes=80)
        
        predictions = {
            'small': Tensor(np.zeros((1, 255, 13, 13), dtype=np.float32)),
            'medium': Tensor(np.zeros((1, 255, 26, 26), dtype=np.float32))
        }
        targets = {
            'small': Tensor(np.zeros((1, 255, 13, 13), dtype=np.float32)),
            'medium': Tensor(np.zeros((1, 255, 26, 26), dtype=np.float32))
        }
        
        loss, loss_dict = loss_fn(predictions, targets)
        
        assert loss == 0.0


class TestEncodeTargetsV3:
    
    def test_encode_single_object(self):
        boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
        labels = np.array([0], dtype=np.int64)
        
        anchors = [
            (10, 13), (16, 30), (33, 23),
            (30, 61), (62, 45), (59, 119),
            (116, 90), (156, 198), (373, 326)
        ]
        
        targets = encode_targets_v3(boxes, labels, anchors, num_classes=80, image_size=416)
        
        assert 'scale_0' in targets
        assert 'scale_1' in targets
        assert 'scale_2' in targets
    
    def test_encode_empty(self):
        boxes = np.array([], dtype=np.float32).reshape(0, 4)
        labels = np.array([], dtype=np.int64)
        
        anchors = [
            (10, 13), (16, 30), (33, 23),
            (30, 61), (62, 45), (59, 119),
            (116, 90), (156, 198), (373, 326)
        ]
        
        targets = encode_targets_v3(boxes, labels, anchors, num_classes=80, image_size=416)
        
        assert np.count_nonzero(targets['scale_0']) == 0


class TestDecodePredictionsV3:
    
    def test_decode_empty(self):
        predictions = np.zeros((3, 85, 13, 13), dtype=np.float32)
        
        anchors = [(116, 90), (156, 198), (373, 326)]
        
        boxes, scores, class_ids = decode_predictions_v3(
            predictions, anchors, conf_threshold=0.5, num_classes=80, image_size=416
        )
        
        assert len(boxes) == 0
    
    def test_decode_with_objects(self):
        predictions = np.zeros((3, 85, 13, 13), dtype=np.float32)
        predictions[0, 4, 6, 6] = 0.8
        predictions[0, 0, 6, 6] = 0.5
        predictions[0, 1, 6, 6] = 0.5
        
        anchors = [(116, 90), (156, 198), (373, 326)]
        
        boxes, scores, class_ids = decode_predictions_v3(
            predictions, anchors, conf_threshold=0.5, num_classes=80, image_size=416
        )
        
        assert len(boxes) >= 1


class TestEndToEnd:
    
    def test_tiny_model_pipeline(self):
        model = YOLOv3Tiny(num_classes=20, input_size=224)
        x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
        
        output = model(x)
        
        assert 'small' in output
    
    def test_model_state_dict(self):
        model = YOLOv3Tiny(num_classes=80)
        
        state = model.state_dict()
        
        assert isinstance(state, dict)
        assert len(state) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
