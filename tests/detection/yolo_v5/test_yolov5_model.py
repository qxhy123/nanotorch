"""
Unit tests for YOLO v5 model components.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from nanotorch.tensor import Tensor
from nanotorch.detection.yolo_v5 import (
    ConvBN,
    Bottleneck,
    C3,
    SPPF,
    Backbone,
    Neck,
    DetectHead,
    YOLOv5,
    YOLOv5Nano,
    YOLOv5Small,
    build_yolov5,
    YOLOv5Loss,
    YOLOv5LossSimple,
    encode_targets_v5,
    decode_predictions_v5
)


class TestConvBN:
    def test_conv_bn_forward(self):
        block = ConvBN(64, 128, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(2, 64, 16, 16).astype(np.float32))
        y = block(x)
        assert y.shape == (2, 128, 16, 16)
    
    def test_conv_bn_stride(self):
        block = ConvBN(64, 128, kernel_size=3, stride=2, padding=1)
        x = Tensor(np.random.randn(2, 64, 16, 16).astype(np.float32))
        y = block(x)
        assert y.shape == (2, 128, 8, 8)


class TestBottleneck:
    def test_bottleneck_forward(self):
        block = Bottleneck(64, 64, shortcut=True)
        x = Tensor(np.random.randn(2, 64, 16, 16).astype(np.float32))
        y = block(x)
        assert y.shape == (2, 64, 16, 16)
    
    def test_bottleneck_no_shortcut(self):
        block = Bottleneck(64, 128, shortcut=False)
        x = Tensor(np.random.randn(2, 64, 16, 16).astype(np.float32))
        y = block(x)
        assert y.shape == (2, 128, 16, 16)


class TestC3:
    def test_c3_forward(self):
        c3 = C3(64, 64, num_blocks=1)
        x = Tensor(np.random.randn(2, 64, 16, 16).astype(np.float32))
        y = c3(x)
        assert y.shape == (2, 64, 16, 16)
    
    def test_c3_multi_blocks(self):
        c3 = C3(64, 128, num_blocks=2)
        x = Tensor(np.random.randn(2, 64, 16, 16).astype(np.float32))
        y = c3(x)
        assert y.shape == (2, 128, 16, 16)


class TestSPPF:
    def test_sppf_forward(self):
        sppf = SPPF(64, 128, kernel_size=5)
        x = Tensor(np.random.randn(2, 64, 16, 16).astype(np.float32))
        y = sppf(x)
        assert y.shape == (2, 128, 16, 16)


class TestBackbone:
    def test_backbone_forward(self):
        backbone = Backbone(in_channels=3, version='s')
        x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
        features = backbone(x)
        
        assert 'scale1' in features
        assert 'scale2' in features
        assert 'scale3' in features
    
    def test_backbone_out_channels(self):
        backbone = Backbone(in_channels=3, version='s')
        assert len(backbone.out_channels) == 3


class TestNeck:
    def test_neck_forward(self):
        neck = Neck(in_channels=[128, 256, 512])
        
        features = {
            'scale1': Tensor(np.random.randn(1, 512, 7, 7).astype(np.float32)),
            'scale2': Tensor(np.random.randn(1, 256, 14, 14).astype(np.float32)),
            'scale3': Tensor(np.random.randn(1, 128, 28, 28).astype(np.float32))
        }
        
        output = neck(features)
        
        assert 'p3' in output
        assert 'p4' in output
        assert 'p5' in output


class TestDetectHead:
    def test_head_forward(self):
        head = DetectHead(in_channels=512, num_anchors=3, num_classes=80)
        x = Tensor(np.random.randn(2, 512, 13, 13).astype(np.float32))
        y = head(x)
        assert y.shape == (2, 255, 13, 13)
    
    def test_head_output_range(self):
        head = DetectHead(in_channels=256, num_anchors=3, num_classes=20)
        x = Tensor(np.random.randn(1, 256, 13, 13).astype(np.float32))
        y = head(x)
        assert y.data.min() >= 0
        assert y.data.max() <= 1


class TestYOLOv5:
    def test_yolov5_forward(self):
        model = YOLOv5(num_classes=80, input_size=224, version='s')
        x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
        output = model(x)
        
        assert 'small' in output
        assert 'medium' in output
        assert 'large' in output
    
    def test_yolov5_parameters(self):
        model = YOLOv5(num_classes=80, input_size=224, version='s')
        params = list(model.parameters())
        assert len(params) > 0


class TestYOLOv5Nano:
    def test_nano_forward(self):
        model = YOLOv5Nano(num_classes=80, input_size=224)
        x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
        output = model(x)
        assert 'small' in output


class TestYOLOv5Small:
    def test_small_forward(self):
        model = YOLOv5Small(num_classes=80, input_size=224)
        x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
        output = model(x)
        assert 'small' in output


class TestBuildYOLOv5:
    def test_build_nano(self):
        model = build_yolov5('n', num_classes=80, input_size=224)
        assert isinstance(model, YOLOv5)
    
    def test_build_small(self):
        model = build_yolov5('s', num_classes=80, input_size=224)
        assert isinstance(model, YOLOv5)


class TestYOLOv5Loss:
    def test_loss_creation(self):
        loss_fn = YOLOv5Loss(num_classes=80)
        assert loss_fn.num_classes == 80
    
    def test_loss_forward(self):
        loss_fn = YOLOv5Loss(num_classes=80)
        
        predictions = {
            'small': Tensor(np.random.rand(1, 255, 13, 13).astype(np.float32)),
            'medium': Tensor(np.random.rand(1, 255, 26, 26).astype(np.float32)),
            'large': Tensor(np.random.rand(1, 255, 52, 52).astype(np.float32))
        }
        
        targets = [{'boxes': np.array([[100, 100, 200, 200]], dtype=np.float32), 'labels': np.array([0])}]
        
        loss, loss_dict = loss_fn(predictions, targets)
        
        assert 'box_loss' in loss_dict
        assert 'obj_loss' in loss_dict


class TestYOLOv5LossSimple:
    def test_simple_loss(self):
        loss_fn = YOLOv5LossSimple(num_classes=80)
        
        predictions = {'small': Tensor(np.random.rand(2, 255, 7, 7).astype(np.float32))}
        targets = {'small': Tensor(np.zeros((2, 255, 7, 7), dtype=np.float32))}
        
        loss, _ = loss_fn(predictions, targets)
        assert isinstance(loss, float)


class TestEncodeTargetsV5:
    def test_encode_single_object(self):
        anchors = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]
        
        boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
        labels = np.array([0], dtype=np.int64)
        
        targets = encode_targets_v5(boxes, labels, anchors, grid_sizes=[80, 40, 20], num_classes=80, image_size=640)
        
        assert 'scale_0' in targets
        assert 'scale_1' in targets
        assert 'scale_2' in targets
    
    def test_encode_empty(self):
        anchors = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]
        
        boxes = np.array([], dtype=np.float32).reshape(0, 4)
        labels = np.array([], dtype=np.int64)
        
        targets = encode_targets_v5(boxes, labels, anchors, grid_sizes=[80, 40, 20], num_classes=80, image_size=640)
        
        assert targets['scale_0'].shape == (3, 85, 80, 80)


class TestDecodePredictionsV5:
    def test_decode_empty(self):
        predictions = np.zeros((3, 85, 13, 13), dtype=np.float32)
        anchors = [(116, 90), (156, 198), (373, 326)]
        
        boxes, scores, class_ids = decode_predictions_v5(predictions, anchors=anchors, conf_threshold=0.5, num_classes=80, image_size=640)
        
        assert len(boxes) == 0
    
    def test_decode_with_objects(self):
        predictions = np.zeros((3, 85, 13, 13), dtype=np.float32)
        predictions[0, 4, 5, 5] = 0.9
        predictions[0, 0, 5, 5] = 0.5
        predictions[0, 1, 5, 5] = 0.5
        predictions[0, 5, 5, 5] = 0.9
        
        anchors = [(116, 90), (156, 198), (373, 326)]
        
        boxes, scores, class_ids = decode_predictions_v5(predictions, anchors=anchors, conf_threshold=0.5, num_classes=80, image_size=640)
        
        assert len(boxes) >= 1


class TestEndToEnd:
    def test_model_pipeline(self):
        model = YOLOv5(num_classes=80, input_size=224, version='s')
        x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
        output = model(x)
        
        assert 'small' in output
        pred = output['small']
        assert pred.shape[0] == 1
    
    def test_model_state_dict(self):
        model = YOLOv5(num_classes=80, input_size=224, version='s')
        state = model.state_dict()
        
        assert isinstance(state, dict)
        assert len(state) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
