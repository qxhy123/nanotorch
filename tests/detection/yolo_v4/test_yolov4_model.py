"""
Unit tests for YOLO v4 model components.

Tests all major components:
- Mish activation
- CSPResBlock
- SPP module
- CSPDarknet53 backbone
- PANet neck
- YOLOHead
- YOLOv4 and YOLOv4Tiny models
- Loss functions
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from nanotorch.tensor import Tensor
from nanotorch.detection.yolo_v4 import (
    Mish,
    ConvBNMish,
    ConvBNLeaky,
    ResBlock,
    CSPResBlock,
    SPP,
    CSPDarknet53,
    PANet,
    YOLOHead,
    YOLOv4,
    YOLOv4Tiny,
    build_yolov4,
    YOLOv4Loss,
    YOLOv4LossSimple,
    encode_targets_v4,
    decode_predictions_v4
)


class TestMish:
    """Tests for Mish activation."""
    
    def test_mish_forward(self):
        mish = Mish()
        x = Tensor(np.random.randn(2, 64, 16, 16).astype(np.float32))
        y = mish(x)
        
        assert y.shape == x.shape
        assert not np.any(np.isnan(y.data))
    
    def test_mish_positive_values(self):
        mish = Mish()
        x = Tensor(np.ones((1, 1, 1, 1), dtype=np.float32) * 2.0)
        y = mish(x)
        
        # Mish(2) should be positive
        assert y.data[0, 0, 0, 0] > 0
    
    def test_mish_negative_values(self):
        mish = Mish()
        x = Tensor(np.ones((1, 1, 1, 1), dtype=np.float32) * -2.0)
        y = mish(x)
        
        # Mish(-2) should be negative but not zero
        assert y.data[0, 0, 0, 0] < 0


class TestConvBNMish:
    """Tests for ConvBNMish block."""
    
    def test_conv_bn_mish_forward(self):
        block = ConvBNMish(64, 128, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(2, 64, 16, 16).astype(np.float32))
        y = block(x)
        
        assert y.shape == (2, 128, 16, 16)
    
    def test_conv_bn_mish_stride(self):
        block = ConvBNMish(64, 128, kernel_size=3, stride=2, padding=1)
        x = Tensor(np.random.randn(2, 64, 16, 16).astype(np.float32))
        y = block(x)
        
        assert y.shape == (2, 128, 8, 8)


class TestCSPResBlock:
    """Tests for CSP Residual Block."""
    
    def test_csp_res_block_forward(self):
        block = CSPResBlock(64, 64, num_blocks=1)
        x = Tensor(np.random.randn(2, 64, 16, 16).astype(np.float32))
        y = block(x)
        
        assert y.shape == (2, 64, 16, 16)
    
    def test_csp_res_block_multi_blocks(self):
        block = CSPResBlock(64, 128, num_blocks=2)
        x = Tensor(np.random.randn(2, 64, 16, 16).astype(np.float32))
        y = block(x)
        
        assert y.shape == (2, 128, 16, 16)


class TestSPP:
    """Tests for Spatial Pyramid Pooling."""
    
    def test_spp_forward(self):
        spp = SPP(64, 128, pool_sizes=[5, 9, 13])
        x = Tensor(np.random.randn(2, 64, 16, 16).astype(np.float32))
        y = spp(x)
        
        assert y.shape == (2, 128, 16, 16)
    
    def test_spp_different_sizes(self):
        spp = SPP(128, 256, pool_sizes=[3, 5, 7])
        x = Tensor(np.random.randn(1, 128, 8, 8).astype(np.float32))
        y = spp(x)
        
        assert y.shape == (1, 256, 8, 8)


class TestCSPDarknet53:
    """Tests for CSPDarknet53 backbone."""
    
    def test_darknet_forward(self):
        backbone = CSPDarknet53(in_channels=3)
        x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
        features = backbone(x)
        
        assert 'scale1' in features
        assert 'scale2' in features
        assert 'scale3' in features
    
    def test_darknet_out_channels(self):
        backbone = CSPDarknet53(in_channels=3)
        
        assert len(backbone.out_channels) == 3
        assert backbone.out_channels == [256, 512, 512]


class TestPANet:
    """Tests for PANet neck."""
    
    def test_panet_forward(self):
        panet = PANet(in_channels=[512, 512, 256])
        
        features = {
            'scale1': Tensor(np.random.randn(1, 512, 7, 7).astype(np.float32)),
            'scale2': Tensor(np.random.randn(1, 512, 14, 14).astype(np.float32)),
            'scale3': Tensor(np.random.randn(1, 256, 28, 28).astype(np.float32))
        }
        
        output = panet(features)
        
        assert 'p3' in output
        assert 'p4' in output
        assert 'p5' in output


class TestYOLOHead:
    """Tests for YOLO v4 detection head."""
    
    def test_head_forward(self):
        head = YOLOHead(in_channels=512, num_anchors=3, num_classes=80)
        x = Tensor(np.random.randn(2, 512, 13, 13).astype(np.float32))
        y = head(x)
        
        # Output: (N, num_anchors * (5 + num_classes), H, W)
        assert y.shape == (2, 255, 13, 13)
    
    def test_head_output_range(self):
        head = YOLOHead(in_channels=256, num_anchors=3, num_classes=20)
        x = Tensor(np.random.randn(1, 256, 13, 13).astype(np.float32))
        y = head(x)
        
        # Output should be in [0, 1] range due to sigmoid
        assert y.data.min() >= 0
        assert y.data.max() <= 1


class TestYOLOv4:
    """Tests for complete YOLOv4 model."""
    
    def test_yolov4_forward(self):
        model = YOLOv4(num_classes=80, input_size=224)
        x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
        output = model(x)
        
        assert 'small' in output
        assert 'medium' in output
        assert 'large' in output
    
    def test_yolov4_parameters(self):
        model = YOLOv4(num_classes=80, input_size=224)
        params = list(model.parameters())
        
        assert len(params) > 0
        
        total_params = sum(p.data.size for p in params)
        assert total_params > 0


class TestYOLOv4Tiny:
    """Tests for YOLOv4Tiny model."""
    
    def test_tiny_forward(self):
        model = YOLOv4Tiny(num_classes=80, input_size=224)
        x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
        output = model(x)
        
        assert 'small' in output
        assert 'route' in output
    
    def test_tiny_parameters(self):
        model = YOLOv4Tiny(num_classes=80, input_size=224)
        params = list(model.parameters())
        
        assert len(params) > 0
        
        # Tiny should have fewer parameters than full
        full_model = YOLOv4(num_classes=80, input_size=224)
        tiny_params = sum(p.data.size for p in model.parameters())
        full_params = sum(p.data.size for p in full_model.parameters())
        
        assert tiny_params < full_params


class TestBuildYOLOv4:
    """Tests for build_yolov4 factory function."""
    
    def test_build_full_model(self):
        model = build_yolov4('full', num_classes=80, input_size=224)
        
        assert isinstance(model, YOLOv4)
    
    def test_build_tiny_model(self):
        model = build_yolov4('tiny', num_classes=80, input_size=224)
        
        assert isinstance(model, YOLOv4Tiny)


class TestYOLOv4Loss:
    """Tests for YOLOv4Loss."""
    
    def test_loss_creation(self):
        loss_fn = YOLOv4Loss(num_classes=80)
        
        assert loss_fn.num_classes == 80
    
    def test_loss_forward(self):
        loss_fn = YOLOv4Loss(num_classes=80)
        
        predictions = {
            'small': Tensor(np.random.rand(1, 255, 13, 13).astype(np.float32)),
            'medium': Tensor(np.random.rand(1, 255, 26, 26).astype(np.float32)),
            'large': Tensor(np.random.rand(1, 255, 52, 52).astype(np.float32))
        }
        
        targets = [
            {
                'boxes': np.array([[100, 100, 200, 200]], dtype=np.float32),
                'labels': np.array([0], dtype=np.int64)
            }
        ]
        
        loss, loss_dict = loss_fn(predictions, targets)
        
        assert 'coord_loss' in loss_dict
        assert 'obj_loss' in loss_dict
        assert 'noobj_loss' in loss_dict
        assert 'class_loss' in loss_dict


class TestYOLOv4LossSimple:
    """Tests for simplified YOLOv4 loss."""
    
    def test_simple_loss(self):
        loss_fn = YOLOv4LossSimple(num_classes=80)
        
        predictions = {
            'small': Tensor(np.random.rand(2, 255, 7, 7).astype(np.float32))
        }
        targets = {
            'small': Tensor(np.zeros((2, 255, 7, 7), dtype=np.float32))
        }
        
        loss, loss_dict = loss_fn(predictions, targets)
        
        assert isinstance(loss, float)
        assert loss >= 0


class TestEncodeTargetsV4:
    """Tests for target encoding."""
    
    def test_encode_single_object(self):
        anchors = [
            (12, 16), (19, 36), (40, 28),
            (36, 75), (76, 55), (72, 146),
            (142, 110), (192, 243), (459, 401)
        ]
        
        boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
        labels = np.array([0], dtype=np.int64)
        
        targets = encode_targets_v4(
            boxes=boxes,
            labels=labels,
            anchors=anchors,
            grid_sizes=[13, 26, 52],
            num_classes=80,
            image_size=416
        )
        
        assert 'scale_0' in targets
        assert 'scale_1' in targets
        assert 'scale_2' in targets
    
    def test_encode_empty(self):
        anchors = [
            (12, 16), (19, 36), (40, 28),
            (36, 75), (76, 55), (72, 146),
            (142, 110), (192, 243), (459, 401)
        ]
        
        boxes = np.array([], dtype=np.float32).reshape(0, 4)
        labels = np.array([], dtype=np.int64)
        
        targets = encode_targets_v4(
            boxes=boxes,
            labels=labels,
            anchors=anchors,
            grid_sizes=[13, 26, 52],
            num_classes=80,
            image_size=416
        )
        
        # Should still return valid shapes
        assert targets['scale_0'].shape == (3, 85, 13, 13)
        assert targets['scale_1'].shape == (3, 85, 26, 26)
        assert targets['scale_2'].shape == (3, 85, 52, 52)


class TestDecodePredictionsV4:
    """Tests for prediction decoding."""
    
    def test_decode_empty(self):
        predictions = np.zeros((3, 85, 13, 13), dtype=np.float32)
        anchors = [(142, 110), (192, 243), (459, 401)]
        
        boxes, scores, class_ids = decode_predictions_v4(
            predictions,
            anchors=anchors,
            conf_threshold=0.5,
            num_classes=80,
            image_size=416
        )
        
        assert len(boxes) == 0
        assert len(scores) == 0
        assert len(class_ids) == 0
    
    def test_decode_with_objects(self):
        predictions = np.zeros((3, 85, 13, 13), dtype=np.float32)
        predictions[0, 4, 5, 5] = 0.9
        predictions[0, 0, 5, 5] = 0.5
        predictions[0, 1, 5, 5] = 0.5
        predictions[0, 2, 5, 5] = 0.0
        predictions[0, 3, 5, 5] = 0.0
        predictions[0, 5, 5, 5] = 0.9
        
        anchors = [(142, 110), (192, 243), (459, 401)]
        
        boxes, scores, class_ids = decode_predictions_v4(
            predictions,
            anchors=anchors,
            conf_threshold=0.5,
            num_classes=80,
            image_size=416
        )
        
        assert len(boxes) >= 1
        assert len(scores) >= 1
        assert len(class_ids) >= 1


class TestEndToEnd:
    """End-to-end tests."""
    
    def test_tiny_model_pipeline(self):
        model = YOLOv4Tiny(num_classes=80, input_size=224)
        
        x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
        output = model(x)
        
        assert 'small' in output
        
        pred = output['small']
        assert pred.shape[0] == 1
        assert pred.shape[1] == 255
    
    def test_model_state_dict(self):
        model = YOLOv4Tiny(num_classes=80, input_size=224)
        
        state = model.state_dict()
        
        assert isinstance(state, dict)
        assert len(state) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
