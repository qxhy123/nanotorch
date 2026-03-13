"""
Integration tests for YOLO v12 detection module.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nanotorch.tensor import Tensor
from nanotorch.detection import (
    Conv, C2f, SPPF, Bottleneck,
    YOLOBackbone, build_backbone,
    YOLONeck, build_neck,
    YOLOHead, build_head,
    CIoULoss, SimpleYOLOLoss,
    box_iou, nms, batched_nms,
    xyxy_to_xywh, xywh_to_xyxy
)


class TestConvBlock:
    
    def test_conv_forward(self):
        conv = Conv(3, 16, kernel_size=3, stride=1)
        x = Tensor(np.random.randn(1, 3, 32, 32).astype(np.float32))
        
        y = conv(x)
        
        assert y.shape == (1, 16, 32, 32)
    
    def test_conv_stride2(self):
        conv = Conv(3, 16, kernel_size=3, stride=2)
        x = Tensor(np.random.randn(1, 3, 32, 32).astype(np.float32))
        
        y = conv(x)
        
        assert y.shape == (1, 16, 16, 16)


class TestC2f:
    
    def test_c2f_forward(self):
        c2f = C2f(32, 32, num_bottlenecks=2)
        x = Tensor(np.random.randn(1, 32, 16, 16).astype(np.float32))
        
        y = c2f(x)
        
        assert y.shape[0] == 1
        assert y.shape[1] == 32
    
    def test_c2f_channel_change(self):
        c2f = C2f(32, 64, num_bottlenecks=2)
        x = Tensor(np.random.randn(1, 32, 16, 16).astype(np.float32))
        
        y = c2f(x)
        
        assert y.shape[1] == 64


class TestSPPF:
    
    def test_sppf_forward(self):
        sppf = SPPF(64, 64, kernel_size=5)
        x = Tensor(np.random.randn(1, 64, 8, 8).astype(np.float32))
        
        y = sppf(x)
        
        assert y.shape == (1, 64, 8, 8)


class TestBackbone:
    
    def test_backbone_forward(self):
        backbone = YOLOBackbone(
            in_channels=3,
            width_mult=0.25,
            depth_mult=0.34,
            use_attention=False
        )
        x = Tensor(np.random.randn(1, 3, 256, 256).astype(np.float32))
        
        features = backbone(x)
        
        assert 'p3' in features
        assert 'p4' in features
        assert 'p5' in features
        
        assert features['p3'].shape[2] == 32
        assert features['p4'].shape[2] == 16
        assert features['p5'].shape[2] == 8
    
    def test_build_backbone_sizes(self):
        for size in ['n', 's', 'm']:
            backbone = build_backbone(model_size=size, use_attention=False)
            
            assert backbone is not None
            assert hasattr(backbone, 'out_channels')


class TestNeck:
    
    def test_neck_forward(self):
        backbone = build_backbone(model_size='n', use_attention=False)
        neck = build_neck(
            neck_type='panet',
            in_channels=backbone.out_channels
        )
        
        x = Tensor(np.random.randn(1, 3, 256, 256).astype(np.float32))
        features = backbone(x)
        
        fused = neck(features)
        
        assert 'p3' in fused
        assert 'p4' in fused
        assert 'p5' in fused


class TestHead:
    
    def test_head_forward(self):
        in_channels = {'p3': 32, 'p4': 64, 'p5': 128}
        head = build_head(
            head_type='decoupled',
            in_channels=in_channels,
            num_classes=10
        )
        
        features = {
            'p3': Tensor(np.random.randn(1, 32, 32, 32).astype(np.float32)),
            'p4': Tensor(np.random.randn(1, 64, 16, 16).astype(np.float32)),
            'p5': Tensor(np.random.randn(1, 128, 8, 8).astype(np.float32)),
        }
        
        predictions = head(features)
        
        assert 'p3' in predictions
        assert 'p4' in predictions
        assert 'p5' in predictions
        
        for scale, (box_pred, cls_pred) in predictions.items():
            assert box_pred is not None
            assert cls_pred is not None


class TestHeadDecode:
    
    def test_decode_predictions_shapes(self):
        head = build_head(
            head_type='decoupled',
            in_channels={'p3': 32, 'p4': 64, 'p5': 128},
            num_classes=3
        )
        predictions = {
            'p3': (
                Tensor(np.zeros((1, 4 * head.reg_max, 2, 2), dtype=np.float32)),
                Tensor(np.full((1, 3, 2, 2), 8.0, dtype=np.float32)),
            )
        }

        boxes, scores, class_ids = head.decode_predictions(
            predictions,
            input_size=(16, 16),
            conf_threshold=0.5,
        )

        assert boxes.ndim == 2
        assert boxes.shape[1] == 4
        assert scores.ndim == 1
        assert class_ids.ndim == 1
        assert len(boxes) == len(scores) == len(class_ids)

    def test_decode_predictions_empty(self):
        head = build_head(
            head_type='decoupled',
            in_channels={'p3': 32, 'p4': 64, 'p5': 128},
            num_classes=3
        )
        predictions = {
            'p3': (
                Tensor(np.zeros((1, 4 * head.reg_max, 2, 2), dtype=np.float32)),
                Tensor(np.full((1, 3, 2, 2), -20.0, dtype=np.float32)),
            )
        }

        boxes, scores, class_ids = head.decode_predictions(
            predictions,
            input_size=(16, 16),
            conf_threshold=0.99,
        )

        assert boxes.shape == (0, 4)
        assert scores.shape == (0,)
        assert class_ids.shape == (0,)


class TestLosses:
    
    def test_ciou_loss(self):
        loss_fn = CIoULoss()
        
        pred = Tensor(np.array([[0, 0, 10, 10]], dtype=np.float32))
        target = Tensor(np.array([[0, 0, 10, 10]], dtype=np.float32))
        
        loss = loss_fn(pred, target)
        
        assert loss == pytest.approx(0.0, abs=1e-4)
    
    def test_simple_loss(self):
        loss_fn = SimpleYOLOLoss(num_classes=10)
        
        pred_boxes = Tensor(np.array([[0, 0, 10, 10]], dtype=np.float32))
        pred_classes = Tensor(np.random.randn(1, 10).astype(np.float32))
        target_boxes = Tensor(np.array([[0, 0, 10, 10]], dtype=np.float32))
        target_classes = Tensor(np.array([0], dtype=np.int64))
        
        loss, loss_dict = loss_fn(
            pred_boxes, pred_classes,
            target_boxes, target_classes
        )
        
        assert isinstance(loss, float)
        assert 'box_loss' in loss_dict
        assert 'cls_loss' in loss_dict


class TestEndToEnd:
    
    def test_full_model_forward(self):
        backbone = build_backbone(model_size='n', use_attention=False)
        neck = build_neck(neck_type='panet', in_channels=backbone.out_channels)
        head = build_head(
            head_type='decoupled',
            in_channels=neck.out_channels,
            num_classes=10
        )
        
        x = Tensor(np.random.randn(1, 3, 256, 256).astype(np.float32))
        
        features = backbone(x)
        features = neck(features)
        predictions = head(features)
        
        for scale_name in ['p3', 'p4', 'p5']:
            assert scale_name in predictions
            box_pred, cls_pred = predictions[scale_name]
            
            assert box_pred is not None
            assert cls_pred is not None
    
    def test_detection_pipeline(self):
        pred_boxes = np.array([
            [10, 10, 50, 50],
            [15, 15, 55, 55],
            [100, 100, 150, 150]
        ], dtype=np.float32)
        pred_scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
        pred_labels = np.array([0, 0, 1], dtype=np.int64)
        
        keep = batched_nms(pred_boxes, pred_scores, pred_labels, iou_threshold=0.5)
        
        assert len(keep) >= 2
        
        filtered_boxes = pred_boxes[keep]
        filtered_scores = pred_scores[keep]
        
        assert len(filtered_boxes) == len(filtered_scores)


class TestBboxOperations:
    
    def test_conversion_roundtrip(self):
        original = np.array([[10, 20, 30, 40]], dtype=np.float32)
        
        xywh = xyxy_to_xywh(original)
        recovered = xywh_to_xyxy(xywh)
        
        np.testing.assert_array_almost_equal(original, recovered)
    
    def test_iou_matrix(self):
        boxes1 = np.array([
            [0, 0, 10, 10],
            [20, 20, 30, 30]
        ], dtype=np.float32)
        boxes2 = np.array([
            [0, 0, 10, 10],
            [5, 5, 15, 15],
            [50, 50, 60, 60]
        ], dtype=np.float32)
        
        iou_matrix = box_iou(boxes1, boxes2)
        
        assert iou_matrix.shape == (2, 3)
        assert iou_matrix[0, 0] == pytest.approx(1.0, abs=1e-5)
        assert iou_matrix[1, 2] == pytest.approx(0.0, abs=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
