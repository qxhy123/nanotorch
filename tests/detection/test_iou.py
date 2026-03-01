"""
Unit tests for IoU functions.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nanotorch.detection.iou import (
    iou,
    giou,
    diou,
    ciou,
    siou,
    compute_iou_loss,
    compute_iou_loss_vectorized
)
from nanotorch.tensor import Tensor


class TestIoU:
    
    def test_identical_boxes(self):
        boxes1 = np.array([[0, 0, 10, 10]], dtype=np.float32)
        boxes2 = np.array([[0, 0, 10, 10]], dtype=np.float32)
        
        result = iou(boxes1, boxes2)
        
        assert result.shape == (1, 1)
        assert result[0, 0] == pytest.approx(1.0, abs=1e-5)
    
    def test_no_overlap(self):
        boxes1 = np.array([[0, 0, 10, 10]], dtype=np.float32)
        boxes2 = np.array([[20, 20, 30, 30]], dtype=np.float32)
        
        result = iou(boxes1, boxes2)
        
        assert result[0, 0] == pytest.approx(0.0, abs=1e-5)
    
    def test_half_overlap(self):
        boxes1 = np.array([[0, 0, 10, 10]], dtype=np.float32)
        boxes2 = np.array([[0, 0, 10, 5]], dtype=np.float32)
        
        result = iou(boxes1, boxes2)
        
        expected = 50.0 / 100.0
        assert result[0, 0] == pytest.approx(expected, abs=1e-5)
    
    def test_pairwise_iou(self):
        boxes1 = np.array([[0, 0, 10, 10], [0, 0, 5, 5]], dtype=np.float32)
        boxes2 = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
        
        result = iou(boxes1, boxes2)
        
        assert result.shape == (2, 2)
        assert result[0, 0] == pytest.approx(1.0, abs=1e-5)
        assert result[1, 1] == pytest.approx(0.0, abs=1e-5)


class TestGIoU:
    
    def test_identical_boxes(self):
        boxes1 = np.array([[0, 0, 10, 10]], dtype=np.float32)
        boxes2 = np.array([[0, 0, 10, 10]], dtype=np.float32)
        
        result = giou(boxes1, boxes2)
        
        assert result[0, 0] == pytest.approx(1.0, abs=1e-5)
    
    def test_no_overlap_positive_area(self):
        boxes1 = np.array([[0, 0, 10, 10]], dtype=np.float32)
        boxes2 = np.array([[20, 20, 30, 30]], dtype=np.float32)
        
        result = giou(boxes1, boxes2)
        
        assert result[0, 0] < 0
    
    def test_giou_range(self):
        boxes1 = np.array([[0, 0, 10, 10]], dtype=np.float32)
        boxes2 = np.array([[50, 50, 60, 60]], dtype=np.float32)
        
        result = giou(boxes1, boxes2)
        
        assert -1.0 <= result[0, 0] <= 1.0


class TestDIoU:
    
    def test_identical_boxes(self):
        boxes1 = np.array([[0, 0, 10, 10]], dtype=np.float32)
        boxes2 = np.array([[0, 0, 10, 10]], dtype=np.float32)
        
        result = diou(boxes1, boxes2)
        
        assert result[0, 0] == pytest.approx(1.0, abs=1e-5)
    
    def test_diou_considers_center_distance(self):
        boxes1 = np.array([[0, 0, 10, 10]], dtype=np.float32)
        boxes2 = np.array([[5, 5, 15, 15]], dtype=np.float32)
        
        result_diou = diou(boxes1, boxes2)
        result_iou = iou(boxes1, boxes2)
        
        assert result_diou[0, 0] < result_iou[0, 0]


class TestCIoU:
    
    def test_identical_boxes(self):
        boxes1 = np.array([[0, 0, 10, 10]], dtype=np.float32)
        boxes2 = np.array([[0, 0, 10, 10]], dtype=np.float32)
        
        result = ciou(boxes1, boxes2)
        
        assert result[0, 0] == pytest.approx(1.0, abs=1e-5)
    
    def test_ciou_penalty_for_different_aspect_ratios(self):
        boxes1 = np.array([[0, 0, 10, 10]], dtype=np.float32)
        boxes2 = np.array([[0, 0, 20, 5]], dtype=np.float32)
        
        result_ciou = ciou(boxes1, boxes2)
        result_diou = diou(boxes1, boxes2)
        
        assert result_ciou[0, 0] <= result_diou[0, 0]


class TestSIoU:
    
    def test_identical_boxes(self):
        boxes1 = np.array([[0, 0, 10, 10]], dtype=np.float32)
        boxes2 = np.array([[0, 0, 10, 10]], dtype=np.float32)
        
        result = siou(boxes1, boxes2)
        
        assert result[0, 0] == pytest.approx(1.0, abs=1e-4)


class TestComputeIoULoss:
    
    def test_iou_loss_zero_for_identical(self):
        boxes1 = np.array([[0, 0, 10, 10]], dtype=np.float32)
        boxes2 = np.array([[0, 0, 10, 10]], dtype=np.float32)
        
        loss = compute_iou_loss(boxes1, boxes2, iou_type='iou')
        
        assert loss == pytest.approx(0.0, abs=1e-5)
    
    def test_iou_loss_positive_for_different(self):
        boxes1 = np.array([[0, 0, 10, 10]], dtype=np.float32)
        boxes2 = np.array([[5, 5, 15, 15]], dtype=np.float32)
        
        loss = compute_iou_loss(boxes1, boxes2, iou_type='iou')
        
        assert loss > 0
    
    def test_ciou_loss(self):
        boxes1 = np.array([[0, 0, 10, 10]], dtype=np.float32)
        boxes2 = np.array([[0, 0, 10, 10]], dtype=np.float32)
        
        loss = compute_iou_loss(boxes1, boxes2, iou_type='ciou')
        
        assert loss == pytest.approx(0.0, abs=1e-5)
    
    def test_vectorized_loss(self):
        boxes1 = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=np.float32)
        boxes2 = np.array([[0, 0, 10, 10], [10, 10, 20, 20]], dtype=np.float32)
        
        losses = compute_iou_loss_vectorized(boxes1, boxes2, iou_type='ciou')
        
        assert losses.shape == (2,)
        assert losses[0] == pytest.approx(0.0, abs=1e-5)
        assert losses[1] > 0
    
    def test_with_tensor_input(self):
        boxes1 = Tensor(np.array([[0, 0, 10, 10]], dtype=np.float32))
        boxes2 = Tensor(np.array([[0, 0, 10, 10]], dtype=np.float32))
        
        loss = compute_iou_loss(boxes1, boxes2, iou_type='iou')
        
        assert loss == pytest.approx(0.0, abs=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
