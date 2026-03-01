"""
Unit tests for NMS functions.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nanotorch.detection.nms import (
    nms,
    batched_nms,
    soft_nms,
    postprocess_detections
)
from nanotorch.tensor import Tensor


class TestNMS:
    
    def test_empty_input(self):
        boxes = np.array([], dtype=np.float32).reshape(0, 4)
        scores = np.array([], dtype=np.float32)
        
        result = nms(boxes, scores, 0.5)
        
        assert len(result) == 0
    
    def test_single_box(self):
        boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        
        result = nms(boxes, scores, 0.5)
        
        assert len(result) == 1
        assert result[0] == 0
    
    def test_no_suppression_low_iou(self):
        boxes = np.array([
            [0, 0, 10, 10],
            [50, 50, 60, 60]
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        
        result = nms(boxes, scores, 0.5)
        
        assert len(result) == 2
    
    def test_suppression_high_iou(self):
        boxes = np.array([
            [0, 0, 10, 10],
            [1, 1, 11, 11]
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        
        result = nms(boxes, scores, 0.5)
        
        assert len(result) == 1
        assert result[0] == 0
    
    def test_keeps_highest_score(self):
        boxes = np.array([
            [0, 0, 10, 10],
            [1, 1, 11, 11]
        ], dtype=np.float32)
        scores = np.array([0.7, 0.9], dtype=np.float32)
        
        result = nms(boxes, scores, 0.5)
        
        assert len(result) == 1
        assert result[0] == 1
    
    def test_multiple_overlapping_groups(self):
        boxes = np.array([
            [0, 0, 10, 10],
            [1, 1, 11, 11],
            [50, 50, 60, 60],
            [51, 51, 61, 61]
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.85, 0.95], dtype=np.float32)
        
        result = nms(boxes, scores, 0.5)
        
        assert len(result) == 2
        assert 0 in result or 1 in result
        assert 3 in result
    
    def test_with_tensor_input(self):
        boxes = Tensor(np.array([[0, 0, 10, 10]], dtype=np.float32))
        scores = Tensor(np.array([0.9], dtype=np.float32))
        
        result = nms(boxes, scores, 0.5)
        
        assert len(result) == 1


class TestBatchedNMS:
    
    def test_different_classes_not_suppressed(self):
        boxes = np.array([
            [0, 0, 10, 10],
            [2, 2, 12, 12]
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        class_ids = np.array([0, 1], dtype=np.int64)
        
        result = batched_nms(boxes, scores, class_ids, 0.5)
        
        assert len(result) == 2
    
    def test_same_class_suppressed(self):
        boxes = np.array([
            [0, 0, 10, 10],
            [1, 1, 11, 11]
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        class_ids = np.array([0, 0], dtype=np.int64)
        
        result = batched_nms(boxes, scores, class_ids, 0.5)
        
        assert len(result) == 1
    
    def test_empty_input(self):
        boxes = np.array([], dtype=np.float32).reshape(0, 4)
        scores = np.array([], dtype=np.float32)
        class_ids = np.array([], dtype=np.int64)
        
        result = batched_nms(boxes, scores, class_ids, 0.5)
        
        assert len(result) == 0


class TestSoftNMS:
    
    def test_empty_input(self):
        boxes = np.array([], dtype=np.float32).reshape(0, 4)
        scores = np.array([], dtype=np.float32)
        
        indices, new_scores = soft_nms(boxes, scores, 0.5)
        
        assert len(indices) == 0
        assert len(new_scores) == 0
    
    def test_gaussian_decay(self):
        boxes = np.array([
            [0, 0, 10, 10],
            [2, 2, 12, 12]
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        
        indices, new_scores = soft_nms(boxes, scores, 0.5, method='gaussian')
        
        assert len(indices) == 2
        assert new_scores[0] == pytest.approx(0.9, abs=0.1)
        assert new_scores[1] < 0.8
    
    def test_linear_decay(self):
        boxes = np.array([
            [0, 0, 10, 10],
            [2, 2, 12, 12]
        ], dtype=np.float32)
        scores = np.array([0.9, 0.8], dtype=np.float32)
        
        indices, new_scores = soft_nms(boxes, scores, 0.5, method='linear')
        
        assert len(indices) >= 1


class TestPostprocessDetections:
    
    def test_empty_predictions(self):
        predictions = np.array([], dtype=np.float32).reshape(0, 6)
        
        boxes, scores, class_ids = postprocess_detections(predictions)
        
        assert len(boxes) == 0
        assert len(scores) == 0
        assert len(class_ids) == 0
    
    def test_below_confidence_threshold(self):
        predictions = np.array([
            [0, 0, 10, 10, 0.1, 0.1]
        ], dtype=np.float32)
        
        boxes, scores, class_ids = postprocess_detections(
            predictions, conf_threshold=0.5
        )
        
        assert len(boxes) == 0
    
    def test_above_confidence_threshold(self):
        predictions = np.array([
            [0, 0, 10, 10, 0.9, 0.1]
        ], dtype=np.float32)
        
        boxes, scores, class_ids = postprocess_detections(
            predictions, conf_threshold=0.5
        )
        
        assert len(boxes) == 1
    
    def test_max_detections_limit(self):
        predictions = np.array([
            [i * 20, i * 20, i * 20 + 10, i * 20 + 10, 0.9, 0.1]
            for i in range(20)
        ], dtype=np.float32)
        
        boxes, scores, class_ids = postprocess_detections(
            predictions, conf_threshold=0.5, max_detections=10
        )
        
        assert len(boxes) <= 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
