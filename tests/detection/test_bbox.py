"""
Unit tests for bounding box utilities.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from nanotorch.detection.bbox import (
    xyxy_to_xywh,
    xywh_to_xyxy,
    normalize_boxes,
    denormalize_boxes,
    box_area,
    box_intersection,
    box_iou,
    clip_boxes,
    encode_boxes,
    decode_boxes,
    generate_anchors
)
from nanotorch.tensor import Tensor


class TestBboxConversions:
    
    def test_xyxy_to_xywh(self):
        boxes = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=np.float32)
        result = xyxy_to_xywh(boxes)
        
        expected = np.array([[5, 5, 10, 10], [10, 10, 10, 10]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_xywh_to_xyxy(self):
        boxes = np.array([[5, 5, 10, 10], [10, 10, 10, 10]], dtype=np.float32)
        result = xywh_to_xyxy(boxes)
        
        expected = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_roundtrip_conversion(self):
        original = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=np.float32)
        
        xywh = xyxy_to_xywh(original)
        recovered = xywh_to_xyxy(xywh)
        
        np.testing.assert_array_almost_equal(original, recovered)
    
    def test_with_tensor_input(self):
        boxes = Tensor(np.array([[0, 0, 10, 10]], dtype=np.float32))
        result = xyxy_to_xywh(boxes)
        
        expected = np.array([[5, 5, 10, 10]], dtype=np.float32)
        np.testing.assert_array_almost_equal(result, expected)


class TestBoxArea:
    
    def test_single_box(self):
        box = np.array([[0, 0, 10, 10]], dtype=np.float32)
        area = box_area(box)
        
        assert area[0] == 100.0
    
    def test_multiple_boxes(self):
        boxes = np.array([[0, 0, 10, 10], [0, 0, 5, 5]], dtype=np.float32)
        areas = box_area(boxes)
        
        np.testing.assert_array_almost_equal(areas, [100.0, 25.0])


class TestBoxIntersection:
    
    def test_no_intersection(self):
        boxes1 = np.array([[0, 0, 10, 10]], dtype=np.float32)
        boxes2 = np.array([[20, 20, 30, 30]], dtype=np.float32)
        
        inter = box_intersection(boxes1, boxes2)
        
        assert inter[0, 0] == 0.0
    
    def test_full_overlap(self):
        boxes1 = np.array([[0, 0, 10, 10]], dtype=np.float32)
        boxes2 = np.array([[0, 0, 10, 10]], dtype=np.float32)
        
        inter = box_intersection(boxes1, boxes2)
        
        assert inter[0, 0] == 100.0
    
    def test_partial_intersection(self):
        boxes1 = np.array([[0, 0, 10, 10]], dtype=np.float32)
        boxes2 = np.array([[5, 5, 15, 15]], dtype=np.float32)
        
        inter = box_intersection(boxes1, boxes2)
        
        assert inter[0, 0] == 25.0


class TestBoxIoU:
    
    def test_identical_boxes(self):
        boxes1 = np.array([[0, 0, 10, 10]], dtype=np.float32)
        boxes2 = np.array([[0, 0, 10, 10]], dtype=np.float32)
        
        iou = box_iou(boxes1, boxes2)
        
        assert iou[0, 0] == 1.0
    
    def test_no_overlap(self):
        boxes1 = np.array([[0, 0, 10, 10]], dtype=np.float32)
        boxes2 = np.array([[20, 20, 30, 30]], dtype=np.float32)
        
        iou = box_iou(boxes1, boxes2)
        
        assert iou[0, 0] == 0.0
    
    def test_half_overlap(self):
        boxes1 = np.array([[0, 0, 10, 10]], dtype=np.float32)
        boxes2 = np.array([[0, 0, 10, 5]], dtype=np.float32)
        
        iou = box_iou(boxes1, boxes2)
        
        expected_iou = 50.0 / 100.0
        assert abs(iou[0, 0] - expected_iou) < 1e-6


class TestClipBoxes:
    
    def test_clip_within_bounds(self):
        boxes = np.array([[5, 5, 15, 15]], dtype=np.float32)
        img_size = (20, 20)
        
        clipped = clip_boxes(boxes, img_size)
        
        np.testing.assert_array_almost_equal(clipped, boxes)
    
    def test_clip_exceeding_bounds(self):
        boxes = np.array([[-5, -5, 25, 25]], dtype=np.float32)
        img_size = (20, 20)
        
        clipped = clip_boxes(boxes, img_size)
        
        expected = np.array([[0, 0, 20, 20]], dtype=np.float32)
        np.testing.assert_array_almost_equal(clipped, expected)


class TestEncodeDecode:
    
    def test_encode_decode_roundtrip(self):
        boxes = np.array([[100, 100, 50, 50]], dtype=np.float32)
        anchors = np.array([[100, 100, 40, 40]], dtype=np.float32)
        
        encoded = encode_boxes(boxes, anchors)
        decoded = decode_boxes(encoded, anchors)
        
        np.testing.assert_array_almost_equal(boxes, decoded, decimal=5)


class TestGenerateAnchors:
    
    def test_anchor_generation(self):
        anchors = generate_anchors(sizes=(32,), ratios=(1.0,))
        
        assert anchors.shape == (1, 4)
        np.testing.assert_array_almost_equal(anchors[0, :2], [0, 0])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
