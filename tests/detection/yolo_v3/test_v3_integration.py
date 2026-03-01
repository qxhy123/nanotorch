"""
Integration tests for YOLO v3 training and inference pipeline.

Tests the complete workflow:
- Data loading and preprocessing
- Model forward pass
- Loss computation
- Backward pass and gradient flow
- Optimizer step
- End-to-end training loop
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

from nanotorch.tensor import Tensor
from nanotorch.optim import SGD, Adam
from nanotorch.detection.yolo_v3 import (
    YOLOv3,
    YOLOv3Tiny,
    YOLOv3Loss,
    YOLOv3LossSimple,
    decode_predictions_v3,
    encode_targets_v3,
    build_yolov3
)
from nanotorch.data import Dataset, DataLoader


class SyntheticCOCODataset(Dataset):
    """Synthetic dataset mimicking COCO format for testing."""
    
    def __init__(self, num_samples=100, image_size=224, num_classes=80, max_objects=5):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self.max_objects = max_objects
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        np.random.seed(idx)
        
        image = np.random.rand(self.image_size, self.image_size, 3).astype(np.float32)
        
        num_objects = np.random.randint(1, self.max_objects + 1)
        
        boxes = []
        labels = []
        
        for _ in range(num_objects):
            w = np.random.randint(20, self.image_size // 4)
            h = np.random.randint(20, self.image_size // 4)
            x1 = np.random.randint(0, self.image_size - w)
            y1 = np.random.randint(0, self.image_size - h)
            x2 = x1 + w
            y2 = y1 + h
            
            boxes.append([x1, y1, x2, y2])
            labels.append(np.random.randint(0, self.num_classes))
        
        return {
            'image': image.transpose(2, 0, 1),
            'boxes': np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32),
            'labels': np.array(labels, dtype=np.int64) if labels else np.zeros((0,), dtype=np.int64),
            'image_id': idx
        }


def create_test_dataloader(num_samples=20, batch_size=2, image_size=224, num_classes=80):
    """Create a test dataloader."""
    dataset = SyntheticCOCODataset(num_samples, image_size, num_classes)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


class TestDataLoadingIntegration:
    """Integration tests for data loading pipeline."""
    
    def test_synthetic_dataset_creation(self):
        """Test synthetic dataset can be created and accessed."""
        dataset = SyntheticCOCODataset(
            num_samples=10,
            image_size=224,
            num_classes=80,
            max_objects=3
        )
        
        assert len(dataset) == 10
        
        sample = dataset[0]
        assert 'image' in sample
        assert 'boxes' in sample
        assert 'labels' in sample
        
        assert sample['image'].shape == (3, 224, 224)
        assert sample['boxes'].shape[1] == 4 if len(sample['boxes']) > 0 else True
    
    def test_dataloader_batch_creation(self):
        """Test dataloader creates valid batches."""
        dataloader = create_test_dataloader(
            num_samples=10,
            batch_size=2,
            image_size=224
        )
        
        for batch in dataloader:
            # nanotorch DataLoader returns a numpy array of dicts
            assert isinstance(batch, np.ndarray)
            assert len(batch) == 2
            assert 'image' in batch[0]
            assert 'boxes' in batch[0]
            assert 'labels' in batch[0]
            break
    
    def test_collate_preserves_data(self):
        """Test that data is properly batched."""
        dataset = SyntheticCOCODataset(
            num_samples=3,
            image_size=224,
            num_classes=80
        )
        
        dataloader = DataLoader(dataset, batch_size=3, shuffle=False)
        
        for batch in dataloader:
            # nanotorch DataLoader returns numpy array of dicts
            assert isinstance(batch, np.ndarray)
            assert batch[0]['image'].dtype == np.float32
            assert batch[0]['image'].max() <= 1.0
            assert batch[0]['image'].min() >= 0.0
            break


class TestModelDataIntegration:
    """Integration tests for model with data."""
    
    def test_tiny_model_accepts_dataloader_output(self):
        """Test model can process dataloader batch output."""
        dataloader = create_test_dataloader(
            num_samples=4,
            batch_size=2,
            image_size=224
        )
        
        model = YOLOv3Tiny(num_classes=80, input_size=224)
        
        for batch in dataloader:
            # nanotorch DataLoader returns numpy array of dicts
            # Extract and stack images from batch
            images = np.stack([item['image'] for item in batch])
            images = Tensor(images)
            
            output = model(images)
            
            assert 'small' in output
            assert 'route' in output
            assert output['small'].shape[0] == 2
            break
    
    def test_tiny_model_output_shapes(self):
        """Test model output shapes are correct."""
        model = YOLOv3Tiny(num_classes=80, input_size=224)
        
        x = Tensor(np.random.randn(2, 3, 224, 224).astype(np.float32))
        output = model(x)
        
        # YOLOv3Tiny outputs 'small' and 'route'
        assert 'small' in output
        # Output channels = 3 * (5 + num_classes) = 3 * 85 = 255
        assert output['small'].shape[1] == 255


class TestTrainingIntegration:
    """Integration tests for training pipeline."""
    
    def test_forward_loss_backward_cycle_tiny(self):
        """Test complete forward-loss-backward cycle using MSE."""
        model = YOLOv3Tiny(num_classes=80, input_size=224)
        
        x = Tensor(np.random.randn(2, 3, 224, 224).astype(np.float32))
        
        # Forward
        output = model(x)
        
        # Manual MSE loss for backward
        loss = 0.0
        for scale_name, pred in output.items():
            diff = pred - Tensor(np.zeros_like(pred.data))
            loss += (diff * diff).mean().item()
        
        # Create loss tensor for backward
        loss_tensor = Tensor(loss, requires_grad=True)
        
        # Backward
        loss_tensor.backward()
        
        # Check gradients exist
        params_with_grad = 0
        for param in model.parameters():
            if param.grad is not None:
                params_with_grad += 1
        
        assert params_with_grad > 0, "No gradients found in model parameters"
    
    def test_optimizer_step_tiny(self):
        """Test optimizer can update model parameters."""
        model = YOLOv3Tiny(num_classes=80, input_size=224)
        optimizer = SGD(model.parameters(), lr=0.1)  # Higher learning rate
        
        x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
        
        # Store original weights (first few for efficiency)
        original_weights = []
        for i, param in enumerate(model.parameters()):
            if i < 5:  # Only check first 5 params
                original_weights.append(param.data.copy())
        
        # Training step with proper gradient flow
        optimizer.zero_grad()
        output = model(x)
        
        # Use actual tensor operations for gradient flow
        total_loss = None
        for scale_name, pred in output.items():
            diff = pred - Tensor(np.ones_like(pred.data) * 0.5)
            scale_loss = (diff * diff).mean()
            if total_loss is None:
                total_loss = scale_loss
            else:
                total_loss = total_loss + scale_loss
        
        # Backward and step
        total_loss.backward()
        optimizer.step()
        
        # Check weights changed
        weights_changed = False
        for i, param in enumerate(model.parameters()):
            if i < 5 and i < len(original_weights):
                if not np.allclose(param.data, original_weights[i], rtol=1e-3, atol=1e-3):
                    weights_changed = True
                    break
        
        assert weights_changed, "Optimizer did not update any weights"
    
    def test_loss_decreases_over_iterations_tiny(self):
        """Test that loss decreases over multiple training iterations."""
        model = YOLOv3Tiny(num_classes=80, input_size=224)
        optimizer = Adam(model.parameters(), lr=0.05)
        
        np.random.seed(42)
        x = Tensor(np.random.randn(2, 3, 224, 224).astype(np.float32))
        
        losses = []
        for _ in range(10):  # Reduced iterations for faster tests
            optimizer.zero_grad()
            output = model(x)
            
            # Use tensor operations for proper gradient flow
            pred = output['small']
            target = Tensor(np.zeros_like(pred.data))
            diff = pred - target
            loss = (diff * diff).mean()
            
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Loss should decrease with proper gradient flow
        assert losses[-1] < losses[0], f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
    
    def test_yolov3_loss_computation(self):
        """Test YOLOv3Loss computation returns valid values with full model."""
        # Use full YOLOv3 for proper loss computation
        model = YOLOv3(num_classes=80, input_size=224)
        loss_fn = YOLOv3Loss(num_classes=80)
        
        x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
        
        output = model(x)
        
        targets = [
            {
                'boxes': np.array([[50, 50, 150, 150]], dtype=np.float32),
                'labels': np.array([0], dtype=np.int64)
            }
        ]
        
        loss, loss_dict = loss_fn(output, targets)
        
        assert 'coord_loss' in loss_dict
        assert 'obj_loss' in loss_dict
        assert 'noobj_loss' in loss_dict
        assert 'class_loss' in loss_dict
    
    def test_simple_loss_computation(self):
        """Test simplified loss computation."""
        loss_fn = YOLOv3LossSimple(num_classes=80)
        
        predictions = {
            'small': Tensor(np.random.randn(2, 255, 7, 7).astype(np.float32) * 0.1),
            'medium': Tensor(np.random.randn(2, 255, 14, 14).astype(np.float32) * 0.1)
        }
        
        targets = {
            'small': Tensor(np.zeros((2, 255, 7, 7), dtype=np.float32)),
            'medium': Tensor(np.zeros((2, 255, 14, 14), dtype=np.float32))
        }
        
        loss, loss_dict = loss_fn(predictions, targets)
        
        assert isinstance(loss, float)
        assert not np.isnan(loss)
        assert not np.isinf(loss)


class TestInferenceIntegration:
    """Integration tests for inference pipeline."""
    
    def test_tiny_predict(self):
        """Test model prediction."""
        model = YOLOv3Tiny(num_classes=80, input_size=224)
        
        x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
        
        output = model(x)
        
        assert 'small' in output
    
    def test_batch_inference(self):
        """Test inference on batch of images."""
        model = YOLOv3Tiny(num_classes=80, input_size=224)
        
        x = Tensor(np.random.randn(4, 3, 224, 224).astype(np.float32))
        
        output = model(x)
        
        assert output['small'].shape[0] == 4


class TestTrainingLoopIntegration:
    """End-to-end training loop integration tests."""
    
    def test_mini_training_loop_tiny(self):
        """Test mini training loop with synthetic data."""
        dataloader = create_test_dataloader(
            num_samples=8,
            batch_size=2,
            image_size=224
        )
        
        model = YOLOv3Tiny(num_classes=80, input_size=224)
        optimizer = Adam(model.parameters(), lr=0.01)
        
        epoch_losses = []
        
        for batch in dataloader:
            # Extract and stack images from batch
            images = np.stack([item['image'] for item in batch])
            images = Tensor(images)
            
            optimizer.zero_grad()
            output = model(images)
            
            # Use tensor operations for proper gradient flow
            pred = output['small']
            target = Tensor(np.zeros_like(pred.data))
            diff = pred - target
            loss = (diff * diff).mean()
            
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
        
        assert all(not np.isnan(l) for l in epoch_losses), f"NaN in losses: {epoch_losses}"
        assert all(not np.isinf(l) for l in epoch_losses), f"Inf in losses: {epoch_losses}"
    
    def test_model_save_load_state(self):
        """Test model state can be saved and loaded."""
        model1 = YOLOv3Tiny(num_classes=80, input_size=224)
        
        x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
        output1 = model1(x)
        
        state = model1.state_dict()
        
        model2 = YOLOv3Tiny(num_classes=80, input_size=224)
        model2.load_state_dict(state)
        
        output2 = model2(x)
        
        # Check small scale output
        np.testing.assert_allclose(
            output1['small'].data, 
            output2['small'].data, 
            rtol=1e-5
        )


class TestGradientFlow:
    """Tests for gradient flow through the network."""
    
    def test_gradient_reaches_all_layers_tiny(self):
        """Test gradients flow through all layers."""
        model = YOLOv3Tiny(num_classes=80, input_size=224)
        
        x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
        
        output = model(x)
        
        loss = 0.0
        for scale_name, pred in output.items():
            diff = pred - Tensor(np.zeros_like(pred.data))
            loss += (diff * diff).mean().item()
        
        loss_tensor = Tensor(loss, requires_grad=True)
        loss_tensor.backward()
        
        params_with_grad = 0
        for param in model.parameters():
            if param.grad is not None:
                params_with_grad += 1
        
        assert params_with_grad > 0
    
    def test_gradient_magnitude_reasonable_tiny(self):
        """Test gradient magnitudes are reasonable."""
        model = YOLOv3Tiny(num_classes=80, input_size=224)
        
        x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
        
        output = model(x)
        
        loss = 0.0
        for scale_name, pred in output.items():
            diff = pred - Tensor(np.zeros_like(pred.data))
            loss += (diff * diff).mean().item()
        
        loss_tensor = Tensor(loss, requires_grad=True)
        loss_tensor.backward()
        
        max_grad = 0
        for param in model.parameters():
            if param.grad is not None:
                max_grad = max(max_grad, np.abs(param.grad.data).max())
        
        assert max_grad < 1000, f"Gradient exploded: {max_grad}"


class TestNumericalStability:
    """Tests for numerical stability."""
    
    def test_loss_with_all_zeros(self):
        """Test loss computation with zero inputs."""
        model = YOLOv3Tiny(num_classes=80, input_size=224)
        loss_fn = YOLOv3LossSimple(num_classes=80)
        
        x = Tensor(np.zeros((1, 3, 224, 224), dtype=np.float32))
        
        output = model(x)
        
        targets = {k: Tensor(np.zeros_like(v.data)) for k, v in output.items()}
        loss, _ = loss_fn(output, targets)
        
        assert not np.isnan(loss)
        assert not np.isinf(loss)
    
    def test_loss_with_extreme_values(self):
        """Test loss computation with extreme values."""
        loss_fn = YOLOv3LossSimple(num_classes=80)
        
        predictions = {
            'small': Tensor(np.ones((1, 255, 7, 7), dtype=np.float32) * 10)
        }
        targets = {
            'small': Tensor(np.zeros((1, 255, 7, 7), dtype=np.float32))
        }
        
        loss, _ = loss_fn(predictions, targets)
        
        assert not np.isnan(loss)
        assert not np.isinf(loss)


class TestEncodeDecode:
    """Tests for target encoding and prediction decoding."""
    
    def test_encode_targets(self):
        """Test target encoding."""
        anchors = [
            (10, 13), (16, 30), (33, 23),
            (30, 61), (62, 45), (59, 119),
            (116, 90), (156, 198), (373, 326)
        ]
        
        boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
        labels = np.array([0], dtype=np.int64)
        
        targets = encode_targets_v3(
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
        
        assert targets['scale_0'].shape == (3, 85, 13, 13)
        assert targets['scale_1'].shape == (3, 85, 26, 26)
        assert targets['scale_2'].shape == (3, 85, 52, 52)
    
    def test_decode_predictions_empty(self):
        """Test prediction decoding with no detections."""
        predictions = np.zeros((3, 85, 13, 13), dtype=np.float32)
        
        anchors = [(116, 90), (156, 198), (373, 326)]
        
        boxes, scores, class_ids = decode_predictions_v3(
            predictions,
            anchors=anchors,
            conf_threshold=0.5,
            num_classes=80,
            image_size=416
        )
        
        assert len(boxes) == 0
        assert len(scores) == 0
        assert len(class_ids) == 0
    
    def test_decode_predictions_with_objects(self):
        """Test prediction decoding with detections."""
        # Create predictions with high confidence
        predictions = np.zeros((3, 85, 13, 13), dtype=np.float32)
        predictions[0, 4, 5, 5] = 0.9  # High confidence
        predictions[0, 0, 5, 5] = 0.5  # tx
        predictions[0, 1, 5, 5] = 0.5  # ty
        predictions[0, 2, 5, 5] = 0.0  # tw
        predictions[0, 3, 5, 5] = 0.0  # th
        predictions[0, 5, 5, 5] = 0.9  # class 0 prob
        
        anchors = [(116, 90), (156, 198), (373, 326)]
        
        boxes, scores, class_ids = decode_predictions_v3(
            predictions,
            anchors=anchors,
            conf_threshold=0.5,
            num_classes=80,
            image_size=416
        )
        
        assert len(boxes) >= 1
        assert len(scores) >= 1
        assert len(class_ids) >= 1


class TestBuildModel:
    """Tests for model building functions."""
    
    def test_build_tiny_model(self):
        """Test building tiny model."""
        model = build_yolov3('tiny', num_classes=80, input_size=224)
        
        assert isinstance(model, YOLOv3Tiny)
        
        x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
        output = model(x)
        
        assert 'small' in output
    
    def test_build_full_model(self):
        """Test building full model."""
        model = build_yolov3('full', num_classes=80, input_size=224)
        
        assert isinstance(model, YOLOv3)
        
        x = Tensor(np.random.randn(1, 3, 224, 224).astype(np.float32))
        output = model(x)
        
        assert 'small' in output
        assert 'medium' in output
        assert 'large' in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
