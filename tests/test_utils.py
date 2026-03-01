"""
Tests for nanotorch.utils module.
"""

import numpy as np
import tempfile
import os
import nanotorch as nt
from nanotorch.utils import (
    manual_seed,
    xavier_uniform_,
    xavier_normal_,
    kaiming_uniform_,
    kaiming_normal_,
    zeros_,
    ones_,
    uniform_,
    normal_,
    flatten,
    cat,
    stack,
    num_parameters,
    count_parameters,
    save_state_dict,
    load_state_dict,
    save,
    load,
    benchmark_operation,
)
from nanotorch.nn import Linear
from nanotorch.tensor import Tensor


def test_manual_seed():
    """Test manual_seed sets random seed for reproducibility."""
    manual_seed(42)
    rand1 = np.random.randn(5)
    
    np.random.seed(None)
    manual_seed(42)
    rand2 = np.random.randn(5)
    
    assert np.allclose(rand1, rand2), "Random numbers should be reproducible with same seed"
    
    print("✓ test_manual_seed passed")


def test_xavier_uniform_():
    """Test Xavier uniform initialization."""
    # Create 2D tensor (weight matrix)
    tensor = Tensor.zeros((10, 20), requires_grad=True)
    initialized = xavier_uniform_(tensor, gain=1.0)
    
    assert initialized is tensor
    
    fan_in = 20
    fan_out = 10
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    
    assert np.max(tensor.data) <= limit + 1e-5
    assert np.min(tensor.data) >= -limit - 1e-5
    assert tensor.data.mean() < 0.1
    
    conv_weight = Tensor.zeros((32, 16, 3, 3), requires_grad=True)
    xavier_uniform_(conv_weight, gain=2.0)
    
    fan_in = 16 * 3 * 3
    fan_out = 32 * 3 * 3
    limit = 2.0 * np.sqrt(6.0 / (fan_in + fan_out))
    
    assert np.max(conv_weight.data) <= limit + 1e-5
    assert np.min(conv_weight.data) >= -limit - 1e-5
    
    try:
        xavier_uniform_(Tensor.zeros((10,)), gain=1.0)
        assert False, "Should have raised ValueError for 1D tensor"
    except ValueError:
        pass
    
    print("✓ test_xavier_uniform_ passed")


def test_xavier_normal_():
    """Test Xavier normal initialization."""
    # Create 2D tensor
    tensor = Tensor.zeros((10, 20), requires_grad=True)
    initialized = xavier_normal_(tensor, gain=1.0)
    
    assert initialized is tensor
    
    fan_in = 20
    fan_out = 10
    std = np.sqrt(2.0 / (fan_in + fan_out))
    
    assert abs(tensor.data.mean()) < 0.1
    sample_std = tensor.data.std()
    assert 0.5 * std < sample_std < 1.5 * std
    
    tensor3d = Tensor.zeros((5, 10, 15), requires_grad=True)
    xavier_normal_(tensor3d, gain=0.5)
    
    fan_in = 10 * 15
    fan_out = 5 * 15
    std = 0.5 * np.sqrt(2.0 / (fan_in + fan_out))
    sample_std = tensor3d.data.std()
    assert 0.5 * std < sample_std < 1.5 * std
    
    print("✓ test_xavier_normal_ passed")


def test_kaiming_uniform_():
    """Test Kaiming uniform initialization."""
    tensor = Tensor.zeros((10, 20), requires_grad=True)
    initialized = kaiming_uniform_(tensor, a=0, mode="fan_in", nonlinearity="relu")
    
    assert initialized is tensor
    
    fan = 20
    bound = np.sqrt(2.0) * np.sqrt(3.0 / fan)
    
    assert np.max(tensor.data) <= bound + 1e-5
    assert np.min(tensor.data) >= -bound - 1e-5
    
    tensor2 = Tensor.zeros((10, 20), requires_grad=True)
    kaiming_uniform_(tensor2, a=0.1, mode="fan_out", nonlinearity="leaky_relu")
    
    fan = 10
    gain = np.sqrt(2.0 / (1 + 0.1**2))
    bound = gain * np.sqrt(3.0 / fan)
    
    assert np.max(tensor2.data) <= bound + 1e-5
    assert np.min(tensor2.data) >= -bound - 1e-5
    
    try:
        kaiming_uniform_(Tensor.zeros((10,)), a=0, mode="fan_in", nonlinearity="relu")
        assert False, "Should have raised ValueError for 1D tensor"
    except ValueError:
        pass
    
    try:
        kaiming_uniform_(tensor, a=0, mode="invalid", nonlinearity="relu")
        assert False, "Should have raised ValueError for invalid mode"
    except ValueError:
        pass
    
    try:
        kaiming_uniform_(tensor, a=0, mode="fan_in", nonlinearity="invalid")
        assert False, "Should have raised ValueError for invalid nonlinearity"
    except ValueError:
        pass
    
    print("✓ test_kaiming_uniform_ passed")


def test_kaiming_normal_():
    """Test Kaiming normal initialization."""
    # Test with fan_in mode
    tensor = Tensor.zeros((10, 20), requires_grad=True)
    initialized = kaiming_normal_(tensor, a=0, mode="fan_in", nonlinearity="relu")
    
    assert initialized is tensor
    
    fan = 20  # fan_in
    std = np.sqrt(2.0) / np.sqrt(fan)
    
    assert abs(tensor.data.mean()) < 0.1
    sample_std = tensor.data.std()
    assert 0.5 * std < sample_std < 1.5 * std
    
    # Test with leaky_relu
    tensor2 = Tensor.zeros((10, 20), requires_grad=True)
    kaiming_normal_(tensor2, a=0.2, mode="fan_out", nonlinearity="leaky_relu")
    
    fan = 10  # fan_out
    gain = np.sqrt(2.0 / (1 + 0.2**2))
    std = gain / np.sqrt(fan)
    
    sample_std = tensor2.data.std()
    assert 0.5 * std < sample_std < 1.5 * std
    
    print("✓ test_kaiming_normal_ passed")


def test_zeros_ones():
    """Test zeros_ and ones_ initialization."""
    # Test zeros_
    tensor = Tensor.randn((3, 4, 5), requires_grad=True)
    zeros_(tensor)
    
    assert np.allclose(tensor.data, 0.0)
    assert tensor.shape == (3, 4, 5)
    
    # Test ones_
    tensor2 = Tensor.randn((2, 2), requires_grad=True)
    ones_(tensor2)
    
    assert np.allclose(tensor2.data, 1.0)
    assert tensor2.shape == (2, 2)
    
    print("✓ test_zeros_ones passed")


def test_uniform_normal():
    """Test uniform_ and normal_ initialization."""
    # Test uniform_
    tensor = Tensor.zeros((100,), requires_grad=True)
    uniform_(tensor, low=-2.0, high=3.0)
    
    assert np.min(tensor.data) >= -2.0 - 1e-5
    assert np.max(tensor.data) <= 3.0 + 1e-5
    # Mean should be approximately (low + high) / 2 = 0.5
    assert 0.3 < tensor.data.mean() < 0.7
    
    # Test normal_
    tensor2 = Tensor.zeros((1000,), requires_grad=True)
    normal_(tensor2, mean=5.0, std=2.0)
    
    # Check statistics roughly
    assert 4.5 < tensor2.data.mean() < 5.5
    assert 1.5 < tensor2.data.std() < 2.5
    
    print("✓ test_uniform_normal passed")


def test_flatten():
    """Test flatten utility function."""
    # Create 3D tensor
    tensor = Tensor.randn((2, 3, 4), requires_grad=True)
    
    # Flatten all dimensions
    flattened = flatten(tensor)
    assert flattened.shape == (24,)
    assert np.allclose(flattened.data.flatten(), tensor.data.flatten())
    
    # Flatten middle dimensions
    flattened2 = flatten(tensor, start_dim=1, end_dim=2)
    assert flattened2.shape == (2, 12)
    
    # Flatten first two dimensions
    flattened3 = flatten(tensor, start_dim=0, end_dim=1)
    assert flattened3.shape == (6, 4)
    
    # Test with negative indices
    flattened4 = flatten(tensor, start_dim=-2, end_dim=-1)
    assert flattened4.shape == (2, 12)
    
    # Test error: start_dim > end_dim
    try:
        flatten(tensor, start_dim=2, end_dim=1)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    # Test with requires_grad
    tensor.requires_grad = True
    flattened_grad = flatten(tensor)
    assert flattened_grad.requires_grad == True
    
    print("✓ test_flatten passed")


def test_cat():
    """Test cat utility function."""
    t1 = Tensor.ones((2, 3), requires_grad=True)
    t2 = Tensor.zeros((2, 3), requires_grad=False)
    t3 = Tensor(np.full((2, 3), 2.0), requires_grad=True)
    
    # Concatenate along dimension 0
    result = cat([t1, t2, t3], dim=0)
    assert result.shape == (6, 3)
    assert result.requires_grad == True  # At least one requires_grad=True
    
    # Check values
    expected = np.vstack([t1.data, t2.data, t3.data])
    assert np.allclose(result.data, expected)
    
    # Concatenate along dimension 1
    result2 = cat([t1, t2], dim=1)
    assert result2.shape == (2, 6)
    
    # Test with empty list
    try:
        cat([], dim=0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    # Test dimension mismatch
    t4 = Tensor.ones((3, 2))  # Different shape
    try:
        cat([t1, t4], dim=0)
        # This might work if shapes are compatible for concatenation
        # Actually, along dim 0, shapes (2,3) and (3,2) are incompatible
        # NumPy will raise ValueError
    except ValueError:
        pass  # Expected
    
    print("✓ test_cat passed")


def test_stack():
    """Test stack utility function."""
    t1 = Tensor.ones((2, 3), requires_grad=True)
    t2 = Tensor.zeros((2, 3), requires_grad=False)
    t3 = Tensor(np.full((2, 3), 2.0), requires_grad=True)
    
    # Stack along new dimension 0
    result = stack([t1, t2, t3], dim=0)
    assert result.shape == (3, 2, 3)
    assert result.requires_grad == True
    
    # Check values
    expected = np.stack([t1.data, t2.data, t3.data], axis=0)
    assert np.allclose(result.data, expected)
    
    # Stack along last dimension
    result2 = stack([t1, t2], dim=-1)
    assert result2.shape == (2, 3, 2)
    
    # Test with empty list
    try:
        stack([], dim=0)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    
    # Test shape mismatch
    t4 = Tensor.ones((3, 2))  # Different shape
    try:
        stack([t1, t4], dim=0)
        assert False, "Should have raised ValueError for shape mismatch"
    except ValueError:
        pass
    
    print("✓ test_stack passed")


def test_num_parameters():
    """Test num_parameters utility function."""
    # Create a simple model
    linear1 = Linear(10, 20)
    linear2 = Linear(20, 5)
    
    # Count parameters
    total_params = num_parameters(linear1)
    # Linear: weight (10×20) + bias (1×20) = 200 + 20 = 220
    assert total_params == 10 * 20 + 20
    
    # Multiple modules (use Sequential or custom container)
    class SimpleModel:
        def __init__(self):
            self.linear1 = Linear(10, 20)
            self.linear2 = Linear(20, 5)
            
        def parameters(self):
            return list(self.linear1.parameters()) + list(self.linear2.parameters())
    
    model = SimpleModel()
    total = num_parameters(model)
    expected = (10 * 20 + 20) + (20 * 5 + 5)
    assert total == expected
    
    print("✓ test_num_parameters passed")


def test_count_parameters():
    """Test count_parameters utility function."""
    # Create model with both trainable and non-trainable parameters
    linear = Linear(5, 3)
    assert linear.bias is not None
    linear.bias.requires_grad = False
    
    # Count all parameters
    total, trainable = count_parameters(linear, trainable_only=False)
    expected_total = 5 * 3 + 3  # weight + bias
    assert total == expected_total
    assert trainable == 5 * 3  # Only weight is trainable
    
    # Count only trainable
    total_trainable_only, trainable_only = count_parameters(linear, trainable_only=True)
    assert total_trainable_only == 5 * 3
    assert trainable_only == 5 * 3
    
    print("✓ test_count_parameters passed")


def test_save_load_state_dict():
    """Test save_state_dict and load_state_dict functions."""
    import tempfile
    import os
    
    # Create a state dictionary
    state_dict = {
        "weight": np.random.randn(10, 20).astype(np.float32),
        "bias": np.random.randn(20).astype(np.float32),
        "metadata": np.array([1, 2, 3], dtype=np.float32),
    }
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        temp_path = f.name
    
    try:
        save_state_dict(state_dict, temp_path)
        
        # Check file exists
        assert os.path.exists(temp_path)
        assert os.path.getsize(temp_path) > 0
        
        # Load and compare
        loaded_dict = load_state_dict(temp_path)
        
        # Check keys match
        assert set(loaded_dict.keys()) == set(state_dict.keys())
        
        # Check values match
        for key in state_dict:
            assert np.allclose(loaded_dict[key], state_dict[key])
            assert loaded_dict[key].dtype == np.float32
            
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    print("✓ test_save_load_state_dict passed")


def test_save_load_module():
    """Test save and load functions for modules."""
    import tempfile
    import os
    
    # Create a simple model
    linear = Linear(8, 4)
    assert linear.bias is not None
    bias = linear.bias
    
    # Store original parameters
    original_weight = linear.weight.data.copy()
    original_bias = bias.data.copy()
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        temp_path = f.name
    
    try:
        save(linear, temp_path)
        assert os.path.exists(temp_path)
        
        # Modify parameters
        linear.weight.data = np.random.randn(*linear.weight.shape).astype(np.float32)
        bias.data = np.random.randn(*bias.shape).astype(np.float32)
        
        # Load and verify parameters restored
        load(linear, temp_path, strict=True)
        
        assert np.allclose(linear.weight.data, original_weight)
        assert np.allclose(bias.data, original_bias)
        
        # Test strict=False with extra keys (not implemented, but test API)
        # Create state dict with extra keys
        extra_dict = linear.state_dict()
        extra_dict["extra_param"] = np.array([1.0], dtype=np.float32)
        
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f2:
            temp_path2 = f2.name
        save_state_dict(extra_dict, temp_path2)
        
        # Should work with strict=False
        load(linear, temp_path2, strict=False)
        
        # Clean up second temp file
        os.unlink(temp_path2)
        
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    print("✓ test_save_load_module passed")


def test_benchmark_operation():
    """Test benchmark_operation utility."""
    # Define a simple operation
    def add_tensors(a, b):
        return a + b
    
    # Create tensors
    a = Tensor.randn((100, 100))
    b = Tensor.randn((100, 100))
    
    # Benchmark
    avg_time = benchmark_operation(add_tensors, a, b, iterations=5, warmup=1)
    
    # Should return a positive number
    assert avg_time > 0
    assert isinstance(avg_time, float)
    
    print(f"✓ test_benchmark_operation passed (avg time: {avg_time:.3f} ms)")


def run_all_utils_tests():
    """Run all utils tests."""
    test_manual_seed()
    test_xavier_uniform_()
    test_xavier_normal_()
    test_kaiming_uniform_()
    test_kaiming_normal_()
    test_zeros_ones()
    test_uniform_normal()
    test_flatten()
    test_cat()
    test_stack()
    test_num_parameters()
    test_count_parameters()
    test_save_load_state_dict()
    test_save_load_module()
    test_benchmark_operation()
    
    print("\n✅ All utils tests passed!")


if __name__ == "__main__":
    run_all_utils_tests()