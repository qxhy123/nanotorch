"""
Tests for model serialization (state_dict, load_state_dict, save, load).
"""

import numpy as np
import nanotorch as nt
from nanotorch.nn import Module, Linear, ReLU, BatchNorm2d
from nanotorch.utils import save_state_dict, load_state_dict, save, load
import tempfile
import os


class NestedModule(Module):
    """A simple nested module for testing."""

    def __init__(self):
        super().__init__()
        self.linear1 = Linear(10, 20)
        self.relu = ReLU()
        self.linear2 = Linear(20, 5)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


def test_state_dict_basic():
    """Test state_dict returns correct parameters."""
    model = Linear(10, 5)
    state_dict = model.state_dict()

    # Check keys
    assert set(state_dict.keys()) == {"weight", "bias"}

    # Check shapes
    assert state_dict["weight"].shape == (5, 10)
    assert state_dict["bias"].shape == (5,)

    # Check values match parameter data
    weight_data = model.weight.data
    bias_data = model.bias.data
    assert np.allclose(state_dict["weight"], weight_data)
    assert np.allclose(state_dict["bias"], bias_data)

    print("✓ test_state_dict_basic passed")


def test_load_state_dict():
    """Test load_state_dict correctly updates parameters."""
    model = Linear(10, 5)
    original_weight = model.weight.data.copy()
    original_bias = model.bias.data.copy()

    # Create new state dict with modified values
    new_state_dict = {"weight": original_weight + 1.0, "bias": original_bias - 0.5}

    # Load into model
    model.load_state_dict(new_state_dict)

    # Check values updated
    assert np.allclose(model.weight.data, original_weight + 1.0)
    assert np.allclose(model.bias.data, original_bias - 0.5)

    print("✓ test_load_state_dict passed")


def test_state_dict_with_buffers():
    """Test state_dict includes buffers (e.g., BatchNorm running stats)."""
    model = BatchNorm2d(3)
    model.train()  # Ensure buffers exist
    _ = model(nt.Tensor(np.random.randn(2, 3, 5, 5)))  # Forward to update buffers

    state_dict = model.state_dict()

    # Check for buffer keys
    expected_keys = {
        "gamma",
        "beta",
        "running_mean",
        "running_var",
        "num_batches_tracked",
    }
    assert expected_keys.issubset(set(state_dict.keys()))

    # Check shapes (BatchNorm stores params as (C,) shape per PyTorch convention)
    assert state_dict["gamma"].shape == (3,)
    assert state_dict["beta"].shape == (3,)
    assert state_dict["running_mean"].shape == (3,)
    assert state_dict["running_var"].shape == (3,)

    print("✓ test_state_dict_with_buffers passed")


def test_load_state_dict_with_buffers():
    """Test load_state_dict updates buffers."""
    model = BatchNorm2d(3)

    # Create new state dict (BatchNorm params have shape (C,) per PyTorch convention)
    new_state_dict = {
        "gamma": np.ones((3,), dtype=np.float32),
        "beta": np.zeros((3,), dtype=np.float32),
        "running_mean": np.full((3,), 2.0, dtype=np.float32),
        "running_var": np.full((3,), 1.5, dtype=np.float32),
        "num_batches_tracked": np.array(10, dtype=np.int64),
    }

    model.load_state_dict(new_state_dict)

    # Check values
    assert np.allclose(model.gamma.data, 1.0)
    assert np.allclose(model.beta.data, 0.0)
    assert np.allclose(model.running_mean.data, 2.0)
    assert np.allclose(model.running_var.data, 1.5)

    print("✓ test_load_state_dict_with_buffers passed")


def test_state_dict_nested_modules():
    """Test state_dict with nested module structure."""
    model = NestedModule()

    state_dict = model.state_dict()

    # Check keys include nested prefixes
    expected_keys = {"linear1.weight", "linear1.bias", "linear2.weight", "linear2.bias"}
    assert expected_keys.issubset(set(state_dict.keys()))

    # Check shapes
    assert state_dict["linear1.weight"].shape == (20, 10)
    assert state_dict["linear1.bias"].shape == (20,)
    assert state_dict["linear2.weight"].shape == (5, 20)
    assert state_dict["linear2.bias"].shape == (5,)

    print("✓ test_state_dict_nested_modules passed")


def test_load_state_dict_nested_modules():
    """Test load_state_dict with nested modules."""
    model = NestedModule()

    # Create state dict with modified values
    state_dict = {
        "linear1.weight": np.ones((20, 10), dtype=np.float32),
        "linear1.bias": np.zeros((20,), dtype=np.float32),
        "linear2.weight": np.ones((5, 20), dtype=np.float32) * 2.0,
        "linear2.bias": np.ones((5,), dtype=np.float32) * 0.5,
    }

    model.load_state_dict(state_dict)

    # Check values
    assert np.allclose(model.linear1.weight.data, 1.0)
    assert np.allclose(model.linear1.bias.data, 0.0)
    assert np.allclose(model.linear2.weight.data, 2.0)
    assert np.allclose(model.linear2.bias.data, 0.5)

    print("✓ test_load_state_dict_nested_modules passed")


def test_save_load_state_dict():
    """Test save_state_dict and load_state_dict roundtrip."""
    # Create a simple model
    model = Linear(8, 4)
    original_state_dict = model.state_dict()

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        filepath = f.name

    try:
        # Save
        save_state_dict(original_state_dict, filepath)

        # Load
        loaded_state_dict = load_state_dict(filepath)

        # Compare
        assert set(original_state_dict.keys()) == set(loaded_state_dict.keys())
        for key in original_state_dict.keys():
            assert np.allclose(original_state_dict[key], loaded_state_dict[key])

        print("✓ test_save_load_state_dict passed")
    finally:
        os.unlink(filepath)


def test_save_load_module():
    """Test save and load module roundtrip."""
    model1 = NestedModule()

    # Modify weights so they're not default
    model1.linear1.weight.data = np.random.randn(20, 10).astype(np.float32) * 0.1
    model1.linear1.bias.data = np.random.randn(20).astype(np.float32) * 0.1
    model1.linear2.weight.data = np.random.randn(5, 20).astype(np.float32) * 0.1
    model1.linear2.bias.data = np.random.randn(5).astype(np.float32) * 0.1

    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
        filepath = f.name

    try:
        # Save model1
        save(model1, filepath)

        # Create model2 with same architecture but different weights
        model2 = NestedModule()

        # Load model1's weights into model2
        load(model2, filepath)

        # Check all parameters match
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            assert np.allclose(p1.data, p2.data)

        print("✓ test_save_load_module passed")
    finally:
        os.unlink(filepath)


def test_load_state_dict_strict():
    """Test strict mode raises errors on missing/unexpected keys."""
    model = Linear(10, 5)

    # Missing key
    state_dict_missing = {"weight": np.ones((10, 5), dtype=np.float32)}
    try:
        model.load_state_dict(state_dict_missing, strict=True)
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "Missing keys" in str(e)

    # Unexpected key
    state_dict_unexpected = {
        "weight": np.ones((10, 5), dtype=np.float32),
        "bias": np.zeros((1, 5), dtype=np.float32),
        "extra": np.ones((5,), dtype=np.float32),
    }
    try:
        model.load_state_dict(state_dict_unexpected, strict=True)
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        assert "Unexpected keys" in str(e)

    print("✓ test_load_state_dict_strict passed")


def test_load_state_dict_non_strict():
    """Test non-strict mode handles missing/unexpected keys with warnings."""
    import io
    import sys

    model = Linear(10, 5)

    # Capture stdout for warnings
    old_stdout = sys.stdout
    sys.stdout = captured = io.StringIO()

    # Missing key
    state_dict_missing = {"weight": np.ones((10, 5), dtype=np.float32)}
    model.load_state_dict(state_dict_missing, strict=False)
    output = captured.getvalue()
    assert "Warning: Missing keys" in output

    # Reset capture
    captured.truncate(0)
    captured.seek(0)

    # Unexpected key
    state_dict_unexpected = {
        "weight": np.ones((10, 5), dtype=np.float32),
        "bias": np.zeros((1, 5), dtype=np.float32),
        "extra": np.ones((5,), dtype=np.float32),
    }
    model.load_state_dict(state_dict_unexpected, strict=False)
    output = captured.getvalue()
    assert "Warning: Unexpected keys" in output

    sys.stdout = old_stdout

    print("✓ test_load_state_dict_non_strict passed")


def run_all_tests():
    """Run all serialization tests."""
    test_state_dict_basic()
    test_load_state_dict()
    test_state_dict_with_buffers()
    test_load_state_dict_with_buffers()
    test_state_dict_nested_modules()
    test_load_state_dict_nested_modules()
    test_save_load_state_dict()
    test_save_load_module()
    test_load_state_dict_strict()
    test_load_state_dict_non_strict()
    print("\n✅ All serialization tests passed!")


if __name__ == "__main__":
    run_all_tests()
