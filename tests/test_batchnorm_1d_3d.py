"""
Tests for BatchNorm1d and BatchNorm3d layers.
"""

import numpy as np
import nanotorch as nt
from nanotorch.nn import BatchNorm1d, BatchNorm3d


def test_batchnorm1d_creation():
    """Test BatchNorm1d initialization."""
    # Default parameters
    bn = BatchNorm1d(64)
    assert bn.num_features == 64
    assert bn.eps == 1e-5
    assert bn.momentum == 0.1
    assert bn.affine == True
    assert bn.track_running_stats == True
    assert bn.training == True

    # Check parameters
    if bn.affine:
        assert bn.gamma is not None
        assert bn.beta is not None
        assert bn.gamma.shape == (64,)
        assert bn.beta.shape == (64,)
        assert bn.gamma.requires_grad == True
        assert bn.beta.requires_grad == True

    # Check buffers
    if bn.track_running_stats:
        assert bn.running_mean is not None
        assert bn.running_var is not None
        assert bn.num_batches_tracked is not None
        assert bn.running_mean.shape == (64,)
        assert bn.running_var.shape == (64,)
        assert np.allclose(bn.running_mean.data, 0.0)
        assert np.allclose(bn.running_var.data, 1.0)
        assert bn.num_batches_tracked.data[0] == 0

    # Custom parameters
    bn2 = BatchNorm1d(
        32, eps=1e-3, momentum=0.5, affine=False, track_running_stats=False
    )
    assert bn2.num_features == 32
    assert bn2.eps == 1e-3
    assert bn2.momentum == 0.5
    assert bn2.affine == False
    assert bn2.track_running_stats == False
    assert bn2.gamma is None
    assert bn2.beta is None
    assert bn2.running_mean is None
    assert bn2.running_var is None
    assert bn2.num_batches_tracked is None

    print("✓ test_batchnorm1d_creation passed")


def test_batchnorm1d_forward_2d():
    """Test BatchNorm1d forward pass with 2D input (N, C)."""
    bn = BatchNorm1d(16, track_running_stats=True)
    x = nt.Tensor.randn((8, 16), requires_grad=False)

    # Forward pass
    out = bn(x)

    # Check output shape
    assert out.shape == x.shape

    # Check that output is approximately normalized (mean ~0, std ~1)
    out_mean = out.mean().item()
    out_std = ((out - out.mean()) ** 2).mean().item() ** 0.5
    assert abs(out_mean) < 0.1, f"Output mean {out_mean} not near 0"
    assert abs(out_std - 1.0) < 0.1, f"Output std {out_std} not near 1"

    print("✓ test_batchnorm1d_forward_2d passed")


def test_batchnorm1d_forward_3d():
    """Test BatchNorm1d forward pass with 3D input (N, C, L)."""
    bn = BatchNorm1d(16, track_running_stats=True)
    x = nt.Tensor.randn((8, 16, 10), requires_grad=False)

    # Forward pass
    out = bn(x)

    # Check output shape
    assert out.shape == x.shape

    # Check that output is approximately normalized (mean ~0, std ~1)
    out_mean = out.mean().item()
    out_std = ((out - out.mean()) ** 2).mean().item() ** 0.5
    assert abs(out_mean) < 0.1, f"Output mean {out_mean} not near 0"
    assert abs(out_std - 1.0) < 0.1, f"Output std {out_std} not near 1"

    print("✓ test_batchnorm1d_forward_3d passed")


def test_batchnorm1d_forward_eval():
    """Test forward pass in evaluation mode."""
    bn = BatchNorm1d(16, track_running_stats=True)
    x = nt.Tensor.randn((8, 16, 5), requires_grad=False)

    # Train for one step to update running stats
    out_train = bn(x)
    running_mean_before = bn.running_mean.data.copy()
    running_var_before = bn.running_var.data.copy()

    # Switch to eval mode
    bn.eval()
    assert bn.training == False

    # Forward pass with different input (should use running stats)
    x2 = nt.Tensor.randn((4, 16, 5), requires_grad=False)
    out_eval = bn(x2)

    # Check output shape
    assert out_eval.shape == x2.shape

    # Running stats should not change in eval mode
    assert np.allclose(bn.running_mean.data, running_mean_before)
    assert np.allclose(bn.running_var.data, running_var_before)

    print("✓ test_batchnorm1d_forward_eval passed")


def test_batchnorm1d_no_track_running_stats():
    """Test batch norm without tracking running statistics."""
    bn = BatchNorm1d(16, track_running_stats=False)
    x = nt.Tensor.randn((8, 16, 7), requires_grad=False)

    # Should use batch statistics in both train and eval modes
    out_train = bn(x)
    bn.eval()
    out_eval = bn(x)

    # Outputs should be different (different batch statistics)
    # but shapes match
    assert out_train.shape == out_eval.shape

    print("✓ test_batchnorm1d_no_track_running_stats passed")


def test_batchnorm1d_affine_false():
    """Test batch norm without affine transformation."""
    bn = BatchNorm1d(16, affine=False)
    x = nt.Tensor.randn((8, 16, 5), requires_grad=False)

    out = bn(x)
    assert out.shape == x.shape

    # Without affine, output should be exactly normalized (mean ~0, std ~1)
    out_mean = out.mean().item()
    out_std = ((out - out.mean()) ** 2).mean().item() ** 0.5
    assert abs(out_mean) < 1e-5, f"Output mean {out_mean} not near 0"
    assert abs(out_std - 1.0) < 1e-5, f"Output std {out_std} not near 1"

    print("✓ test_batchnorm1d_affine_false passed")


def test_batchnorm1d_gradient_flow():
    """Test gradient computation through BatchNorm1d."""
    bn = BatchNorm1d(8, affine=True)
    x = nt.Tensor.randn((4, 8, 6), requires_grad=True)

    # Forward pass
    out = bn(x)

    # Create a loss (sum of outputs)
    loss = out.sum()

    # Backward pass
    loss.backward()

    # Check gradients
    assert x.grad is not None
    assert x.grad.shape == x.shape

    if bn.affine:
        assert bn.gamma.grad is not None
        assert bn.beta.grad is not None
        assert bn.gamma.grad.shape == bn.gamma.shape
        assert bn.beta.grad.shape == bn.beta.shape

    print("✓ test_batchnorm1d_gradient_flow passed")


def test_batchnorm3d_creation():
    """Test BatchNorm3d initialization."""
    # Default parameters
    bn = BatchNorm3d(64)
    assert bn.num_features == 64
    assert bn.eps == 1e-5
    assert bn.momentum == 0.1
    assert bn.affine == True
    assert bn.track_running_stats == True
    assert bn.training == True

    # Check parameters
    if bn.affine:
        assert bn.gamma is not None
        assert bn.beta is not None
        assert bn.gamma.shape == (64,)
        assert bn.beta.shape == (64,)
        assert bn.gamma.requires_grad == True
        assert bn.beta.requires_grad == True

    # Check buffers
    if bn.track_running_stats:
        assert bn.running_mean is not None
        assert bn.running_var is not None
        assert bn.num_batches_tracked is not None
        assert bn.running_mean.shape == (64,)
        assert bn.running_var.shape == (64,)
        assert np.allclose(bn.running_mean.data, 0.0)
        assert np.allclose(bn.running_var.data, 1.0)
        assert bn.num_batches_tracked.data[0] == 0

    # Custom parameters
    bn2 = BatchNorm3d(
        32, eps=1e-3, momentum=0.5, affine=False, track_running_stats=False
    )
    assert bn2.num_features == 32
    assert bn2.eps == 1e-3
    assert bn2.momentum == 0.5
    assert bn2.affine == False
    assert bn2.track_running_stats == False
    assert bn2.gamma is None
    assert bn2.beta is None
    assert bn2.running_mean is None
    assert bn2.running_var is None
    assert bn2.num_batches_tracked is None

    print("✓ test_batchnorm3d_creation passed")


def test_batchnorm3d_forward():
    """Test BatchNorm3d forward pass with 5D input (N, C, D, H, W)."""
    bn = BatchNorm3d(16, track_running_stats=True)
    x = nt.Tensor.randn((8, 16, 5, 6, 7), requires_grad=False)

    # Forward pass
    out = bn(x)

    # Check output shape
    assert out.shape == x.shape

    # Check that output is approximately normalized (mean ~0, std ~1)
    out_mean = out.mean().item()
    out_std = ((out - out.mean()) ** 2).mean().item() ** 0.5
    assert abs(out_mean) < 0.1, f"Output mean {out_mean} not near 0"
    assert abs(out_std - 1.0) < 0.1, f"Output std {out_std} not near 1"

    print("✓ test_batchnorm3d_forward passed")


def test_batchnorm3d_forward_eval():
    """Test forward pass in evaluation mode."""
    bn = BatchNorm3d(16, track_running_stats=True)
    x = nt.Tensor.randn((8, 16, 3, 4, 5), requires_grad=False)

    # Train for one step to update running stats
    out_train = bn(x)
    running_mean_before = bn.running_mean.data.copy()
    running_var_before = bn.running_var.data.copy()

    # Switch to eval mode
    bn.eval()
    assert bn.training == False

    # Forward pass with different input (should use running stats)
    x2 = nt.Tensor.randn((4, 16, 3, 4, 5), requires_grad=False)
    out_eval = bn(x2)

    # Check output shape
    assert out_eval.shape == x2.shape

    # Running stats should not change in eval mode
    assert np.allclose(bn.running_mean.data, running_mean_before)
    assert np.allclose(bn.running_var.data, running_var_before)

    print("✓ test_batchnorm3d_forward_eval passed")


def test_batchnorm3d_no_track_running_stats():
    """Test batch norm without tracking running statistics."""
    bn = BatchNorm3d(16, track_running_stats=False)
    x = nt.Tensor.randn((8, 16, 2, 3, 4), requires_grad=False)

    # Should use batch statistics in both train and eval modes
    out_train = bn(x)
    bn.eval()
    out_eval = bn(x)

    # Outputs should be different (different batch statistics)
    # but shapes match
    assert out_train.shape == out_eval.shape

    print("✓ test_batchnorm3d_no_track_running_stats passed")


def test_batchnorm3d_affine_false():
    """Test batch norm without affine transformation."""
    bn = BatchNorm3d(16, affine=False)
    x = nt.Tensor.randn((8, 16, 2, 3, 4), requires_grad=False)

    out = bn(x)
    assert out.shape == x.shape

    # Without affine, output should be exactly normalized (mean ~0, std ~1)
    out_mean = out.mean().item()
    out_std = ((out - out.mean()) ** 2).mean().item() ** 0.5
    assert abs(out_mean) < 1e-5, f"Output mean {out_mean} not near 0"
    assert abs(out_std - 1.0) < 1e-5, f"Output std {out_std} not near 1"

    print("✓ test_batchnorm3d_affine_false passed")


def test_batchnorm3d_gradient_flow():
    """Test gradient computation through BatchNorm3d."""
    bn = BatchNorm3d(8, affine=True)
    x = nt.Tensor.randn((4, 8, 3, 4, 5), requires_grad=True)

    # Forward pass
    out = bn(x)

    # Create a loss (sum of outputs)
    loss = out.sum()

    # Backward pass
    loss.backward()

    # Check gradients
    assert x.grad is not None
    assert x.grad.shape == x.shape

    if bn.affine:
        assert bn.gamma.grad is not None
        assert bn.beta.grad is not None
        assert bn.gamma.grad.shape == bn.gamma.shape
        assert bn.beta.grad.shape == bn.beta.shape

    print("✓ test_batchnorm3d_gradient_flow passed")


def test_batchnorm1d_invalid_input():
    """Test error handling for invalid inputs in BatchNorm1d."""
    bn = BatchNorm1d(4)

    # Wrong number of dimensions (1D)
    try:
        x = nt.Tensor.randn((4,))  # 1D
        bn(x)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "expects 2D or 3D input" in str(e)

    # Wrong number of channels
    try:
        x = nt.Tensor.randn((2, 5, 10))  # 5 channels, expected 4
        bn(x)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "channels" in str(e).lower()

    print("✓ test_batchnorm1d_invalid_input passed")


def test_batchnorm3d_invalid_input():
    """Test error handling for invalid inputs in BatchNorm3d."""
    bn = BatchNorm3d(4)

    # Wrong number of dimensions (4D)
    try:
        x = nt.Tensor.randn((2, 4, 3, 4))  # 4D
        bn(x)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "expects 5D input" in str(e)

    # Wrong number of channels
    try:
        x = nt.Tensor.randn((2, 5, 3, 4, 5))  # 5 channels, expected 4
        bn(x)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "channels" in str(e).lower()

    print("✓ test_batchnorm3d_invalid_input passed")


def run_all_tests():
    """Run all batch norm 1D/3D tests."""
    test_batchnorm1d_creation()
    test_batchnorm1d_forward_2d()
    test_batchnorm1d_forward_3d()
    test_batchnorm1d_forward_eval()
    test_batchnorm1d_no_track_running_stats()
    test_batchnorm1d_affine_false()
    test_batchnorm1d_gradient_flow()
    test_batchnorm1d_invalid_input()
    
    test_batchnorm3d_creation()
    test_batchnorm3d_forward()
    test_batchnorm3d_forward_eval()
    test_batchnorm3d_no_track_running_stats()
    test_batchnorm3d_affine_false()
    test_batchnorm3d_gradient_flow()
    test_batchnorm3d_invalid_input()
    
    print("\n✅ All BatchNorm1d/3D tests passed!")


if __name__ == "__main__":
    run_all_tests()