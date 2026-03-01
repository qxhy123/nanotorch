"""
Tests for nanotorch.Tensor class and operations.
"""

import numpy as np
import nanotorch as nt


def test_tensor_creation():
    """Test basic tensor creation."""
    # From list
    t = nt.Tensor([1, 2, 3])
    assert t.shape == (3,)
    assert np.allclose(t.data, [1, 2, 3])

    # From numpy array
    arr = np.array([[1, 2], [3, 4]], dtype=np.float32)
    t = nt.Tensor(arr)
    assert t.shape == (2, 2)
    assert np.allclose(t.data, arr)

    # Scalar
    t = nt.Tensor(5.0)
    assert t.shape == ()
    assert t.item() == 5.0

    print("✓ test_tensor_creation passed")


def test_tensor_operations():
    """Test basic arithmetic operations."""
    a = nt.Tensor([1, 2, 3])
    b = nt.Tensor([4, 5, 6])

    # Addition
    c = a + b
    assert np.allclose(c.data, [5, 7, 9])

    # Subtraction
    c = a - b
    assert np.allclose(c.data, [-3, -3, -3])

    # Multiplication (element-wise)
    c = a * b
    assert np.allclose(c.data, [4, 10, 18])

    # Division
    c = a / b
    assert np.allclose(c.data, [0.25, 0.4, 0.5])

    # Negation
    c = -a
    assert np.allclose(c.data, [-1, -2, -3])

    print("✓ test_tensor_operations passed")


def test_tensor_matmul():
    """Test matrix multiplication."""
    a = nt.Tensor([[1, 2], [3, 4]])
    b = nt.Tensor([[5, 6], [7, 8]])

    c = a @ b
    expected = np.array([[19, 22], [43, 50]], dtype=np.float32)
    assert c.shape == (2, 2)
    assert np.allclose(c.data, expected)

    print("✓ test_tensor_matmul passed")


def test_tensor_autograd_simple():
    """Test automatic differentiation with simple graph."""
    # Create leaf tensors with gradient tracking
    x = nt.Tensor(3.0, requires_grad=True)
    y = nt.Tensor(4.0, requires_grad=True)

    # Build computation graph
    z = x * y  # 12
    w = z + y  # 16

    # Backward pass
    w.backward()

    # Check gradients
    # dz/dx = y = 4, dw/dz = 1, so dx = y * 1 = 4
    # dz/dy = x = 3, dw/dy = 1 (from z) + 1 (direct) = 2? Actually w = z + y, dw/dy = dz/dy + 1 = x + 1 = 4
    # Let's compute analytically: w = x*y + y, dw/dx = y = 4, dw/dy = x + 1 = 4
    assert np.allclose(x.grad.data, 4.0)
    assert np.allclose(y.grad.data, 4.0)

    print("✓ test_tensor_autograd_simple passed")


def test_tensor_no_grad():
    """Test no_grad context manager."""
    x = nt.Tensor(2.0, requires_grad=True)

    with nt.no_grad():
        y = x * 3
        assert not y.requires_grad

    # Outside context, gradient tracking resumes
    z = x * 4
    assert z.requires_grad

    print("✓ test_tensor_no_grad passed")


def test_tensor_factory_methods():
    """Test static factory methods."""
    # Zeros
    z = nt.Tensor.zeros((2, 3))
    assert z.shape == (2, 3)
    assert np.allclose(z.data, 0)

    # Ones
    o = nt.Tensor.ones((3, 2))
    assert o.shape == (3, 2)
    assert np.allclose(o.data, 1)

    # Randn (just check shape)
    r = nt.Tensor.randn((4, 5))
    assert r.shape == (4, 5)

    # Eye
    i = nt.Tensor.eye(3)
    assert i.shape == (3, 3)
    assert np.allclose(i.data, np.eye(3, dtype=np.float32))

    # Arange
    a = nt.Tensor.arange(5)
    assert a.shape == (5,)
    assert np.allclose(a.data, [0, 1, 2, 3, 4])

    a = nt.Tensor.arange(2, 8, 2)
    assert np.allclose(a.data, [2, 4, 6])

    print("✓ test_tensor_factory_methods passed")


def test_tensor_activation_functions():
    """Test activation functions."""
    x = nt.Tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

    # ReLU
    r = x.relu()
    expected = [0, 0, 0, 1, 2]
    assert np.allclose(r.data, expected)

    # Sigmoid
    s = x.sigmoid()
    expected = 1 / (1 + np.exp(-np.array([-2, -1, 0, 1, 2], dtype=np.float32)))
    assert np.allclose(s.data, expected, rtol=1e-5)

    # Tanh
    t = x.tanh()
    expected = np.tanh([-2, -1, 0, 1, 2])
    assert np.allclose(t.data, expected, rtol=1e-5)

    # Softmax (1D)
    x1 = nt.Tensor([1.0, 2.0, 3.0])
    s1 = x1.softmax(dim=-1)
    # numpy softmax
    exp = np.exp([1.0, 2.0, 3.0])
    expected1 = exp / np.sum(exp)
    assert np.allclose(s1.data, expected1, rtol=1e-5)

    # Softmax (2D) along different dimensions
    x2 = nt.Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # dim=0 (columns)
    s2_dim0 = x2.softmax(dim=0)
    exp_dim0 = np.exp(x2.data)
    expected_dim0 = exp_dim0 / np.sum(exp_dim0, axis=0, keepdims=True)
    assert np.allclose(s2_dim0.data, expected_dim0, rtol=1e-5)
    # dim=1 (rows)
    s2_dim1 = x2.softmax(dim=1)
    exp_dim1 = np.exp(x2.data)
    expected_dim1 = exp_dim1 / np.sum(exp_dim1, axis=1, keepdims=True)
    assert np.allclose(s2_dim1.data, expected_dim1, rtol=1e-5)
    # dim=-1 (last dimension, same as rows)
    s2_dim_neg1 = x2.softmax(dim=-1)
    assert np.allclose(s2_dim_neg1.data, expected_dim1, rtol=1e-5)

    # LogSoftmax (1D)
    ls1 = x1.log_softmax(dim=-1)
    expected_ls1 = np.log(expected1)
    assert np.allclose(ls1.data, expected_ls1, rtol=1e-5)

    # LogSoftmax (2D) dim=1
    ls2 = x2.log_softmax(dim=1)
    expected_ls2 = np.log(expected_dim1)
    assert np.allclose(ls2.data, expected_ls2, rtol=1e-5)

    print("✓ test_tensor_activation_functions passed")


def test_tensor_shape_operations():
    """Test reshape, sum, mean, max, min."""
    x = nt.Tensor([[1, 2], [3, 4]])

    # Reshape
    y = x.reshape((4,))
    assert y.shape == (4,)
    assert np.allclose(y.data, [1, 2, 3, 4])

    # Sum
    s = x.sum()
    assert np.allclose(s.data, 10)

    s_axis = x.sum(axis=0)
    assert s_axis.shape == (2,)
    assert np.allclose(s_axis.data, [4, 6])

    # Mean
    m = x.mean()
    assert np.allclose(m.data, 2.5)

    m_axis = x.mean(axis=1)
    assert m_axis.shape == (2,)
    assert np.allclose(m_axis.data, [1.5, 3.5])

    # Max
    max_val = x.max()
    assert np.allclose(max_val.data, 4)

    max_axis = x.max(axis=0)
    assert max_axis.shape == (2,)
    assert np.allclose(max_axis.data, [3, 4])

    # Min
    min_val = x.min()
    assert np.allclose(min_val.data, 1)

    min_axis = x.min(axis=1)
    assert min_axis.shape == (2,)
    assert np.allclose(min_axis.data, [1, 3])

    print("✓ test_tensor_shape_operations passed")


def test_tensor_permute_view():
    """Test permute and view operations."""
    import nanotorch as nt
    
    # Test permute
    x = nt.Tensor(np.arange(24).reshape(2, 3, 4).astype(np.float32))
    
    # Basic permute
    y = x.permute(2, 0, 1)  # (2, 3, 4) -> (4, 2, 3)
    assert y.shape == (4, 2, 3)
    assert np.allclose(y.data, x.data.transpose(2, 0, 1))
    
    # Permute with negative indices
    y2 = x.permute(-1, 0, -2)  # Same as permute(2, 0, 1)
    assert y2.shape == (4, 2, 3)
    assert np.allclose(y2.data, x.data.transpose(2, 0, 1))
    
    # Test view (alias for reshape)
    z = x.view(6, 4)
    assert z.shape == (6, 4)
    assert np.allclose(z.data, x.data.reshape(6, 4))
    
    # Test gradient for permute
    x = nt.Tensor(np.random.randn(2, 3, 4).astype(np.float32), requires_grad=True)
    y = x.permute(2, 0, 1)
    
    # Forward pass
    assert y.shape == (4, 2, 3)
    
    # Backward pass
    grad_output = nt.Tensor(np.ones_like(y.data))
    y.backward(grad_output)
    
    # Gradient should have same shape as input
    assert x.grad is not None
    assert x.grad.shape == x.shape
    
    # Gradient should be permuted back
    # dy/dx = 1 (since y = permute(x)), gradient should be ones permuted back
    # Actually gradient should be ones with inverse permutation
    expected_grad = np.ones_like(y.data).transpose(1, 2, 0)  # inverse of (2,0,1) is (1,2,0)
    assert np.allclose(x.grad.data, expected_grad, rtol=1e-5, atol=1e-5)
    
    # Test gradient for view (should be same as reshape)
    x = nt.Tensor(np.random.randn(2, 3, 4).astype(np.float32), requires_grad=True)
    y = x.view(6, 4)
    
    grad_output = nt.Tensor(np.ones_like(y.data))
    y.backward(grad_output)
    
    assert x.grad is not None
    assert x.grad.shape == x.shape
    # Gradient should be reshaped back
    expected_grad = np.ones_like(x.data)
    assert np.allclose(x.grad.data, expected_grad, rtol=1e-5, atol=1e-5)
    
    print("✓ test_tensor_permute_view passed")


def test_tensor_split_chunk():
    """Test split and chunk operations."""
    import nanotorch as nt
    from nanotorch.utils import split, chunk
    
    # Test split with integer split_size
    x = nt.Tensor(np.arange(12).reshape(3, 4).astype(np.float32), requires_grad=True)
    parts = split(x, 2, dim=1)  # Split along columns into chunks of size 2
    
    assert len(parts) == 2
    assert parts[0].shape == (3, 2)
    assert parts[1].shape == (3, 2)
    
    # Check data
    assert np.allclose(parts[0].data, x.data[:, :2])
    assert np.allclose(parts[1].data, x.data[:, 2:])
    
    # Test gradient for split
    # Compute loss as sum of all parts
    loss = sum(p.sum() for p in parts)
    loss.backward()
    
    assert x.grad is not None
    # Gradient should be ones (since loss = sum of all elements)
    expected_grad = np.ones_like(x.data)
    assert np.allclose(x.grad.data, expected_grad, rtol=1e-5, atol=1e-5)
    
    # Test split with list of sizes
    x = nt.Tensor(np.arange(12).reshape(3, 4).astype(np.float32), requires_grad=True)
    parts = split(x, [1, 3], dim=1)  # First column, then remaining 3 columns
    
    assert len(parts) == 2
    assert parts[0].shape == (3, 1)
    assert parts[1].shape == (3, 3)
    
    # Test chunk
    x = nt.Tensor(np.arange(12).reshape(3, 4).astype(np.float32), requires_grad=True)
    chunks = chunk(x, 2, dim=0)  # Split into 2 chunks along rows
    
    assert len(chunks) == 2
    # First chunk should have 2 rows, second 1 row (since 3 rows split into 2 chunks)
    assert chunks[0].shape == (2, 4)
    assert chunks[1].shape == (1, 4)
    
    # Test gradient for chunk
    loss = sum(c.sum() for c in chunks)
    loss.backward()
    
    assert x.grad is not None
    expected_grad = np.ones_like(x.data)
    assert np.allclose(x.grad.data, expected_grad, rtol=1e-5, atol=1e-5)
    
    # Test split without gradient tracking
    x = nt.Tensor(np.arange(12).reshape(3, 4).astype(np.float32), requires_grad=False)
    parts = split(x, 2, dim=1)
    assert len(parts) == 2
    assert not any(p.requires_grad for p in parts)
    
    print("✓ test_tensor_split_chunk passed")


def test_tensor_gather():
    """Test gather operation."""
    import numpy as np
    
    # Test basic gather along dimension 0
    x = nt.Tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
    index = nt.Tensor([[0, 1, 0], [1, 0, 1]])
    result = x.gather(dim=0, index=index)
    
    # Check shape
    assert result.shape == index.shape
    assert result.shape == (2, 3)
    
    # Check values: result[i,j] = x[index[i,j], j]
    expected = np.array([[1., 5., 3.], [4., 2., 6.]])
    assert np.allclose(result.data, expected, rtol=1e-5, atol=1e-5)
    
    # Test gradient
    loss = result.sum()
    loss.backward()
    
    assert x.grad is not None
    # Gradient should be number of times each element was gathered
    # x[0,0] gathered once (index[0,0]=0), x[0,1] once (index[1,1]=0), x[0,2] once (index[0,2]=0)
    # x[1,0] once (index[1,0]=1), x[1,1] once (index[0,1]=1), x[1,2] once (index[1,2]=1)
    expected_grad = np.ones_like(x.data)
    assert np.allclose(x.grad.data, expected_grad, rtol=1e-5, atol=1e-5)
    
    # Test gather along dimension 1
    x = nt.Tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
    index = nt.Tensor([[0, 0], [1, 2]])
    result = x.gather(dim=1, index=index)
    
    assert result.shape == (2, 2)
    expected = np.array([[1., 1.], [5., 6.]])
    assert np.allclose(result.data, expected, rtol=1e-5, atol=1e-5)
    
    # Test gradient with finite differences
    x = nt.Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
    index = nt.Tensor(np.random.randint(0, 3, size=(3, 4)).astype(np.float32))
    
    def gather_sum(t):
        return t.gather(dim=0, index=index).sum()
    
    # Use gradient checking utility if available, else simple test
    loss = gather_sum(x)
    loss.backward()
    assert x.grad is not None
    
    print("✓ gather tests passed")


def test_tensor_scatter():
    """Test scatter operation."""
    import numpy as np
    
    # Test basic scatter along dimension 0
    x = nt.Tensor(np.zeros((3, 5), dtype=np.float32), requires_grad=True)
    src = nt.Tensor(np.arange(1., 11.).reshape(2, 5), requires_grad=True)
    index = nt.Tensor([[0, 1, 2, 0, 0], [0, 1, 2, 0, 0]])
    
    result = x.scatter(dim=0, index=index, src=src)
    
    # Check shape matches x
    assert result.shape == x.shape
    assert result.shape == (3, 5)
    
    # Check values: src values scattered to positions in x
    expected = np.zeros((3, 5))
    # For each position in index/src, write src[i,j] to x[index[i,j], j]
    # Since index has shape (2,5), we iterate
    for i in range(2):
        for j in range(5):
            idx = int(index.data[i, j])
            expected[idx, j] = src.data[i, j]
    
    assert np.allclose(result.data, expected, rtol=1e-5, atol=1e-5)
    
    # Test gradient
    loss = result.sum()
    loss.backward()
    
    # Gradient for src: each src element contributes to loss through its scattered position
    # Since loss = sum(result), gradient for src is 1 for each element
    assert src.grad is not None
    assert np.allclose(src.grad.data, np.ones_like(src.data), rtol=1e-5, atol=1e-5)
    
    # Gradient for x: zero at positions that were overwritten, 1 elsewhere
    # Since x was all zeros, all positions were overwritten (some multiple times)
    # So gradient for x should be all zeros
    assert x.grad is not None
    # Actually, positions not overwritten should have gradient 1, but all positions were overwritten
    # However, overlapping indices: some positions overwritten multiple times, last write wins
    # Still, all positions have at least one write, so gradient zero
    # Let's compute expected gradient manually
    expected_x_grad = np.ones_like(x.data)
    for i in range(2):
        for j in range(5):
            idx = int(index.data[i, j])
            expected_x_grad[idx, j] = 0  # Position overwritten
    
    assert np.allclose(x.grad.data, expected_x_grad, rtol=1e-5, atol=1e-5)
    
    # Test scatter along dimension 1
    x = nt.Tensor(np.zeros((5, 3), dtype=np.float32), requires_grad=True)
    src = nt.Tensor(np.arange(1., 16.).reshape(5, 3), requires_grad=True)
    index = nt.Tensor([[0, 1, 2], [2, 0, 1], [1, 2, 0], [0, 2, 1], [1, 0, 2]])
    
    result = x.scatter(dim=1, index=index, src=src)
    assert result.shape == (5, 3)
    
    # Verify values
    expected = np.zeros((5, 3))
    for i in range(5):
        for j in range(3):
            idx = int(index.data[i, j])
            expected[i, idx] = src.data[i, j]
    
    assert np.allclose(result.data, expected, rtol=1e-5, atol=1e-5)
    
    # Test gradient with finite differences (simplified)
    x = nt.Tensor(np.random.randn(4, 5).astype(np.float32), requires_grad=True)
    src = nt.Tensor(np.random.randn(4, 5).astype(np.float32), requires_grad=True)
    index = nt.Tensor(np.random.randint(0, 5, size=(4, 5)).astype(np.float32))
    
    def scatter_sum(t, s):
        return t.scatter(dim=1, index=index, src=s).sum()
    
    loss = scatter_sum(x, src)
    loss.backward()
    assert x.grad is not None
    assert src.grad is not None
    
    print("✓ scatter tests passed")


if __name__ == "__main__":
    test_tensor_creation()
    test_tensor_operations()
    test_tensor_matmul()
    test_tensor_autograd_simple()
    test_tensor_no_grad()
    test_tensor_factory_methods()
    test_tensor_activation_functions()
    test_tensor_shape_operations()
    test_tensor_permute_view()
    test_tensor_split_chunk()
    test_tensor_gather()
    test_tensor_scatter()
    print("\nAll tensor tests passed!")
