"""
Unit tests for device management in nanotorch.
"""

import unittest
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanotorch.device import Device, cpu, is_cuda_available, device_count, cuda


class TestDevice(unittest.TestCase):
    """Test Device class."""

    def test_cpu_device_creation(self):
        """Test creating CPU device."""
        device = Device('cpu')
        self.assertEqual(device.type, 'cpu')
        self.assertEqual(device.index, 0)
        self.assertTrue(device.is_cpu)
        self.assertFalse(device.is_cuda)

    def test_cuda_device_creation(self):
        """Test creating CUDA device."""
        device = Device('cuda', 0)
        self.assertEqual(device.type, 'cuda')
        self.assertEqual(device.index, 0)
        self.assertFalse(device.is_cpu)
        self.assertTrue(device.is_cuda)

    def test_cuda_device_with_index(self):
        """Test creating CUDA device with different index."""
        device = Device('cuda', 2)
        self.assertEqual(device.type, 'cuda')
        self.assertEqual(device.index, 2)

    def test_device_from_string_cpu(self):
        """Test Device.from_string() for CPU."""
        device = Device.from_string('cpu')
        self.assertEqual(device.type, 'cpu')
        self.assertTrue(device.is_cpu)

    def test_device_from_string_cuda(self):
        """Test Device.from_string() for CUDA."""
        device = Device.from_string('cuda')
        self.assertEqual(device.type, 'cuda')
        self.assertTrue(device.is_cuda)

    def test_device_from_string_cuda_with_index(self):
        """Test Device.from_string() for CUDA with index."""
        device = Device.from_string('cuda:1')
        self.assertEqual(device.type, 'cuda')
        self.assertEqual(device.index, 1)

    def test_device_from_string_case_insensitive(self):
        """Test that device string is case insensitive."""
        device = Device.from_string('CUDA')
        self.assertEqual(device.type, 'cuda')
        device = Device.from_string('CPU')
        self.assertEqual(device.type, 'cpu')

    def test_invalid_device_type(self):
        """Test that invalid device type raises error."""
        with self.assertRaises(ValueError):
            Device('invalid')

    def test_invalid_device_string(self):
        """Test that invalid device string raises error."""
        with self.assertRaises(ValueError):
            Device.from_string('invalid')

    def test_device_repr(self):
        """Test device string representation."""
        cpu_device = Device('cpu')
        self.assertEqual(repr(cpu_device), "Device(type='cpu')")

        cuda_device = Device('cuda', 0)
        self.assertEqual(repr(cuda_device), "Device(type='cuda', index=0)")

    def test_device_str(self):
        """Test device string conversion."""
        cpu_device = Device('cpu')
        self.assertEqual(str(cpu_device), 'cpu')

        cuda_device = Device('cuda', 0)
        self.assertEqual(str(cuda_device), 'cuda:0')

        cuda_device_2 = Device('cuda', 2)
        self.assertEqual(str(cuda_device_2), 'cuda:2')

    def test_device_equality(self):
        """Test device equality comparison."""
        cpu1 = Device('cpu')
        cpu2 = Device('cpu')
        self.assertEqual(cpu1, cpu2)

        cuda1 = Device('cuda', 0)
        cuda2 = Device('cuda', 0)
        self.assertEqual(cuda1, cuda2)

        cuda3 = Device('cuda', 1)
        self.assertNotEqual(cuda1, cuda3)

        self.assertNotEqual(cpu1, cuda1)

    def test_device_equality_with_string(self):
        """Test device equality with string."""
        cpu_device = Device('cpu')
        self.assertEqual(cpu_device, 'cpu')

        cuda_device = Device('cuda', 0)
        self.assertEqual(cuda_device, 'cuda')
        self.assertEqual(cuda_device, 'cuda:0')

    def test_device_hash(self):
        """Test device hashing."""
        cpu1 = Device('cpu')
        cpu2 = Device('cpu')
        self.assertEqual(hash(cpu1), hash(cpu2))

        cuda1 = Device('cuda', 0)
        cuda2 = Device('cuda', 0)
        self.assertEqual(hash(cuda1), hash(cuda2))

        # Can use devices as dict keys
        device_dict = {cpu1: 'cpu_value', cuda1: 'cuda_value'}
        self.assertEqual(device_dict[cpu2], 'cpu_value')


class TestDeviceFunctions(unittest.TestCase):
    """Test device utility functions."""

    def test_cpu_predefined(self):
        """Test predefined cpu device."""
        from nanotorch.device import cpu as cpu_device
        self.assertEqual(cpu_device.type, 'cpu')
        self.assertTrue(cpu_device.is_cpu)

    def test_is_cuda_available(self):
        """Test is_cuda_available function."""
        # Should return a boolean (False on systems without CUDA)
        result = is_cuda_available()
        self.assertIsInstance(result, bool)

    def test_device_count(self):
        """Test device_count function."""
        count = device_count()
        self.assertIsInstance(count, int)
        self.assertGreaterEqual(count, 0)

    def test_cuda_namespace(self):
        """Test cuda namespace class."""
        self.assertEqual(cuda.is_available(), is_cuda_available())
        self.assertEqual(cuda.device_count(), device_count())


class TestTensorDevice(unittest.TestCase):
    """Test Tensor device support."""

    def setUp(self):
        """Set up test fixtures."""
        from nanotorch.tensor import Tensor
        self.Tensor = Tensor

    def test_tensor_default_device_cpu(self):
        """Test that default tensor device is CPU."""
        x = self.Tensor([1, 2, 3])
        self.assertTrue(x.device.is_cpu)
        self.assertFalse(x.is_cuda)

    def test_tensor_explicit_cpu_device(self):
        """Test creating tensor with explicit CPU device."""
        x = self.Tensor([1, 2, 3], device='cpu')
        self.assertTrue(x.device.is_cpu)

    def test_tensor_device_property(self):
        """Test tensor device property."""
        x = self.Tensor([1, 2, 3])
        self.assertTrue(x.device.is_cpu)
        self.assertEqual(x.device.type, 'cpu')

    def test_tensor_is_cuda_property(self):
        """Test tensor is_cuda property."""
        x = self.Tensor([1, 2, 3])
        self.assertFalse(x.is_cuda)

    def test_tensor_to_cpu_from_cpu(self):
        """Test tensor.to('cpu') when already on CPU."""
        x = self.Tensor([1, 2, 3])
        y = x.to('cpu')
        # Should return same tensor (no copy needed)
        self.assertIs(y, x)

    def test_tensor_cpu_method(self):
        """Test tensor.cpu() method."""
        x = self.Tensor([1, 2, 3])
        y = x.cpu()
        self.assertTrue(y.device.is_cpu)

    def test_tensor_numpy_on_cpu(self):
        """Test tensor.numpy() on CPU tensor."""
        x = self.Tensor([1, 2, 3])
        arr = x.numpy()
        self.assertIsInstance(arr, np.ndarray)
        np.testing.assert_array_almost_equal(arr, np.array([1, 2, 3], dtype=np.float32))

    def test_tensor_repr_shows_cuda_device(self):
        """Test that __repr__ shows device for CUDA tensors (would need GPU to fully test)."""
        x = self.Tensor([1, 2, 3])
        repr_str = repr(x)
        # CPU tensors should not show device in repr
        self.assertNotIn("device='cuda", repr_str)

    def test_factory_zeros_device(self):
        """Test Tensor.zeros with device parameter."""
        x = self.Tensor.zeros((2, 3), device='cpu')
        self.assertTrue(x.device.is_cpu)
        self.assertEqual(x.shape, (2, 3))

    def test_factory_ones_device(self):
        """Test Tensor.ones with device parameter."""
        x = self.Tensor.ones((2, 3), device='cpu')
        self.assertTrue(x.device.is_cpu)
        self.assertEqual(x.shape, (2, 3))

    def test_factory_randn_device(self):
        """Test Tensor.randn with device parameter."""
        x = self.Tensor.randn((2, 3), device='cpu')
        self.assertTrue(x.device.is_cpu)
        self.assertEqual(x.shape, (2, 3))

    def test_factory_rand_device(self):
        """Test Tensor.rand with device parameter."""
        x = self.Tensor.rand((2, 3), device='cpu')
        self.assertTrue(x.device.is_cpu)
        self.assertEqual(x.shape, (2, 3))

    def test_factory_eye_device(self):
        """Test Tensor.eye with device parameter."""
        x = self.Tensor.eye(3, device='cpu')
        self.assertTrue(x.device.is_cpu)
        self.assertEqual(x.shape, (3, 3))

    def test_factory_arange_device(self):
        """Test Tensor.arange with device parameter."""
        x = self.Tensor.arange(0, 5, device='cpu')
        self.assertTrue(x.device.is_cpu)
        self.assertEqual(x.shape, (5,))

    def test_zeros_like_device(self):
        """Test Tensor.zeros_like preserves device."""
        x = self.Tensor([1, 2, 3])
        y = self.Tensor.zeros_like(x)
        self.assertEqual(y.device, x.device)

    def test_ones_like_device(self):
        """Test Tensor.ones_like preserves device."""
        x = self.Tensor([1, 2, 3])
        y = self.Tensor.ones_like(x)
        self.assertEqual(y.device, x.device)

    def test_clone_preserves_device(self):
        """Test that clone preserves device."""
        x = self.Tensor([1, 2, 3])
        y = x.clone()
        self.assertEqual(y.device, x.device)

    def test_detach_preserves_device(self):
        """Test that detach preserves device."""
        x = self.Tensor([1, 2, 3], requires_grad=True)
        y = x.detach()
        self.assertEqual(y.device, x.device)
        self.assertFalse(y.requires_grad)


class TestModuleDevice(unittest.TestCase):
    """Test Module device support."""

    def setUp(self):
        """Set up test fixtures."""
        from nanotorch.nn import Linear, ReLU
        from nanotorch.nn.module import Module, Sequential
        self.Linear = Linear
        self.ReLU = ReLU
        self.Sequential = Sequential

    def test_module_to_cpu(self):
        """Test module.to('cpu')."""
        model = self.Linear(4, 2)
        model.to('cpu')
        # Check parameters are on CPU
        for param in model.parameters():
            self.assertTrue(param.device.is_cpu)

    def test_module_cpu_method(self):
        """Test module.cpu()."""
        model = self.Linear(4, 2)
        model.cpu()
        for param in model.parameters():
            self.assertTrue(param.device.is_cpu)

    def test_sequential_device(self):
        """Test Sequential module device movement."""
        model = self.Sequential(
            self.Linear(4, 8),
            self.ReLU(),
            self.Linear(8, 2)
        )
        model.to('cpu')
        for param in model.parameters():
            self.assertTrue(param.device.is_cpu)


class TestBackend(unittest.TestCase):
    """Test backend module."""

    def test_get_backend(self):
        """Test get_backend returns default NumPy backend."""
        from nanotorch.backend import get_backend
        backend = get_backend()
        self.assertEqual(backend.name, 'numpy')

    def test_set_backend_cpu(self):
        """Test set_backend to CPU."""
        from nanotorch.backend import set_backend, get_backend
        set_backend('cpu')
        backend = get_backend()
        self.assertEqual(backend.name, 'numpy')

    def test_numpy_backend_array(self):
        """Test NumPyBackend array creation."""
        from nanotorch.backend.numpy_backend import NumPyBackend
        backend = NumPyBackend()
        arr = backend.array([1, 2, 3])
        self.assertIsInstance(arr, np.ndarray)

    def test_numpy_backend_zeros(self):
        """Test NumPyBackend zeros creation."""
        from nanotorch.backend.numpy_backend import NumPyBackend
        backend = NumPyBackend()
        arr = backend.zeros((2, 3))
        self.assertEqual(arr.shape, (2, 3))
        np.testing.assert_array_equal(arr, np.zeros((2, 3)))

    def test_numpy_backend_ones(self):
        """Test NumPyBackend ones creation."""
        from nanotorch.backend.numpy_backend import NumPyBackend
        backend = NumPyBackend()
        arr = backend.ones((2, 3))
        self.assertEqual(arr.shape, (2, 3))
        np.testing.assert_array_equal(arr, np.ones((2, 3)))


if __name__ == '__main__':
    unittest.main()
