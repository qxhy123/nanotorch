"""
Device management for nanotorch.
"""

from typing import Optional, Union


class Device:
    """Represents a computation device (CPU or CUDA GPU).

    Args:
        type: Device type, either 'cpu' or 'cuda'.
        index: Device index for multi-GPU systems (default: 0).

    Examples:
        >>> device_cpu = Device('cpu')
        >>> device_gpu = Device('cuda', 0)
        >>> device_gpu = Device('cuda:0')  # Alternative syntax
    """

    def __init__(self, type: str, index: int = 0):
        self._type = type.lower()
        self._index = index

        if self._type not in ('cpu', 'cuda'):
            raise ValueError(f"Invalid device type: {type}. Must be 'cpu' or 'cuda'.")

    @classmethod
    def from_string(cls, device_string: str) -> 'Device':
        """Create a Device from a string.

        Args:
            device_string: String like 'cpu', 'cuda', or 'cuda:0'.

        Returns:
            Device instance.
        """
        device_string = device_string.lower()
        if device_string == 'cpu':
            return cls('cpu')
        elif device_string == 'cuda':
            return cls('cuda')
        elif device_string.startswith('cuda:'):
            index = int(device_string.split(':')[1])
            return cls('cuda', index)
        else:
            raise ValueError(f"Invalid device string: {device_string}")

    @property
    def type(self) -> str:
        """Return device type ('cpu' or 'cuda')."""
        return self._type

    @property
    def index(self) -> int:
        """Return device index."""
        return self._index

    @property
    def is_cpu(self) -> bool:
        """Check if this is a CPU device."""
        return self._type == 'cpu'

    @property
    def is_cuda(self) -> bool:
        """Check if this is a CUDA device."""
        return self._type == 'cuda'

    def __repr__(self) -> str:
        if self._type == 'cpu':
            return "Device(type='cpu')"
        return f"Device(type='cuda', index={self._index})"

    def __str__(self) -> str:
        if self._type == 'cpu':
            return 'cpu'
        return f'cuda:{self._index}'

    def __eq__(self, other) -> bool:
        if isinstance(other, Device):
            return self._type == other._type and self._index == other._index
        elif isinstance(other, str):
            try:
                other_device = Device.from_string(other)
                return self == other_device
            except ValueError:
                return False
        return False

    def __hash__(self) -> int:
        return hash((self._type, self._index))


# Pre-defined devices
cpu = Device('cpu')


def is_cuda_available() -> bool:
    """Check if CUDA is available.

    Returns:
        True if CuPy can be imported and CUDA is available.
    """
    try:
        import cupy as cp
        # Try to get device properties to confirm CUDA works
        cp.cuda.Device(0).compute_capability
        return True
    except (ImportError, AttributeError, RuntimeError):
        # ImportError: CuPy not installed
        # AttributeError: CUDA module not available
        # RuntimeError: CUDA runtime error
        return False


def device_count() -> int:
    """Return the number of available CUDA devices.

    Returns:
        Number of GPUs, or 0 if CUDA is not available.
    """
    if not is_cuda_available():
        return 0
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount()
    except (AttributeError, RuntimeError):
        return 0


def current_device() -> Device:
    """Return the current CUDA device.

    Returns:
        Current device, or CPU device if CUDA is not available.
    """
    if not is_cuda_available():
        return cpu
    try:
        import cupy as cp
        index = cp.cuda.Device().id
        return Device('cuda', index)
    except (AttributeError, RuntimeError):
        return cpu


def set_device(device: Union[Device, str, int]) -> None:
    """Set the current CUDA device.

    Args:
        device: Device instance, device string, or device index.
    """
    if isinstance(device, int):
        device = Device('cuda', device)
    elif isinstance(device, str):
        device = Device.from_string(device)

    if device.is_cuda and is_cuda_available():
        import cupy as cp
        cp.cuda.Device(device.index).use()


def get_device_name(device: Optional[Device] = None) -> str:
    """Get the name of a CUDA device.

    Args:
        device: Device to query. If None, uses current device.

    Returns:
        Device name string, or 'CPU' for CPU devices.
    """
    if device is None:
        device = current_device()

    if device.is_cpu:
        return 'CPU'

    if not is_cuda_available():
        return 'Unknown GPU'

    try:
        import cupy as cp
        return cp.cuda.Device(device.index).name.decode()
    except (AttributeError, RuntimeError, UnicodeDecodeError):
        return 'Unknown GPU'


def get_device_capability(device: Optional[Device] = None) -> tuple:
    """Get the compute capability of a CUDA device.

    Args:
        device: Device to query. If None, uses current device.

    Returns:
        Tuple of (major, minor) compute capability, or (0, 0) for CPU.
    """
    if device is None:
        device = current_device()

    if device.is_cpu:
        return (0, 0)

    if not is_cuda_available():
        return (0, 0)

    try:
        import cupy as cp
        return cp.cuda.Device(device.index).compute_capability
    except (AttributeError, RuntimeError):
        return (0, 0)


class cuda:
    """CUDA device management namespace (similar to torch.cuda)."""

    @staticmethod
    def is_available() -> bool:
        """Check if CUDA is available."""
        return is_cuda_available()

    @staticmethod
    def device_count() -> int:
        """Return the number of available CUDA devices."""
        return device_count()

    @staticmethod
    def current_device() -> Device:
        """Return the current CUDA device."""
        return current_device()

    @staticmethod
    def set_device(device: Union[Device, str, int]) -> None:
        """Set the current CUDA device."""
        set_device(device)

    @staticmethod
    def get_device_name(device: Optional[Device] = None) -> str:
        """Get the name of a CUDA device."""
        return get_device_name(device)

    @staticmethod
    def get_device_capability(device: Optional[Device] = None) -> tuple:
        """Get the compute capability of a CUDA device."""
        return get_device_capability(device)

    @staticmethod
    def synchronize(device: Optional[Device] = None) -> None:
        """Synchronize CUDA stream.

        Args:
            device: Device to synchronize. If None, uses current device.
        """
        if not is_cuda_available():
            return

        try:
            import cupy as cp
            if device is None:
                cp.cuda.Stream.null.synchronize()
            else:
                with cp.cuda.Device(device.index):
                    cp.cuda.Stream.null.synchronize()
        except (AttributeError, RuntimeError):
            pass

    # Device as static property
    Device = Device
