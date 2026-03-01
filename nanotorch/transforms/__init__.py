"""Data augmentation transforms for nanotorch."""

import numpy as np
from typing import Optional, Tuple, List, Callable, Union
from numpy.typing import NDArray


class Compose:
    """Composes several transforms together.

    Args:
        transforms: List of transforms to compose.

    Example:
        >>> transform = Compose([
        ...     ToFloat(),
        ...     RandomHorizontalFlip(p=0.5),
        ...     Normalize(mean=[0.5], std=[0.5]),
        ... ])
    """

    def __init__(self, transforms: List[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, img: NDArray) -> NDArray:
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += f"\n    {t}"
        format_string += "\n)"
        return format_string


class ToFloat:
    """Convert numpy array to float32."""

    def __call__(self, x: NDArray) -> NDArray:
        return x.astype(np.float32)


class Normalize:
    """Normalize a tensor with mean and standard deviation.

    Args:
        mean: Sequence of means for each channel.
        std: Sequence of standard deviations for each channel.
    """

    def __init__(self, mean: List[float], std: List[float]) -> None:
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)

    def __call__(self, x: NDArray) -> NDArray:
        return (x - self.mean) / self.std


class RandomHorizontalFlip:
    """Horizontally flip the image randomly with a given probability.

    Args:
        p: Probability of the image being flipped (default: 0.5).
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, x: NDArray) -> NDArray:
        if np.random.rand() < self.p:
            if x.ndim == 3:
                return x[:, ::-1, :].copy()
            elif x.ndim == 2:
                return x[:, ::-1].copy()
        return x


class RandomVerticalFlip:
    """Vertically flip the image randomly with a given probability.

    Args:
        p: Probability of the image being flipped (default: 0.5).
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, x: NDArray) -> NDArray:
        if np.random.rand() < self.p:
            if x.ndim == 3:
                return x[::-1, :, :].copy()
            elif x.ndim == 2:
                return x[::-1, :].copy()
        return x


class RandomRotation:
    """Rotate the image by a random angle.

    Args:
        degrees: Range of degrees to select from (-degrees, +degrees).
    """

    def __init__(self, degrees: float) -> None:
        self.degrees = degrees

    def __call__(self, x: NDArray) -> NDArray:
        angle = np.random.uniform(-self.degrees, self.degrees)
        return rotate(x, angle)


class RandomCrop:
    """Crop randomly the image in a square.

    Args:
        size: Desired output size of the crop.
        padding: Optional padding on each border (default: None).
    """

    def __init__(self, size: Union[int, Tuple[int, int]], padding: Optional[int] = None) -> None:
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.padding = padding

    def __call__(self, x: NDArray) -> NDArray:
        if self.padding is not None:
            x = pad(x, self.padding)

        h, w = x.shape[:2]
        th, tw = self.size

        if h < th or w < tw:
            raise ValueError(f"Requested crop size {self.size} is larger than image size {(h, w)}")

        i = np.random.randint(0, h - th + 1)
        j = np.random.randint(0, w - tw + 1)

        if x.ndim == 3:
            return x[i:i+th, j:j+tw, :].copy()
        else:
            return x[i:i+th, j:j+tw].copy()


class CenterCrop:
    """Crop the center of an image.

    Args:
        size: Desired output size of the crop.
    """

    def __init__(self, size: Union[int, Tuple[int, int]]) -> None:
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, x: NDArray) -> NDArray:
        h, w = x.shape[:2]
        th, tw = self.size

        i = (h - th) // 2
        j = (w - tw) // 2

        if x.ndim == 3:
            return x[i:i+th, j:j+tw, :].copy()
        else:
            return x[i:i+th, j:j+tw].copy()


class RandomResizedCrop:
    """Crop the image and resize it to a given size.

    Args:
        size: Expected output size.
        scale: Range of size of the origin size cropped (default: (0.08, 1.0)).
        ratio: Range of aspect ratio (default: (3./4., 4./3.)).
    """

    def __init__(
        self,
        size: Union[int, Tuple[int, int]],
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
    ) -> None:
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, x: NDArray) -> NDArray:
        h, w = x.shape[:2]
        area = h * w

        for _ in range(10):
            target_area = np.random.uniform(*self.scale) * area
            aspect_ratio = np.random.uniform(*self.ratio)

            tw = int(round(np.sqrt(target_area * aspect_ratio)))
            th = int(round(np.sqrt(target_area / aspect_ratio)))

            if 0 < tw <= w and 0 < th <= h:
                i = np.random.randint(0, h - th + 1)
                j = np.random.randint(0, w - tw + 1)

                if x.ndim == 3:
                    cropped = x[i:i+th, j:j+tw, :]
                else:
                    cropped = x[i:i+th, j:j+tw]

                return resize(cropped, self.size)

        return CenterCrop(self.size)(x)


class ColorJitter:
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness: How much to jitter brightness (default: 0).
        contrast: How much to jitter contrast (default: 0).
        saturation: How much to jitter saturation (default: 0).
    """

    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
    ) -> None:
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, x: NDArray) -> NDArray:
        x = x.copy()

        if self.brightness > 0:
            factor = np.random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            x = x * factor

        if self.contrast > 0:
            factor = np.random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            mean = x.mean()
            x = (x - mean) * factor + mean

        if self.saturation > 0 and x.ndim == 3:
            factor = np.random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            gray = np.mean(x, axis=2, keepdims=True)
            x = gray + (x - gray) * factor

        return x.astype(np.float32)


class RandomErasing:
    """Randomly erase a region of the image.

    Args:
        p: Probability of erasing (default: 0.5).
        scale: Range of proportion of erased area (default: (0.02, 0.33)).
        ratio: Range of aspect ratio (default: (0.3, 3.3)).
        value: Erasing value (default: 0).
    """

    def __init__(
        self,
        p: float = 0.5,
        scale: Tuple[float, float] = (0.02, 0.33),
        ratio: Tuple[float, float] = (0.3, 3.3),
        value: float = 0.0,
    ) -> None:
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value

    def __call__(self, x: NDArray) -> NDArray:
        if np.random.rand() > self.p:
            return x

        h, w = x.shape[:2]
        area = h * w

        for _ in range(10):
            target_area = np.random.uniform(*self.scale) * area
            aspect_ratio = np.random.uniform(*self.ratio)

            ew = int(round(np.sqrt(target_area * aspect_ratio)))
            eh = int(round(np.sqrt(target_area / aspect_ratio)))

            if 0 < ew <= w and 0 < eh <= h:
                i = np.random.randint(0, h - eh + 1)
                j = np.random.randint(0, w - ew + 1)

                x = x.copy()
                if x.ndim == 3:
                    x[i:i+eh, j:j+ew, :] = self.value
                else:
                    x[i:i+eh, j:j+ew] = self.value
                return x

        return x


class GaussianBlur:
    """Apply Gaussian blur to the image.

    Args:
        kernel_size: Size of the Gaussian kernel.
        sigma: Standard deviation of the Gaussian kernel (default: (0.1, 2.0)).
    """

    def __init__(
        self,
        kernel_size: int,
        sigma: Tuple[float, float] = (0.1, 2.0),
    ) -> None:
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, x: NDArray) -> NDArray:
        sigma = np.random.uniform(*self.sigma)
        return gaussian_blur(x, self.kernel_size, sigma)


def pad(x: NDArray, padding: int) -> NDArray:
    """Pad an image with zeros.

    Args:
        x: Input image.
        padding: Padding size.

    Returns:
        Padded image.
    """
    if x.ndim == 3:
        return np.pad(x, ((padding,), (padding,), (0,)), mode="constant")
    else:
        return np.pad(x, padding, mode="constant")


def resize(x: NDArray, size: Tuple[int, int]) -> NDArray:
    """Resize an image to a given size.

    Args:
        x: Input image.
        size: Target size (height, width).

    Returns:
        Resized image.
    """
    h, w = x.shape[:2]
    th, tw = size

    if x.ndim == 3:
        result = np.zeros((th, tw, x.shape[2]), dtype=x.dtype)
        for c in range(x.shape[2]):
            result[:, :, c] = _resize_channel(x[:, :, c], th, tw)
    else:
        result = _resize_channel(x, th, tw)

    return result.astype(x.dtype)


def _resize_channel(x: NDArray, th: int, tw: int) -> NDArray:
    """Resize a single channel using nearest neighbor."""
    h, w = x.shape

    row_indices = (np.arange(th) * h / th).astype(int)
    col_indices = (np.arange(tw) * w / tw).astype(int)

    return x[np.ix_(row_indices, col_indices)]


def rotate(x: NDArray, angle: float) -> NDArray:
    """Rotate an image by a given angle.

    Args:
        x: Input image.
        angle: Rotation angle in degrees.

    Returns:
        Rotated image.
    """
    h, w = x.shape[:2]
    cx, cy = w // 2, h // 2

    angle_rad = np.deg2rad(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    y_coords, x_coords = np.mgrid[:h, :w]
    x_coords = x_coords - cx
    y_coords = y_coords - cy

    x_rot = cos_a * x_coords + sin_a * y_coords + cx
    y_rot = -sin_a * x_coords + cos_a * y_coords + cy

    x_rot = np.clip(x_rot, 0, w - 1).astype(int)
    y_rot = np.clip(y_rot, 0, h - 1).astype(int)

    if x.ndim == 3:
        result = np.zeros_like(x)
        for c in range(x.shape[2]):
            result[:, :, c] = x[y_rot, x_rot, c]
    else:
        result = x[y_rot, x_rot]

    return result


def gaussian_blur(x: NDArray, kernel_size: int, sigma: float) -> NDArray:
    """Apply Gaussian blur to an image.

    Args:
        x: Input image.
        kernel_size: Size of the Gaussian kernel.
        sigma: Standard deviation.

    Returns:
        Blurred image.
    """
    ax = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    from scipy.ndimage import convolve

    if x.ndim == 3:
        result = np.zeros_like(x)
        for c in range(x.shape[2]):
            result[:, :, c] = convolve(x[:, :, c], kernel, mode="reflect")
        return result
    else:
        return convolve(x, kernel, mode="reflect")
