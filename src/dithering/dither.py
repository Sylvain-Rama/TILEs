import numpy as np
from PIL import Image

from .dithering_kernels import DITHERING_KERNELS


def _get_new_val(old_val, nc):
    return np.round(old_val * (nc - 1)) / (nc - 1)


def dither(img, kernel="Floyd-Steinberg", nc=2):
    """Dither a PIL image using one of the bundled error-diffusion kernels."""
    if not isinstance(img, Image.Image):
        raise TypeError("Image must be a valid PIL image")
    if kernel not in DITHERING_KERNELS:
        raise ValueError(f"Available dithering kernels are {DITHERING_KERNELS.keys()}")

    img = img.convert("L")
    width, height = img.size
    arr = np.array(img, dtype=float) / 255

    ker = np.asarray(DITHERING_KERNELS[kernel], dtype=float)
    ker = ker / np.sum(ker)
    ker_h, ker_w = ker.shape
    ker_h //= 2
    ker_w //= 2
    pad = max(ker_h, ker_w)
    arr = np.pad(arr, pad)

    for row in range(height):
        for column in range(width):
            old_val = arr[row + pad, column + pad].copy()
            new_val = _get_new_val(old_val, nc)
            arr[row + pad, column + pad] = new_val
            error = old_val - new_val
            arr[
                row + pad - ker_h : row + pad + ker_h + 1,
                column + pad - ker_w : column + pad + ker_w + 1,
            ] += (
                error * ker
            )

    cropped = arr[pad:-pad, pad:-pad]
    return Image.fromarray(np.asarray(cropped / np.max(cropped) * 255, dtype=np.uint8))
