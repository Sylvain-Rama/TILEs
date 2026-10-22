import numpy as np

from .types import RGBColor


def _validate_rgb_pixels(rgb_pixels: np.ndarray) -> None:
    if rgb_pixels.ndim != 2 or rgb_pixels.shape[0] == 0 or rgb_pixels.shape[1] != 3:
        raise ValueError("RGB pixels must be a non-empty array with shape (n, 3)")


class MeanRGBColorPolicy:
    def representative_color(self, rgb_pixels: np.ndarray) -> RGBColor:
        _validate_rgb_pixels(rgb_pixels)
        return tuple(np.rint(np.mean(rgb_pixels, axis=0)).astype(np.uint8).tolist())


class MedianRGBColorPolicy:
    def representative_color(self, rgb_pixels: np.ndarray) -> RGBColor:
        _validate_rgb_pixels(rgb_pixels)
        return tuple(np.rint(np.median(rgb_pixels, axis=0)).astype(np.uint8).tolist())


# Björn Ottosson's OKLab matrices (linear sRGB <-> LMS <-> OKLab).
_RGB_TO_LMS = np.array(
    [
        [0.4122214708, 0.5363325363, 0.0514459929],
        [0.2119034982, 0.6806995451, 0.1073969566],
        [0.0883024619, 0.2817188376, 0.6299787005],
    ]
)
_LMS_TO_LAB = np.array(
    [
        [0.2104542553, 0.7936177850, -0.0040720468],
        [1.9779984951, -2.4285922050, 0.4505937099],
        [0.0259040371, 0.7827717662, -0.8086757660],
    ]
)
_LAB_TO_LMS = np.linalg.inv(_LMS_TO_LAB)
_LMS_TO_RGB = np.linalg.inv(_RGB_TO_LMS)


def _srgb_to_linear(values: np.ndarray) -> np.ndarray:
    values = values / 255.0
    return np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0.0, 1.0)
    return np.where(values <= 0.0031308, values * 12.92, 1.055 * values ** (1 / 2.4) - 0.055)


def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    """Convert (..., 3) sRGB values in 0..255 to OKLab."""
    lms = _srgb_to_linear(np.asarray(rgb, dtype=np.float64)) @ _RGB_TO_LMS.T
    return np.cbrt(lms) @ _LMS_TO_LAB.T


def _oklab_to_linear_rgb(lab: np.ndarray) -> np.ndarray:
    lms = (lab @ _LAB_TO_LMS.T) ** 3
    return lms @ _LMS_TO_RGB.T


def _in_gamut(linear_rgb: np.ndarray, tolerance: float = 1e-6) -> bool:
    return bool(np.all(linear_rgb >= -tolerance) and np.all(linear_rgb <= 1 + tolerance))


def oklab_to_srgb(lab: np.ndarray) -> RGBColor:
    """Convert one OKLab color to sRGB, reducing chroma until it fits the sRGB gamut."""
    lab = np.asarray(lab, dtype=np.float64)
    linear = _oklab_to_linear_rgb(lab)
    if not _in_gamut(linear):
        # Keep lightness and hue: bisect on the chroma scale instead of clipping channels.
        low, high = 0.0, 1.0
        for _ in range(24):
            middle = (low + high) / 2
            candidate = _oklab_to_linear_rgb(np.array([lab[0], lab[1] * middle, lab[2] * middle]))
            if _in_gamut(candidate):
                low = middle
            else:
                high = middle
        linear = _oklab_to_linear_rgb(np.array([lab[0], lab[1] * low, lab[2] * low]))
    srgb = np.rint(_linear_to_srgb(linear) * 255.0).astype(np.uint8)
    return (int(srgb[0]), int(srgb[1]), int(srgb[2]))


class OKLabChromaColorPolicy:
    """Mean colour in OKLab, with chroma pulled back towards the pixels' mean chroma.

    Averaging differently coloured pixels gives a colour greyer than any of them. This policy keeps
    the mean lightness and hue direction, and scales the chroma of the average from the chroma of
    the mean vector (``vividness=0``, the plain OKLab mean) towards the mean chroma of the pixels
    (``vividness=1``). Geometrically, ``chroma = mean_chroma**v * mean_vector_chroma**(1 - v)``.

    Full restoration is not the default: when hues nearly cancel, the leftover direction is
    arbitrary and amplifying it all the way produces a misleadingly vivid colour.
    """

    def __init__(self, vividness: float = 0.5) -> None:
        if not 0.0 <= vividness <= 1.0:
            raise ValueError("vividness must be between 0 and 1")
        self.vividness = vividness

    def representative_color(self, rgb_pixels: np.ndarray) -> RGBColor:
        _validate_rgb_pixels(rgb_pixels)
        lab = srgb_to_oklab(rgb_pixels)
        lightness = float(lab[:, 0].mean())
        mean_a, mean_b = float(lab[:, 1].mean()), float(lab[:, 2].mean())
        mean_chroma = float(np.hypot(lab[:, 1], lab[:, 2]).mean())

        direction = float(np.hypot(mean_a, mean_b))
        if direction < 1e-9:
            return oklab_to_srgb(np.array([lightness, mean_a, mean_b]))
        scale = (mean_chroma / direction) ** self.vividness
        return oklab_to_srgb(np.array([lightness, mean_a * scale, mean_b * scale]))


class DominantColorPolicy:
    """Mean of the pixels in the most populated colour bin.

    Pixels are quantised to ``bits`` bits per channel. Only the most frequent bin contributes, so
    different hues are never mixed. Ties resolve to the lowest bin, which keeps results deterministic.
    """

    def __init__(self, bits: int = 4) -> None:
        if not 1 <= bits <= 8:
            raise ValueError("bits must be between 1 and 8")
        self.bits = bits

    def representative_color(self, rgb_pixels: np.ndarray) -> RGBColor:
        _validate_rgb_pixels(rgb_pixels)
        pixels = rgb_pixels.astype(np.int64)
        shift = 8 - self.bits
        bins = (
            ((pixels[:, 0] >> shift) << (2 * self.bits))
            | ((pixels[:, 1] >> shift) << self.bits)
            | (pixels[:, 2] >> shift)
        )
        dominant = np.bincount(bins).argmax()
        color = np.rint(pixels[bins == dominant].mean(axis=0)).astype(np.uint8)
        return (int(color[0]), int(color[1]), int(color[2]))
