import numpy as np

# Tables cost about 40 bytes per pixel, so larger images fall back to slicing the pixel array.
INTEGRAL_MAX_PIXELS = 4_000_000

# Per-channel sums are stored as uint32, which bounds the number of pixels.
_UINT32_PIXEL_LIMIT = (2**32 - 1) // 255


def _summed_area_table(values: np.ndarray, dtype: type[np.unsignedinteger]) -> np.ndarray:
    height, width = values.shape[:2]
    table = np.zeros((height + 1, width + 1, *values.shape[2:]), dtype=dtype)
    body = table[1:, 1:]
    np.cumsum(values, axis=0, dtype=dtype, out=body)
    np.cumsum(body, axis=1, out=body)
    return table


class IntegralImage:
    """Summed-area tables over the visible (non-transparent) pixels of an RGBA image."""

    def __init__(self, rgba: np.ndarray) -> None:
        height, width = rgba.shape[:2]
        if height * width > _UINT32_PIXEL_LIMIT:
            raise ValueError("Image is too large for summed-area tables")

        visible = rgba[:, :, 3] != 0
        rgb = rgba[:, :, :3] * visible[:, :, None]
        self.counts = _summed_area_table(visible, np.uint32)
        self.sums = _summed_area_table(rgb, np.uint32)
        self.square_sums = _summed_area_table(rgb.astype(np.uint16) ** 2, np.uint64)

    @staticmethod
    def _box(table: np.ndarray, left: int, top: int, right: int, bottom: int) -> list[int]:
        # Python ints: exact, and no unsigned wrap-around or numpy overhead on tiny vectors.
        bottom_right = table[bottom, right].tolist()
        bottom_left = table[bottom, left].tolist()
        top_right = table[top, right].tolist()
        top_left = table[top, left].tolist()
        if not isinstance(bottom_right, list):
            return [bottom_right - bottom_left - top_right + top_left]
        return [br - bl - tr + tl for br, bl, tr, tl in zip(bottom_right, bottom_left, top_right, top_left)]

    def box_sums(self, left: int, top: int, right: int, bottom: int) -> tuple[int, list[int], list[int]]:
        """Visible pixel count, per-channel sums and sums of squares over pixels [top:bottom, left:right]."""
        return (
            self._box(self.counts, left, top, right, bottom)[0],
            self._box(self.sums, left, top, right, bottom),
            self._box(self.square_sums, left, top, right, bottom),
        )


def build_integral_image(rgba: np.ndarray) -> IntegralImage | None:
    height, width = rgba.shape[:2]
    if height * width > min(INTEGRAL_MAX_PIXELS, _UINT32_PIXEL_LIMIT):
        return None
    return IntegralImage(rgba)
