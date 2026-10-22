import math
from functools import cached_property

import numpy as np
from PIL import Image, ImageDraw

from .geometry import bounds_bbox, polygon_bbox, polygon_center, rectangle_bounds, translate_polygon
from .protocols import RepresentativeColorPolicy
from .types import ImageContext, Region, RegionStats, RGBColor

_EMPTY_STATS = RegionStats(0.0, (0, 0, 0), 0.0, 0)


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


def _region_mask(region: Region, bbox: tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = bbox
    width = max(0, right - left)
    height = max(0, bottom - top)
    mask_image = Image.new("1", (width, height), 0)
    ImageDraw.Draw(mask_image).polygon(
        [tuple(point) for point in translate_polygon(region.polygon, left, top)],
        fill=1,
    )
    return np.asarray(mask_image, dtype=bool)


def _stats_from_samples(selected: np.ndarray, selected_heightmap: np.ndarray) -> RegionStats:
    if selected.size == 0:
        return _EMPTY_STATS

    standard_deviation = float(np.max(np.std(selected, axis=0)))
    color = tuple(np.mean(selected, axis=0).astype(np.uint8).tolist())
    return RegionStats(
        standard_deviation=standard_deviation,
        mean_color=color,
        heightmap_max=float(np.max(selected_heightmap)),
        visible_pixel_count=len(selected),
    )


def _rectangle_stats(context: ImageContext, bbox: tuple[int, int, int, int]) -> RegionStats | None:
    integral = context.integral
    if integral is None:
        return None

    left, top, right, bottom = bbox
    count, sums, square_sums = integral.box_sums(left, top, right, bottom)
    if count == 0:
        return _EMPTY_STATS

    # Plain floats: this runs once per tree node, where numpy call overhead dominates.
    means = [total / count for total in sums]
    variance = max(square_total / count - mean * mean for square_total, mean in zip(square_sums, means))
    return RegionStats(
        standard_deviation=math.sqrt(max(variance, 0.0)),
        mean_color=(int(means[0]), int(means[1]), int(means[2])),
        heightmap_max=float(context.visible_heightmap[top:bottom, left:right].max()),
        visible_pixel_count=count,
    )


class RegionAnalysis:
    """Pixel statistics of one region.

    The polygon mask is built at most once, and not at all for axis-aligned rectangles, whose
    rasterised mask is always their whole bounding box. Statistics of rectangles come from the
    context's summed-area tables when available.
    """

    def __init__(self, context: ImageContext, region: Region) -> None:
        self.context = context
        self.region = region
        height, width = context.rgba.shape[:2]
        self._bounds = rectangle_bounds(region.polygon)
        self._is_rectangle = self._bounds is not None
        if self._bounds is None:
            self.bbox = polygon_bbox(region.polygon, width, height)
        else:
            self.bbox = bounds_bbox(self._bounds, width, height)
        left, top, right, bottom = self.bbox
        self._is_empty = right <= left or bottom <= top

    @property
    def center(self) -> tuple[float, float]:
        if self._bounds is None:
            return polygon_center(self.region.polygon)
        left, top, right, bottom = self._bounds
        return (left + right) / 2, (top + bottom) / 2

    @cached_property
    def samples(self) -> tuple[np.ndarray, np.ndarray]:
        """RGB values and heightmap values of the visible pixels inside the region."""
        context = self.context
        if self._is_empty:
            return (
                np.empty((0, 3), dtype=context.rgba.dtype),
                np.empty((0,), dtype=context.heightmap.dtype),
            )

        left, top, right, bottom = self.bbox
        pixels = context.rgba[top:bottom, left:right]
        visible = pixels[:, :, 3] != 0
        if not self._is_rectangle:
            visible &= _region_mask(self.region, self.bbox)
        return pixels[visible, :3], context.heightmap[top:bottom, left:right][visible]

    @cached_property
    def stats(self) -> RegionStats:
        if self._is_empty:
            return _EMPTY_STATS
        if self._is_rectangle:
            stats = _rectangle_stats(self.context, self.bbox)
            if stats is not None:
                return stats
        return _stats_from_samples(*self.samples)

    def representative_color(self, policy: RepresentativeColorPolicy) -> RGBColor:
        selected, _ = self.samples
        if selected.size == 0:
            return (0, 0, 0)
        return policy.representative_color(selected)


def representative_region_color(
    context: ImageContext,
    region: Region,
    policy: RepresentativeColorPolicy,
) -> RGBColor:
    return RegionAnalysis(context, region).representative_color(policy)


def region_stats(context: ImageContext, region: Region) -> RegionStats:
    return RegionAnalysis(context, region).stats
