import numpy as np
from PIL import Image, ImageDraw

from .geometry import polygon_bbox, translate_polygon
from .protocols import RepresentativeColorPolicy
from .types import ImageContext, Region, RegionStats, RGBColor


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


def _region_samples(context: ImageContext, region: Region) -> tuple[np.ndarray, np.ndarray]:
    height, width = context.rgba.shape[:2]
    bbox = polygon_bbox(region.polygon, width, height)
    left, top, right, bottom = bbox
    if right <= left or bottom <= top:
        return (
            np.empty((0, 3), dtype=context.rgba.dtype),
            np.empty((0,), dtype=context.heightmap.dtype),
        )

    pixels = context.rgba[top:bottom, left:right]
    heightmap = context.heightmap[top:bottom, left:right]
    visible = (pixels[:, :, 3] != 0) & _region_mask(region, bbox)
    return pixels[visible, :3], heightmap[visible]


def representative_region_color(
    context: ImageContext,
    region: Region,
    policy: RepresentativeColorPolicy,
) -> RGBColor:
    selected, _ = _region_samples(context, region)
    if selected.size == 0:
        return (0, 0, 0)
    return policy.representative_color(selected)


def region_stats(context: ImageContext, region: Region) -> RegionStats:
    selected, selected_heightmap = _region_samples(context, region)
    if selected.size == 0:
        return RegionStats(0.0, (0, 0, 0), 0.0, 0)

    standard_deviation = float(np.max(np.std(selected, axis=0)))
    color = tuple(np.mean(selected, axis=0).astype(np.uint8).tolist())
    return RegionStats(
        standard_deviation=standard_deviation,
        mean_color=color,
        heightmap_max=float(np.max(selected_heightmap)),
        visible_pixel_count=len(selected),
    )
