import numpy as np
from PIL import Image, ImageDraw

from .geometry import polygon_bbox, translate_polygon
from .types import ImageContext, Region, RegionStats


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


def region_stats(context: ImageContext, region: Region) -> RegionStats:
    height, width = context.rgba.shape[:2]
    bbox = polygon_bbox(region.polygon, width, height)
    left, top, right, bottom = bbox
    if right <= left or bottom <= top:
        return RegionStats(0.0, (0, 0, 0), 0.0, 0)

    pixels = context.rgba[top:bottom, left:right]
    heightmap = context.heightmap[top:bottom, left:right]
    visible = (pixels[:, :, 3] != 0) & _region_mask(region, bbox)
    selected = pixels[visible, :3]
    selected_heightmap = heightmap[visible]
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
