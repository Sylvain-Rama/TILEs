from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .geometry import polygon_bbox, polygon_center, translate_polygon
from .protocols import Partitioner, RepresentativeColorPolicy, StopPolicy
from .statistics import region_stats, representative_region_color
from .types import ImageContext, Region, RegionStats, Tile, TileSet


@dataclass(frozen=True)
class ThresholdPolicy:
    standard_deviation: float = 40.0
    max_level: int = 6
    heightmap_threshold: float = 0.1
    heightmap_decay: float = 0.1

    def should_stop(self, region: Region, stats: RegionStats) -> bool:
        return (
            stats.standard_deviation < self.standard_deviation
            or region.level >= self.max_level
            or stats.heightmap_max - region.level * self.heightmap_decay < self.heightmap_threshold
        )


def extract_tile(context: ImageContext, region: Region) -> Tile:
    image_height, image_width = context.rgba.shape[:2]
    bbox = polygon_bbox(region.polygon, image_width, image_height)
    left, top = bbox[:2]
    cropped = context.image.crop(bbox)
    polygon = translate_polygon(region.polygon, left, top)

    mask = Image.new("L", cropped.size, 0)
    ImageDraw.Draw(mask).polygon(
        [tuple(point) for point in polygon],
        fill=255,
    )
    cropped.putalpha(mask)

    return Tile(
        region=region,
        bbox=bbox,
        center=polygon_center(region.polygon),
        color=(0, 0, 0),
        image=cropped,
    )


def grow_tree(
    context: ImageContext,
    root: Region,
    partitioner: Partitioner,
    stop_policy: StopPolicy,
    *,
    rng: np.random.Generator,
    color_policy: RepresentativeColorPolicy | None = None,
) -> TileSet:
    tiles: list[Tile] = []

    def add_tile(region: Region, stats: RegionStats) -> None:
        tile = extract_tile(context, region)
        tile.color = (
            stats.mean_color
            if color_policy is None
            else representative_region_color(context, region, color_policy)
        )
        tiles.append(tile)

    def visit(region: Region) -> None:
        stats = region_stats(context, region)
        if stop_policy.should_stop(region, stats):
            add_tile(region, stats)
            return

        children = partitioner.split(region, context, rng)
        if not children:
            add_tile(region, stats)
            return
        for child in children:
            visit(child)

    visit(root)
    return TileSet(tuple(tiles))
