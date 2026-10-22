from dataclasses import dataclass
from functools import partial

import numpy as np
from PIL import Image, ImageDraw

from .geometry import translate_polygon
from .protocols import Partitioner, RepresentativeColorPolicy, StopPolicy
from .statistics import RegionAnalysis
from .types import ImageContext, Region, RegionStats, RGBColor, Tile, TileSet


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


def render_tile_image(context: ImageContext, region: Region, bbox: tuple[int, int, int, int]) -> Image.Image:
    left, top = bbox[:2]
    cropped = context.image.crop(bbox)
    polygon = translate_polygon(region.polygon, left, top)

    mask = Image.new("L", cropped.size, 0)
    ImageDraw.Draw(mask).polygon(
        [tuple(point) for point in polygon],
        fill=255,
    )
    cropped.putalpha(mask)
    return cropped


def extract_tile(
    context: ImageContext,
    region: Region,
    color: RGBColor = (0, 0, 0),
    *,
    analysis: RegionAnalysis | None = None,
) -> Tile:
    """Create a tile whose image is rendered on access rather than stored."""
    if analysis is None:
        analysis = RegionAnalysis(context, region)
    return Tile(
        region=region,
        bbox=analysis.bbox,
        center=analysis.center,
        color=color,
        image_source=partial(render_tile_image, context, region, analysis.bbox),
    )


def grow_tree(
    context: ImageContext,
    partitioner: Partitioner,
    stop_policy: StopPolicy,
    color_policy: RepresentativeColorPolicy | None = None,
    root: Region | None = None,
    rng: np.random.Generator | None = None,
) -> TileSet:
    tiles: list[Tile] = []

    if root is None:
        root = Region(
            polygon=np.array(
                [
                    [0, 0],
                    [context.rgba.shape[1], 0],
                    [context.rgba.shape[1], context.rgba.shape[0]],
                    [0, context.rgba.shape[0]],
                ]
            ),
            level=0,
        )

    if rng is None:
        rng = np.random.default_rng()

    def add_tile(region: Region, analysis: RegionAnalysis) -> None:
        color = analysis.stats.mean_color if color_policy is None else analysis.representative_color(color_policy)
        tiles.append(extract_tile(context, region, color, analysis=analysis))

    def visit(region: Region) -> None:
        analysis = RegionAnalysis(context, region)
        if stop_policy.should_stop(region, analysis.stats):
            add_tile(region, analysis)
            return

        children = partitioner.split(region, context, rng)
        if not children:
            add_tile(region, analysis)
            return
        for child in children:
            visit(child)

    visit(root)
    return TileSet(tuple(tiles))
