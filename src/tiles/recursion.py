from functools import partial

import numpy as np
from PIL import Image, ImageDraw

from .geometry import root_region, translate_polygon
from .protocols import Partitioner, RepresentativeColorPolicy, StopPolicy
from .statistics import RegionAnalysis
from .types import ImageContext, Region, RGBColor, Tile, TileSet


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
    seed: float | None = None,
) -> TileSet:
    tiles: list[Tile] = []

    if root is None:
        root = root_region(context.size[0], context.size[1])

    if seed is None:
        seed = np.random.SeedSequence().entropy

    rng = np.random.default_rng(seed)

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
    return TileSet(tuple(tiles), seed=seed)
