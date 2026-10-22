from __future__ import annotations

import numpy as np

from ..input import prepare_image
from ..protocols import RepresentativeColorPolicy
from ..recursion import extract_tile
from ..statistics import RegionAnalysis
from ..types import ImageContext, Region, TileSet


def generate_random_polygons(
    context: ImageContext,
    *,
    count: int = 100,
    corners: int = 3,
    distance: float = 10,
    rng: np.random.Generator,
    color_policy: RepresentativeColorPolicy | None = None,
) -> TileSet:
    if count < 0:
        raise ValueError("count must be non-negative")
    if corners < 3:
        raise ValueError("corners must be at least 3")
    if distance < 0:
        raise ValueError("distance must be non-negative")

    height, width = context.rgba.shape[:2]
    probabilities = context.heightmap.reshape(-1).astype(float)
    total = probabilities.sum()
    if total <= 0:
        probabilities = np.full(width * height, 1 / (width * height))
    else:
        probabilities /= total

    indices = rng.choice(width * height, size=count, p=probabilities)
    tiles = []
    for index in indices:
        x, y = int(index % width), int(index // width)
        angles = rng.uniform(0, 2 * np.pi, size=corners)
        radii = rng.uniform(0, distance, size=corners)
        polygon = np.column_stack(
            (
                np.clip(x + radii * np.cos(angles), 0, width - 1),
                np.clip(y + radii * np.sin(angles), 0, height - 1),
            )
        )
        region = Region(polygon)
        analysis = RegionAnalysis(context, region)
        color = analysis.stats.mean_color if color_policy is None else analysis.representative_color(color_policy)
        tile = extract_tile(context, region, color, analysis=analysis)
        tiles.append(tile)
    return TileSet(tuple(tiles))


def random_polygons(
    image,
    *,
    count: int = 100,
    corners: int = 3,
    distance: float = 10,
    heightmap=None,
    rng: np.random.Generator | None = None,
    color_policy: RepresentativeColorPolicy | None = None,
) -> TileSet:
    context = prepare_image(image, heightmap)
    return generate_random_polygons(
        context,
        count=count,
        corners=corners,
        distance=distance,
        rng=np.random.default_rng() if rng is None else rng,
        color_policy=color_policy,
    )
