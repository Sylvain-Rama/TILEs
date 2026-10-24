from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from .geometry import root_region
from .partitioners.quadtree import QuadtreePartitioner
from .partitioners.random_polygons import random_polygons
from .partitioners.slicing import SlicePartitioner
from .partitioners.voronoi import VoronoiPartitioner
from .protocols import RepresentativeColorPolicy
from .recursion import ThresholdPolicy, grow_tree
from .types import ImageContext, Tile


class LazyTileImages(Sequence):
    """Sequence of tile images, each rendered when accessed instead of all being held in memory."""

    def __init__(self, tiles: Iterable[Tile]) -> None:
        self._tiles = tuple(tiles)

    def __len__(self) -> int:
        return len(self._tiles)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [tile.image for tile in self._tiles[index]]
        return self._tiles[index].image


def tiles_to_legacy_dict(tiles: Iterable[Tile], *, include_images: bool = False) -> dict[str, Any]:
    tile_list = list(tiles)
    result: dict[str, Any] = {
        "top": [tile.bbox[1] for tile in tile_list],
        "left": [tile.bbox[0] for tile in tile_list],
        "x": [tile.center[0] for tile in tile_list],
        "y": [tile.center[1] for tile in tile_list],
        "width": [tile.bbox[2] - tile.bbox[0] for tile in tile_list],
        "height": [tile.bbox[3] - tile.bbox[1] for tile in tile_list],
        "colors": [tile.color for tile in tile_list],
        "polys": [tile.region.polygon for tile in tile_list],
        "shifted_polys": [tile.region.polygon - tile.region.polygon.min(axis=0) for tile in tile_list],
        "level": [tile.region.level for tile in tile_list],
    }
    if include_images:
        result["images"] = LazyTileImages(tile_list)
    return result


def quadtree(
    context: ImageContext,
    std_thr=40,
    max_level=6,
    *,
    color_policy: RepresentativeColorPolicy | None = None,
):
    height, width = context.rgba.shape[:2]
    tiles = grow_tree(
        context,
        partitioner=QuadtreePartitioner(),
        stop_policy=ThresholdPolicy(std_thr, max_level),
        color_policy=color_policy,
        root=root_region(width, height),
    )
    return tiles_to_legacy_dict(tiles)


def recursive_slice(
    context: ImageContext,
    std_thr=40,
    max_level=7,
    color_policy: RepresentativeColorPolicy | None = None,
):
    height, width = context.rgba.shape[:2]
    tiles = grow_tree(
        context,
        partitioner=SlicePartitioner(),
        stop_policy=ThresholdPolicy(std_thr, max_level, heightmap_threshold=0.2, heightmap_decay=0.01),
        color_policy=color_policy,
        root=root_region(width, height),
    )
    return tiles_to_legacy_dict(tiles)


def voronoitree(
    context: ImageContext,
    npoints=10,
    max_level=6,
    std_thr=40,
    first_poly=None,
    color_policy: RepresentativeColorPolicy | None = None,
):
    height, width = context.rgba.shape[:2]
    tiles = grow_tree(
        context,
        partitioner=VoronoiPartitioner(npoints),
        stop_policy=ThresholdPolicy(std_thr, max_level),
        color_policy=color_policy,
        root=root_region(width, height, first_poly),
    )
    return tiles_to_legacy_dict(tiles, include_images=True)


def throw_polys(
    context: ImageContext,
    n_points=100,
    n_corners=3,
    distance=10,
    color_policy: RepresentativeColorPolicy | None = None,
):
    rng = np.random.default_rng()
    tiles = random_polygons(
        context,
        count=n_points,
        corners=n_corners,
        distance=distance,
        rng=rng,
        color_policy=color_policy,
    )
    return tiles_to_legacy_dict(tiles)
