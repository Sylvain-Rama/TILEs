from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np

from .input import prepare_image
from .partitioners.quadtree import QuadtreePartitioner
from .partitioners.random_polygons import random_polygons
from .partitioners.slicing import SlicePartitioner
from .partitioners.voronoi import VoronoiPartitioner
from .recursion import grow_tree
from .stopping import ThresholdPolicy
from .types import Region, Tile


def tiles_to_legacy_dict(tiles: Iterable[Tile], *, include_images: bool = False) -> dict[str, list[Any]]:
    tile_list = list(tiles)
    result: dict[str, list[Any]] = {
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
        result["images"] = [tile.image for tile in tile_list]
    return result


def _root_region(width: int, height: int, first_poly=None) -> Region:
    polygon = first_poly
    if polygon is None:
        polygon = np.asarray([[0, 0], [width, 0], [width, height], [0, height]], dtype=float)
    return Region(np.asarray(polygon, dtype=float))


def quadtree(img, std_thr=40, heightmap=None, max_level=6, rng=None):
    context = prepare_image(img, heightmap)
    height, width = context.rgba.shape[:2]
    tiles = grow_tree(
        context,
        _root_region(width, height),
        QuadtreePartitioner(),
        ThresholdPolicy(std_thr, max_level),
        rng=np.random.default_rng() if rng is None else rng,
    )
    return tiles_to_legacy_dict(tiles)


def recursive_slice(img, heightmap=None, std_thr=40, max_level=7, rng=None):
    context = prepare_image(img, heightmap)
    height, width = context.rgba.shape[:2]
    tiles = grow_tree(
        context,
        _root_region(width, height),
        SlicePartitioner(),
        ThresholdPolicy(std_thr, max_level, heightmap_threshold=0.2, heightmap_decay=0.01),
        rng=np.random.default_rng() if rng is None else rng,
    )
    return tiles_to_legacy_dict(tiles)


def voronoitree(
    img,
    npoints=10,
    max_level=6,
    std_thr=40,
    heightmap=None,
    first_poly=None,
    rng=None,
):
    context = prepare_image(img, heightmap)
    height, width = context.rgba.shape[:2]
    tiles = grow_tree(
        context,
        _root_region(width, height, first_poly),
        VoronoiPartitioner(npoints),
        ThresholdPolicy(std_thr, max_level),
        rng=np.random.default_rng() if rng is None else rng,
    )
    return tiles_to_legacy_dict(tiles, include_images=True)


def throw_polys(
    img,
    n_points=100,
    n_corners=3,
    distance=10,
    heightmap=None,
    rng=None,
):
    tiles = random_polygons(
        img,
        count=n_points,
        corners=n_corners,
        distance=distance,
        heightmap=heightmap,
        rng=np.random.default_rng() if rng is None else rng,
    )
    return tiles_to_legacy_dict(tiles)
