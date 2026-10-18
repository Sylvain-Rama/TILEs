from typing import Protocol

import numpy as np

from .extraction import extract_tile
from .statistics import region_stats
from .types import ImageContext, Region, Tile, TileSet


class Partitioner(Protocol):
    def split(
        self,
        region: Region,
        context: ImageContext,
        rng: np.random.Generator,
    ) -> list[Region]: ...


class StopPolicy(Protocol):
    def should_stop(self, region: Region, stats) -> bool: ...


def grow_tree(
    context: ImageContext,
    root: Region,
    partitioner: Partitioner,
    stop_policy: StopPolicy,
    *,
    rng: np.random.Generator,
) -> TileSet:
    tiles: list[Tile] = []

    def visit(region: Region) -> None:
        stats = region_stats(context, region)
        if stop_policy.should_stop(region, stats):
            tile = extract_tile(context, region)
            tile.color = stats.mean_color
            tiles.append(tile)
            return

        children = partitioner.split(region, context, rng)
        if not children:
            tile = extract_tile(context, region)
            tile.color = stats.mean_color
            tiles.append(tile)
            return
        for child in children:
            visit(child)

    visit(root)
    return TileSet(tuple(tiles))
