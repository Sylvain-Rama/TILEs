import numpy as np

from ..types import ImageContext, Region


class SlicePartitioner:
    def split(
        self,
        region: Region,
        context: ImageContext,
        rng: np.random.Generator,
    ) -> list[Region]:
        polygon = region.polygon
        left, top = polygon.min(axis=0)
        right, bottom = polygon.max(axis=0)
        level = region.level + 1
        direction = region.level % 2
        if direction == 0:
            split = rng.uniform(top, bottom)
            if split <= top or split >= bottom:
                return []
            return [
                Region(np.array([[left, top], [right, top], [right, split], [left, split]]), level, id(region)),
                Region(np.array([[left, split], [right, split], [right, bottom], [left, bottom]]), level, id(region)),
            ]

        split = rng.uniform(left, right)
        if split <= left or split >= right:
            return []
        return [
            Region(np.array([[left, top], [split, top], [split, bottom], [left, bottom]]), level, id(region)),
            Region(np.array([[split, top], [right, top], [right, bottom], [split, bottom]]), level, id(region)),
        ]
