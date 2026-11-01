import numpy as np

from ..types import ImageContext, Region


class SlicePartitioner:
    """Cut a region in two across its longer side at a random position.

    ``level_step`` is how much a cut adds to the recursion level. Two cuts are needed to halve both
    dimensions, so a step of 0.5 makes a level comparable to a quadtree level for level-based stop policies.
    """

    def __init__(self, level_step: float = 0.5) -> None:
        if level_step <= 0:
            raise ValueError("level_step must be positive")
        self.level_step = level_step

    def split(
        self,
        region: Region,
        context: ImageContext,
        rng: np.random.Generator,
    ) -> list[Region]:
        polygon = region.polygon
        left, top = polygon.min(axis=0)
        right, bottom = polygon.max(axis=0)
        level = region.level + self.level_step
        if bottom - top >= right - left:
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
