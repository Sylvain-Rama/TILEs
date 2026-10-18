import numpy as np

from ..types import ImageContext, Region


class QuadtreePartitioner:
    def split(
        self,
        region: Region,
        context: ImageContext,
        rng: np.random.Generator,
    ) -> list[Region]:
        polygon = region.polygon
        left, top = polygon.min(axis=0)
        right, bottom = polygon.max(axis=0)
        middle_x = (left + right) / 2
        middle_y = (top + bottom) / 2
        level = region.level + 1
        parent_id = id(region)
        return [
            Region(np.array([[left, top], [middle_x, top], [middle_x, middle_y], [left, middle_y]]), level, parent_id),
            Region(
                np.array([[middle_x, top], [right, top], [right, middle_y], [middle_x, middle_y]]), level, parent_id
            ),
            Region(
                np.array([[middle_x, middle_y], [right, middle_y], [right, bottom], [middle_x, bottom]]),
                level,
                parent_id,
            ),
            Region(
                np.array([[left, middle_y], [middle_x, middle_y], [middle_x, bottom], [left, bottom]]), level, parent_id
            ),
        ]
