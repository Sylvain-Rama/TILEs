from __future__ import annotations

import math

import numpy as np
from shapely.geometry import Polygon

from .types import Region


def polygon_bbox(polygon: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    xs = polygon[:, 0]
    ys = polygon[:, 1]
    left = max(0, int(np.floor(xs.min())))
    top = max(0, int(np.floor(ys.min())))
    right = min(width, int(np.ceil(xs.max())))
    bottom = min(height, int(np.ceil(ys.max())))
    return left, top, right, bottom


def rectangle_bounds(polygon: np.ndarray) -> tuple[float, float, float, float] | None:
    """Return (left, top, right, bottom) if the polygon is a non-degenerate axis-aligned rectangle."""
    if len(polygon) not in (4, 5):
        return None
    points = np.asarray(polygon).tolist()
    if len(points) == 5:
        if points[0] != points[4]:
            return None
        points = points[:4]

    # Plain floats: this runs once per tree node, where numpy call overhead dominates.
    (x0, y0), (x1, y1), (x2, y2), (x3, y3) = points
    vertical_first = x0 == x1 and y1 == y2 and x2 == x3 and y3 == y0
    horizontal_first = y0 == y1 and x1 == x2 and y2 == y3 and x3 == x0
    if not (vertical_first or horizontal_first):
        return None

    left, right = min(x0, x1, x2, x3), max(x0, x1, x2, x3)
    top, bottom = min(y0, y1, y2, y3), max(y0, y1, y2, y3)
    if left == right or top == bottom:
        return None
    return left, top, right, bottom


def bounds_bbox(bounds: tuple[float, float, float, float], width: int, height: int) -> tuple[int, int, int, int]:
    """Same result as polygon_bbox for a polygon with the given (left, top, right, bottom) bounds."""
    left, top, right, bottom = bounds
    return (
        max(0, math.floor(left)),
        max(0, math.floor(top)),
        min(width, math.ceil(right)),
        min(height, math.ceil(bottom)),
    )


def polygon_center(polygon: np.ndarray) -> tuple[float, float]:
    return (
        float((polygon[:, 0].min() + polygon[:, 0].max()) / 2),
        float((polygon[:, 1].min() + polygon[:, 1].max()) / 2),
    )


def translate_polygon(polygon: np.ndarray, left: int, top: int) -> np.ndarray:
    return polygon - np.array([left, top])


def as_polygon(points: np.ndarray) -> Polygon:
    polygon = Polygon(points)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return polygon


def root_region(width: int, height: int, first_poly=None) -> Region:
    polygon = first_poly
    if polygon is None:
        polygon = np.asarray([[0, 0], [width, 0], [width, height], [0, height]], dtype=float)
    return Region(np.asarray(polygon, dtype=float))
