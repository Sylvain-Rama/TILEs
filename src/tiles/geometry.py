from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon


def polygon_bbox(polygon: np.ndarray, width: int, height: int) -> tuple[int, int, int, int]:
    xs = polygon[:, 0]
    ys = polygon[:, 1]
    left = max(0, int(np.floor(xs.min())))
    top = max(0, int(np.floor(ys.min())))
    right = min(width, int(np.ceil(xs.max())))
    bottom = min(height, int(np.ceil(ys.max())))
    return left, top, right, bottom


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
