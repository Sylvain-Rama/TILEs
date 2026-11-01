from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import shapely
from shapely.geometry import Point, Polygon
from shapely.geometry.base import BaseGeometry

from ..geometry import as_polygon
from ..types import ImageContext, Region


def _polygon_parts(geometry: BaseGeometry) -> Iterator[Polygon]:
    if geometry.is_empty:
        return
    if isinstance(geometry, Polygon):
        if geometry.area > 0:
            yield geometry
    elif hasattr(geometry, "geoms"):
        for part in geometry.geoms:
            yield from _polygon_parts(part)


class VoronoiPartitioner:
    def __init__(self, npoints: int = 10) -> None:
        if npoints < 3:
            raise ValueError("npoints must be at least 3")
        self.npoints = npoints

    @staticmethod
    def _sample_points(polygon: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
        shape = as_polygon(polygon)
        center = polygon.mean(axis=0)
        if shape.is_empty or shape.area <= 0:
            return np.repeat(center[None, :], count, axis=0)

        # The vertex mean can fall outside a non-convex polygon.
        if not shape.covers(Point(center)):
            center = np.asarray(shape.representative_point().coords[0])

        # Triangulating the polygon itself keeps samples inside non-convex shapes.
        triangles = [
            np.asarray(triangle.exterior.coords)[:3]
            for triangle in shapely.constrained_delaunay_triangles(shape).geoms
            if triangle.area > 0
        ]
        areas = np.asarray(
            [
                0.5
                * abs((second[0] - first[0]) * (third[1] - first[1]) - (third[0] - first[0]) * (second[1] - first[1]))
                for first, second, third in triangles
            ]
        )
        probabilities = areas / areas.sum()
        samples = [center]
        for index in rng.choice(len(triangles), size=count - 1, p=probabilities):
            first, second, third = triangles[index]
            root = np.sqrt(rng.uniform())
            weight = rng.uniform()
            samples.append((1 - root) * first + root * (1 - weight) * second + root * weight * third)
        return np.asarray(samples)

    def split(
        self,
        region: Region,
        context: ImageContext,
        rng: np.random.Generator,
    ) -> list[Region]:
        clipping_polygon = as_polygon(region.polygon)
        if clipping_polygon.is_empty or clipping_polygon.area <= 0:
            return []

        count = max(3, self.npoints - region.level)
        points = np.unique(self._sample_points(region.polygon, count, rng), axis=0)
        if len(points) < 2:
            return []

        # Cells are bounded to the clipping polygon's envelope, so they always cover it.
        cells = shapely.voronoi_polygons(shapely.MultiPoint(points), extend_to=clipping_polygon)

        children: list[Region] = []
        for cell in cells.geoms:
            # Clipping a cell with a non-convex polygon can yield several pieces.
            for part in _polygon_parts(cell.intersection(clipping_polygon)):
                children.append(
                    Region(
                        np.asarray(part.exterior.coords)[:-1],
                        region.level + 1,
                        id(region),
                    )
                )
        return children
