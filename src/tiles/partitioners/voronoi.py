from __future__ import annotations

import numpy as np
import scipy.spatial
from shapely.geometry import Polygon

from ..geometry import as_polygon
from ..types import ImageContext, Region


class VoronoiPartitioner:
    def __init__(self, npoints: int = 10) -> None:
        if npoints < 3:
            raise ValueError("npoints must be at least 3")
        self.npoints = npoints

    @staticmethod
    def _bounded_regions(points: np.ndarray) -> tuple[list[list[int]], np.ndarray]:
        voronoi = scipy.spatial.Voronoi(points)
        regions: list[list[int]] = []
        vertices = voronoi.vertices.tolist()
        center = voronoi.points.mean(axis=0)
        radius = np.ptp(voronoi.points, axis=0).max() * 2
        ridges: dict[int, list[tuple[int, int, int]]] = {}
        for (point_a, point_b), (vertex_a, vertex_b) in zip(voronoi.ridge_points, voronoi.ridge_vertices):
            ridges.setdefault(point_a, []).append((point_b, vertex_a, vertex_b))
            ridges.setdefault(point_b, []).append((point_a, vertex_a, vertex_b))

        for point_index, region_index in enumerate(voronoi.point_region):
            region_vertices = voronoi.regions[region_index]
            if all(vertex >= 0 for vertex in region_vertices):
                regions.append(region_vertices)
                continue

            new_region = [vertex for vertex in region_vertices if vertex >= 0]
            for other_index, vertex_a, vertex_b in ridges[point_index]:
                if vertex_b < 0:
                    vertex_a, vertex_b = vertex_b, vertex_a
                if vertex_a >= 0:
                    continue
                tangent = voronoi.points[other_index] - voronoi.points[point_index]
                tangent /= np.linalg.norm(tangent)
                normal = np.array([-tangent[1], tangent[0]])
                midpoint = voronoi.points[[point_index, other_index]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, normal)) * normal
                far_point = voronoi.vertices[vertex_b] + direction * radius
                new_region.append(len(vertices))
                vertices.append(far_point.tolist())

            coordinates = np.asarray([vertices[vertex] for vertex in new_region])
            centroid = coordinates.mean(axis=0)
            angles = np.arctan2(coordinates[:, 1] - centroid[1], coordinates[:, 0] - centroid[0])
            regions.append(np.asarray(new_region)[np.argsort(angles)].tolist())
        return regions, np.asarray(vertices)

    @staticmethod
    def _sample_points(polygon: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
        center = polygon.mean(axis=0)
        triangles = [(center, polygon[index], polygon[index + 1]) for index in range(len(polygon) - 1)]
        areas = np.asarray(
            [
                0.5
                * abs((second[0] - first[0]) * (third[1] - first[1]) - (third[0] - first[0]) * (second[1] - first[1]))
                for first, second, third in triangles
            ]
        )
        if areas.sum() <= 0:
            return np.repeat(center[None, :], count, axis=0)
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
        count = max(3, self.npoints - region.level)
        points = self._sample_points(region.polygon, count, rng)
        regions, vertices = self._bounded_regions(points)
        clipping_polygon = as_polygon(region.polygon)
        children: list[Region] = []
        for region_vertices in regions:
            candidate = Polygon(vertices[region_vertices]).intersection(clipping_polygon)
            if candidate.is_empty or candidate.geom_type != "Polygon" or candidate.area <= 0:
                continue
            children.append(
                Region(
                    np.asarray(candidate.exterior.coords)[:-1],
                    region.level + 1,
                    id(region),
                )
            )
        return children
