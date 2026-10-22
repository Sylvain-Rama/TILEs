import numpy as np
import pytest
from PIL import Image
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

import TILEs
from tiles import MedianRGBColorPolicy, Region
from tiles.partitioners.voronoi import VoronoiPartitioner


@pytest.fixture
def image():
    return Image.new("RGB", (16, 12), (0, 0, 0))


@pytest.mark.parametrize(
    "run",
    [
        pytest.param(
            lambda image: TILEs.quadtree(
                image,
                max_level=1,
                std_thr=0,
                color_policy=MedianRGBColorPolicy(),
            ),
            id="quadtree",
        ),
        pytest.param(
            lambda image: TILEs.recursive_slice(
                image,
                max_level=1,
                std_thr=0,
                color_policy=MedianRGBColorPolicy(),
            ),
            id="recursive-slice",
        ),
        pytest.param(
            lambda image: TILEs.voronoitree(
                image,
                npoints=5,
                max_level=1,
                std_thr=0,
                rng=np.random.default_rng(1),
                color_policy=MedianRGBColorPolicy(),
            ),
            id="voronoi-tree",
        ),
        pytest.param(
            lambda image: TILEs.throw_polys(
                image,
                n_points=3,
                rng=np.random.default_rng(1),
                color_policy=MedianRGBColorPolicy(),
            ),
            id="random-polygons",
        ),
    ],
)
def test_partitioners_return_aligned_records(image, run):
    result = run(image)
    lengths = {len(values) for values in result.values()}
    assert len(lengths) == 1


def test_dither_returns_image(image):
    result = TILEs.dither(image)
    assert isinstance(result, Image.Image)
    assert result.size == image.size


def test_quadtree_accepts_representative_color_policy():
    pixels = np.array(
        [
            [[0, 0, 0], [10, 10, 10]],
            [[20, 20, 20], [100, 100, 100]],
        ],
        dtype=np.uint8,
    )
    image = Image.fromarray(pixels)

    result = TILEs.quadtree(
        image,
        max_level=0,
        color_policy=MedianRGBColorPolicy(),
    )

    assert result["colors"] == [(15, 15, 15)]


def test_default_quadtree_color_remains_legacy_mean():
    pixels = np.array(
        [
            [[0, 0, 0], [10, 10, 10]],
            [[20, 20, 20], [100, 100, 100]],
        ],
        dtype=np.uint8,
    )
    image = Image.fromarray(pixels)

    result = TILEs.quadtree(image, max_level=0)

    assert result["colors"] == [(32, 32, 32)]


SQUARE = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=float)
U_SHAPE = np.array([[0, 0], [100, 0], [100, 100], [70, 100], [70, 30], [30, 30], [30, 100], [0, 100]], dtype=float)


def _split(polygon, npoints, seed):
    return VoronoiPartitioner(npoints).split(Region(polygon), None, np.random.default_rng(seed))


def test_voronoi_sampling_reaches_closing_edge():
    samples = VoronoiPartitioner._sample_points(SQUARE, 2000, np.random.default_rng(0))
    assert ((samples[:, 0] < 25) & (samples[:, 1] > 40) & (samples[:, 1] < 60)).any()


@pytest.mark.parametrize("polygon", [SQUARE, U_SHAPE], ids=["convex", "non-convex"])
def test_voronoi_samples_stay_inside_polygon(polygon):
    shape = Polygon(polygon)
    samples = VoronoiPartitioner._sample_points(polygon, 200, np.random.default_rng(0))
    assert all(shape.covers(Point(sample)) for sample in samples)


@pytest.mark.parametrize("polygon", [SQUARE, U_SHAPE], ids=["convex", "non-convex"])
@pytest.mark.parametrize("npoints", [3, 4, 10])
@pytest.mark.parametrize("seed", range(20))
def test_voronoi_children_tile_parent_exactly(polygon, npoints, seed):
    parent = Polygon(polygon)
    children = [Polygon(child.polygon) for child in _split(polygon, npoints, seed)]
    assert sum(child.area for child in children) == pytest.approx(parent.area)
    assert unary_union(children).area == pytest.approx(parent.area)


@pytest.mark.parametrize(
    "polygon",
    [
        np.array([[0, 0], [10, 0], [20, 0], [30, 0]], dtype=float),
        np.array([[5, 5], [5, 5], [5, 5]], dtype=float),
    ],
    ids=["zero-area-line", "single-point"],
)
def test_voronoi_degenerate_region_is_left_unsplit(polygon):
    assert _split(polygon, 4, 0) == []
