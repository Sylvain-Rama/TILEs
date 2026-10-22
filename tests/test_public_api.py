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


# --- performance: shared region analysis, summed-area tables, lazy tile images ---


def _transparent_image(width=40, height=30, seed=0):
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 256, size=(height, width, 4), dtype=np.uint8)
    pixels[..., 3] = np.where(rng.random((height, width)) < 0.3, 0, 255)
    return Image.fromarray(pixels, "RGBA")


def _reference_stats(context, left, top, right, bottom):
    """Statistics straight from a pixel slice, without summed-area tables."""
    pixels = context.rgba[top:bottom, left:right]
    visible = pixels[:, :, 3] != 0
    selected = pixels[visible, :3]
    return selected, context.heightmap[top:bottom, left:right][visible]


@pytest.mark.parametrize("seed", range(5))
def test_summed_area_stats_match_pixel_slice(seed):
    from tiles.input import prepare_image
    from tiles.statistics import RegionAnalysis

    rng = np.random.default_rng(seed)
    context = prepare_image(
        _transparent_image(seed=seed), Image.fromarray(rng.integers(0, 256, (30, 40), dtype=np.uint8))
    )
    assert context.integral is not None

    for _ in range(50):
        left, right = sorted(rng.integers(0, 41, size=2).tolist())
        top, bottom = sorted(rng.integers(0, 31, size=2).tolist())
        polygon = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=float)
        stats = RegionAnalysis(context, Region(polygon)).stats

        selected, heights = _reference_stats(context, left, top, right, bottom)
        assert stats.visible_pixel_count == len(selected)
        if len(selected) == 0:
            assert (stats.standard_deviation, stats.mean_color, stats.heightmap_max) == (0.0, (0, 0, 0), 0.0)
            continue
        assert stats.mean_color == tuple(np.mean(selected, axis=0).astype(np.uint8).tolist())
        assert stats.standard_deviation == pytest.approx(float(np.max(np.std(selected, axis=0))), abs=1e-6)
        assert stats.heightmap_max == float(np.max(heights))


def test_constant_region_has_zero_deviation():
    from tiles.input import prepare_image
    from tiles.statistics import RegionAnalysis

    context = prepare_image(Image.new("RGB", (64, 64), (123, 45, 67)))
    polygon = np.array([[3, 5], [60, 5], [60, 50], [3, 50]], dtype=float)
    assert RegionAnalysis(context, Region(polygon)).stats.standard_deviation == 0.0


@pytest.mark.parametrize(
    "polygon, expected",
    [
        ([[0, 0], [4, 0], [4, 3], [0, 3]], (0, 0, 4, 3)),
        ([[0, 0], [4, 0], [4, 3], [0, 3], [0, 0]], (0, 0, 4, 3)),
        ([[0, 3], [0, 0], [4, 0], [4, 3]], (0, 0, 4, 3)),
        ([[0, 0], [4, 3], [4, 0], [0, 3]], None),  # bow-tie
        ([[0, 0], [4, 0], [2, 3]], None),
        ([[0, 0], [4, 0], [4, 0], [0, 0]], None),  # degenerate
        ([[0, 0], [4, 1], [4, 3], [0, 3]], None),  # not axis-aligned
    ],
)
def test_rectangle_bounds(polygon, expected):
    from tiles.geometry import rectangle_bounds

    assert rectangle_bounds(np.array(polygon, dtype=float)) == expected


@pytest.mark.parametrize(
    "run",
    [
        lambda: TILEs.quadtree(_transparent_image(64, 64), max_level=4, std_thr=0),
        lambda: TILEs.recursive_slice(_transparent_image(64, 64), max_level=5, std_thr=0, rng=np.random.default_rng(2)),
    ],
    ids=["quadtree", "recursive-slice"],
)
def test_summed_area_tables_do_not_change_results(monkeypatch, run):
    from tiles import integral

    with_tables = run()
    monkeypatch.setattr(integral, "INTEGRAL_MAX_PIXELS", 0)
    without_tables = run()

    assert with_tables["colors"] == without_tables["colors"]
    assert all(np.array_equal(a, b) for a, b in zip(with_tables["polys"], without_tables["polys"], strict=True))


def test_integral_image_is_skipped_for_huge_images(monkeypatch):
    from tiles import integral
    from tiles.input import prepare_image

    monkeypatch.setattr(integral, "INTEGRAL_MAX_PIXELS", 10)
    assert prepare_image(Image.new("RGB", (8, 8))).integral is None


@pytest.mark.parametrize("policy", [None, MedianRGBColorPolicy()], ids=["default", "median"])
def test_region_mask_is_built_once_per_polygon_region(monkeypatch, policy):
    from tiles import statistics

    masks = []
    original_mask = statistics._region_mask
    monkeypatch.setattr(statistics, "_region_mask", lambda region, bbox: masks.append(1) or original_mask(region, bbox))

    children = []
    original_split = VoronoiPartitioner.split

    def counting_split(self, region, context, rng):
        result = original_split(self, region, context, rng)
        children.extend(result)
        return result

    monkeypatch.setattr(VoronoiPartitioner, "split", counting_split)

    result = TILEs.voronoitree(
        _transparent_image(64, 64),
        npoints=4,
        max_level=2,
        std_thr=0,
        rng=np.random.default_rng(3),
        color_policy=policy,
    )
    _ = list(result["images"])

    # The root is a rectangle; every Voronoi cell is a polygon and costs exactly one mask,
    # shared by its statistics, its colour policy and its tile image.
    assert len(masks) == len(children)


def test_axis_aligned_partitioners_never_build_a_mask(monkeypatch):
    from tiles import statistics

    monkeypatch.setattr(statistics, "_region_mask", lambda *a: pytest.fail("mask built for a rectangle"))
    image = _transparent_image(64, 64)
    TILEs.quadtree(image, max_level=3, std_thr=0, color_policy=MedianRGBColorPolicy())
    TILEs.recursive_slice(image, max_level=4, std_thr=0)


def _voronoi(image, **kwargs):
    return TILEs.voronoitree(image, npoints=4, max_level=2, std_thr=0, rng=np.random.default_rng(3), **kwargs)


def test_tile_images_are_rendered_lazily(monkeypatch):
    from tiles import recursion

    renders = []
    original = recursion.render_tile_image
    monkeypatch.setattr(recursion, "render_tile_image", lambda *a: renders.append(1) or original(*a))

    result = _voronoi(_transparent_image(64, 64))
    assert renders == []

    images = result["images"]
    count = len(result["x"])
    assert len(images) == count
    assert renders == []

    assert isinstance(images[0], Image.Image)
    assert len(renders) == 1
    assert len(images[1:4]) == 3
    assert len(list(images)) == count


def test_lazy_images_match_cropped_reference():
    image = _transparent_image(64, 64)
    result = _voronoi(image)
    rgba = np.asarray(image)

    for left, top, width, height, tile in zip(
        result["left"], result["top"], result["width"], result["height"], result["images"], strict=True
    ):
        tile_pixels = np.asarray(tile)
        assert tile_pixels.shape[:2] == (height, width)
        assert np.array_equal(tile_pixels[..., :3], rgba[top : top + height, left : left + width, :3])
