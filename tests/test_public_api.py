import tempfile
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest
from PIL import Image
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

import TILEs
from tiles import (
    DominantColor,
    MeanRGBColor,
    MedianRGBColor,
    MostSaturatedColor,
    OKLabChromaColor,
    Region,
    Tile,
    TileSet,
    load_images,
    recompute_tile_colors,
)
from tiles.partitioners.voronoi import VoronoiPartitioner

matplotlib.use("Agg")


def prepare_image(image, heightmap=None):
    """Build an ImageContext from PIL images through the public file loader."""
    with tempfile.TemporaryDirectory() as folder:
        image_path = Path(folder) / "image.png"
        image.save(image_path)
        heightmap_path = None
        if heightmap is not None:
            heightmap_path = Path(folder) / "heightmap.png"
            heightmap.save(heightmap_path)
        return load_images(str(image_path), None if heightmap_path is None else str(heightmap_path))


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
                color_policy=MedianRGBColor(),
            ),
            id="quadtree",
        ),
        pytest.param(
            lambda image: TILEs.recursive_slice(
                image,
                max_level=1,
                std_thr=0,
                color_policy=MedianRGBColor(),
            ),
            id="recursive-slice",
        ),
        pytest.param(
            lambda image: TILEs.voronoitree(
                image,
                npoints=5,
                max_level=1,
                std_thr=0,
                color_policy=MedianRGBColor(),
            ),
            id="voronoi-tree",
        ),
        pytest.param(
            lambda image: TILEs.throw_polys(
                image,
                n_points=3,
                color_policy=MedianRGBColor(),
            ),
            id="random-polygons",
        ),
    ],
)
def test_partitioners_return_aligned_records(image, run):
    result = run(prepare_image(image))
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
    image = prepare_image(Image.fromarray(pixels))

    result = TILEs.quadtree(
        image,
        max_level=0,
        color_policy=MedianRGBColor(),
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
    image = prepare_image(Image.fromarray(pixels))

    result = TILEs.quadtree(image, max_level=0)

    assert result["colors"] == [(32, 32, 32)]


def test_recompute_tile_colors_preserves_tiles_and_applies_policy():
    pixels = np.array(
        [
            [[0, 0, 0], [10, 10, 10]],
            [[20, 20, 20], [100, 100, 100]],
        ],
        dtype=np.uint8,
    )
    context = prepare_image(Image.fromarray(pixels))
    region = Region(np.array([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float))
    original = Tile(region, (0, 0, 2, 2), (1.0, 1.0), (255, 0, 0))
    tile_set = TileSet((original,), seed=42)

    result = recompute_tile_colors(tile_set, context, MedianRGBColor())

    assert result is not tile_set
    assert result.seed == 42
    assert result.tiles[0] is not original
    assert result.tiles[0].color == (15, 15, 15)
    assert result.tiles[0].region is original.region
    assert result.tiles[0].bbox == original.bbox
    assert result.tiles[0].center == original.center
    assert result.tiles[0].image_source is original.image_source
    assert original.color == (255, 0, 0)


def test_tileset_can_be_saved_as_png_and_svg(tmp_path):
    from tiles import save_tileset

    region = Region(np.array([[0, 0], [4, 0], [4, 2], [0, 2]], dtype=float))
    tile_set = TileSet((Tile(region, (0, 0, 4, 2), (2.0, 1.0), (20, 40, 60)),))

    for suffix in ("png", "svg"):
        output = tmp_path / f"tiles.{suffix}"
        save_tileset(tile_set, output, canvas_size=(4, 2), figsize=(2, 1), dpi=100)
        assert output.stat().st_size > 0


def test_tileset_width_preserves_canvas_ratio():
    from tiles import tileset_to_figure

    region = Region(np.array([[0, 0], [4, 0], [4, 2], [0, 2]], dtype=float))
    tile_set = TileSet((Tile(region, (0, 0, 4, 2), (2.0, 1.0), (20, 40, 60)),))

    figure = tileset_to_figure(tile_set, canvas_size=(4, 2), width=8)

    assert figure.get_size_inches().tolist() == [8.0, 4.0]
    import matplotlib.pyplot as plt

    plt.close(figure)


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
        lambda: TILEs.quadtree(prepare_image(_transparent_image(64, 64)), max_level=4, std_thr=0),
        lambda: TILEs.recursive_slice(prepare_image(_transparent_image(64, 64)), max_level=5, std_thr=0),
    ],
    ids=["quadtree", "recursive-slice"],
)
def test_summed_area_tables_do_not_change_results(monkeypatch, run):
    from tiles import integral

    # The legacy functions no longer take an rng, so pin the seed grow_tree draws.
    monkeypatch.setattr(np.random, "SeedSequence", lambda: SimpleNamespace(entropy=2))

    with_tables = run()
    monkeypatch.setattr(integral, "INTEGRAL_MAX_PIXELS", 0)
    without_tables = run()

    assert with_tables["colors"] == without_tables["colors"]
    assert all(np.array_equal(a, b) for a, b in zip(with_tables["polys"], without_tables["polys"], strict=True))


def test_integral_image_is_skipped_for_huge_images(monkeypatch):
    from tiles import integral

    monkeypatch.setattr(integral, "INTEGRAL_MAX_PIXELS", 10)
    assert prepare_image(Image.new("RGB", (8, 8))).integral is None


@pytest.mark.parametrize("policy", [None, MedianRGBColor()], ids=["default", "median"])
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
        prepare_image(_transparent_image(64, 64)),
        npoints=4,
        max_level=2,
        std_thr=0,
        color_policy=policy,
    )
    _ = list(result["images"])

    # The root is a rectangle; every Voronoi cell is a polygon and costs exactly one mask,
    # shared by its statistics, its colour policy and its tile image.
    assert len(masks) == len(children)


def test_axis_aligned_partitioners_never_build_a_mask(monkeypatch):
    from tiles import statistics

    monkeypatch.setattr(statistics, "_region_mask", lambda *a: pytest.fail("mask built for a rectangle"))
    image = prepare_image(_transparent_image(64, 64))
    TILEs.quadtree(image, max_level=3, std_thr=0, color_policy=MedianRGBColor())
    TILEs.recursive_slice(image, max_level=4, std_thr=0)


def _voronoi(image, **kwargs):
    return TILEs.voronoitree(image, npoints=4, max_level=2, std_thr=0, **kwargs)


def test_tile_images_are_rendered_lazily(monkeypatch):
    from tiles import recursion

    renders = []
    original = recursion.render_tile_image
    monkeypatch.setattr(recursion, "render_tile_image", lambda *a: renders.append(1) or original(*a))

    result = _voronoi(prepare_image(_transparent_image(64, 64)))
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
    result = _voronoi(prepare_image(image))
    rgba = np.asarray(image)

    for left, top, width, height, tile in zip(
        result["left"], result["top"], result["width"], result["height"], result["images"], strict=True
    ):
        tile_pixels = np.asarray(tile)
        assert tile_pixels.shape[:2] == (height, width)
        assert np.array_equal(tile_pixels[..., :3], rgba[top : top + height, left : left + width, :3])


# --- vivid colour policies ---


def _pixels(*groups):
    return np.array([color for color, count in groups for _ in range(count)], dtype=np.uint8)


@pytest.mark.parametrize("policy", [OKLabChromaColor(), DominantColor()], ids=["oklab", "dominant"])
def test_vivid_policies_return_the_colour_of_a_uniform_region(policy):
    assert policy.representative_color(_pixels(((10, 200, 30), 7))) == (10, 200, 30)


@pytest.mark.parametrize("policy", [OKLabChromaColor(), DominantColor()], ids=["oklab", "dominant"])
def test_vivid_policies_reject_empty_input(policy):
    with pytest.raises(ValueError):
        policy.representative_color(np.empty((0, 3), dtype=np.uint8))


def test_dominant_color_ignores_minority_hues():
    pixels = _pixels(((220, 30, 30), 60), ((20, 40, 230), 40))
    assert DominantColor().representative_color(pixels) == (220, 30, 30)


def test_dominant_color_averages_within_the_winning_bin():
    pixels = _pixels(((200, 10, 10), 5), ((202, 12, 14), 5), ((0, 0, 255), 3))
    assert DominantColor().representative_color(pixels) == (201, 11, 12)


@pytest.mark.parametrize("bits", [0, 9])
def test_dominant_color_validates_bits(bits):
    with pytest.raises(ValueError):
        DominantColor(bits=bits)


def _oklab_chroma(rgb):
    from tiles.color_policies import srgb_to_oklab

    lab = srgb_to_oklab(np.array(rgb, dtype=float))
    return float(np.hypot(lab[1], lab[2]))


MIXED_HUES = [
    [((255, 0, 0), 50), ((0, 0, 255), 50)],
    [((255, 0, 0), 50), ((0, 255, 0), 50)],
    [((220, 30, 30), 60), ((128, 128, 128), 40)],
]


@pytest.mark.parametrize("groups", MIXED_HUES)
def test_oklab_chroma_policy_vividness_is_monotonic(groups):
    pixels = _pixels(*groups)
    chromas = [_oklab_chroma(OKLabChromaColor(vividness=v).representative_color(pixels)) for v in (0, 0.5, 1)]
    assert chromas == sorted(chromas)


def test_oklab_chroma_policy_restores_chroma_of_mixed_hues():
    pixels = _pixels(((255, 0, 0), 50), ((0, 0, 255), 50))
    plain, default, vivid = (
        _oklab_chroma(OKLabChromaColor(vividness=v).representative_color(pixels)) for v in (0, 0.5, 1)
    )
    assert plain < default < vivid


@pytest.mark.parametrize("groups", MIXED_HUES[:2])
def test_oklab_chroma_policy_is_more_vivid_than_the_srgb_mean(groups):
    pixels = _pixels(*groups)
    mean = MeanRGBColor().representative_color(pixels)
    assert _oklab_chroma(OKLabChromaColor().representative_color(pixels)) > _oklab_chroma(mean)


def test_oklab_chroma_policy_does_not_amplify_cancelling_hues():
    pixels = _pixels(((255, 0, 0), 50), ((0, 255, 255), 50))
    plain = OKLabChromaColor(vividness=0).representative_color(pixels)
    default = OKLabChromaColor().representative_color(pixels)
    assert _oklab_chroma(default) < 0.5 * _oklab_chroma((255, 0, 0))
    assert _oklab_chroma(plain) <= _oklab_chroma(default)


@pytest.mark.parametrize("vividness", [-0.1, 1.1])
def test_oklab_chroma_policy_validates_vividness(vividness):
    with pytest.raises(ValueError):
        OKLabChromaColor(vividness=vividness)


def test_oklab_round_trip():
    from tiles.color_policies import oklab_to_srgb, srgb_to_oklab

    rng = np.random.default_rng(0)
    for rgb in rng.integers(0, 256, size=(100, 3)):
        assert max(abs(np.array(oklab_to_srgb(srgb_to_oklab(rgb))) - rgb)) <= 1


def test_oklab_out_of_gamut_keeps_channels_valid():
    from tiles.color_policies import oklab_to_srgb

    color = oklab_to_srgb(np.array([0.7, 0.4, 0.3]))
    assert all(0 <= channel <= 255 for channel in color)


def test_most_saturated_policy_picks_the_most_colourful_pixel():
    pixels = _pixels(((128, 128, 128), 50), ((200, 40, 40), 1), ((150, 120, 110), 20))
    assert MostSaturatedColor(top_fraction=0).representative_color(pixels) == (200, 40, 40)


def test_most_saturated_policy_ignores_dark_noise():
    # Near-black pixels have a high HSV saturation but almost no chroma.
    pixels = _pixels(((3, 0, 0), 10), ((60, 120, 200), 3))
    assert MostSaturatedColor(top_fraction=0).representative_color(pixels) == (60, 120, 200)


def test_most_saturated_policy_averages_the_top_fraction():
    pixels = _pixels(((128, 128, 128), 90), ((220, 30, 30), 5), ((200, 50, 50), 5))
    single = MostSaturatedColor(top_fraction=0).representative_color(pixels)
    top_tenth = MostSaturatedColor(top_fraction=0.1).representative_color(pixels)
    assert single == (220, 30, 30)
    assert top_tenth != single
    assert 200 <= top_tenth[0] <= 220 and 30 <= top_tenth[1] <= 50


def test_most_saturated_policy_is_more_vivid_than_the_mean_and_equals_it_at_one():
    pixels = _pixels(((255, 0, 0), 10), ((0, 0, 255), 10), ((128, 128, 128), 30))
    vivid = MostSaturatedColor().representative_color(pixels)
    assert _oklab_chroma(vivid) > _oklab_chroma(MeanRGBColor().representative_color(pixels))
    assert MostSaturatedColor(top_fraction=1).representative_color(pixels) == OKLabChromaColor(
        vividness=0
    ).representative_color(pixels)


def test_most_saturated_policy_returns_the_colour_of_a_uniform_region():
    assert MostSaturatedColor().representative_color(_pixels(((10, 200, 30), 7))) == (10, 200, 30)


def test_most_saturated_policy_rejects_empty_input_and_bad_fraction():
    with pytest.raises(ValueError):
        MostSaturatedColor().representative_color(np.empty((0, 3), dtype=np.uint8))
    for fraction in (-0.1, 1.1):
        with pytest.raises(ValueError):
            MostSaturatedColor(top_fraction=fraction)


@pytest.mark.parametrize(
    "policy",
    [OKLabChromaColor(), DominantColor(), MostSaturatedColor()],
    ids=["oklab", "dominant", "saturated"],
)
def test_vivid_policies_work_in_partitioners(policy):
    result = TILEs.quadtree(prepare_image(_transparent_image(32, 32)), max_level=2, std_thr=0, color_policy=policy)
    assert len(result["colors"]) == 16


# --- stop policies ---


def _stats(**overrides):
    from tiles.types import RegionStats

    values = {
        "standard_deviation": 10.0,
        "mean_color": (0, 0, 0),
        "heightmap_max": 1.0,
        "visible_pixel_count": 100,
        "heightmap_mean": 0.5,
        "bbox_size": (16, 8),
    }
    return RegionStats(**{**values, **overrides})


def _grow(context, stop_policy):
    from tiles.partitioners import QuadtreePartitioner
    from tiles.recursion import grow_tree

    return grow_tree(context, QuadtreePartitioner(), stop_policy, seed=0)


def _half_bright_context(gradient=True):
    """Horizontal colour gradient; heightmap is 0 on the left half and 1 on the right half."""
    pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    if gradient:
        pixels[:] = (np.arange(64) * 4)[None, :, None]
    heightmap = np.zeros((64, 64), dtype=np.uint8)
    heightmap[:, 32:] = 255
    return prepare_image(Image.fromarray(pixels), Image.fromarray(heightmap))


@pytest.mark.parametrize("use_tables", [True, False], ids=["tables", "slices"])
def test_stats_report_heightmap_mean_and_bbox_size(monkeypatch, use_tables):
    from tiles import integral
    from tiles.statistics import RegionAnalysis

    if not use_tables:
        monkeypatch.setattr(integral, "INTEGRAL_MAX_PIXELS", 0)
    rng = np.random.default_rng(0)
    context = prepare_image(_transparent_image(), Image.fromarray(rng.integers(0, 256, (30, 40), dtype=np.uint8)))

    for left, top, right, bottom in [(0, 0, 40, 30), (3, 5, 20, 17), (10, 10, 11, 11)]:
        polygon = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=float)
        stats = RegionAnalysis(context, Region(polygon)).stats
        _, heights = _reference_stats(context, left, top, right, bottom)
        assert stats.bbox_size == (right - left, bottom - top)
        if len(heights):
            assert stats.heightmap_mean == pytest.approx(float(np.mean(heights)))


def test_polygon_stats_report_heightmap_mean():
    from tiles.statistics import RegionAnalysis

    context = _half_bright_context(gradient=False)
    triangle = np.array([[40, 0], [60, 0], [40, 20]], dtype=float)
    stats = RegionAnalysis(context, Region(triangle)).stats
    assert stats.heightmap_mean == 1.0
    assert stats.bbox_size[0] > 0


def test_contrast_policy_tolerance_shrinks_with_heightmap():
    from tiles import HeightmapContrastStop

    policy = HeightmapContrastStop(standard_deviation=40, max_level=6)
    region = Region(SQUARE)
    assert policy.should_stop(region, _stats(standard_deviation=39, heightmap_mean=0.0))
    assert not policy.should_stop(region, _stats(standard_deviation=39, heightmap_mean=0.5))
    assert policy.should_stop(region, _stats(standard_deviation=19, heightmap_mean=0.5))
    assert not policy.should_stop(region, _stats(standard_deviation=0.1, heightmap_mean=1.0))
    assert policy.should_stop(region, _stats(standard_deviation=0.0, heightmap_mean=1.0))


def test_contrast_policy_gamma_confines_detail_to_bright_areas():
    from tiles import HeightmapContrastStop

    region = Region(SQUARE)
    stats = _stats(standard_deviation=15, heightmap_mean=0.5)
    assert HeightmapContrastStop(40, gamma=1.0).should_stop(region, stats)  # tolerance 20
    assert not HeightmapContrastStop(40, gamma=3.0).should_stop(region, stats)  # tolerance 5


def test_contrast_policy_stops_on_depth_and_transparency():
    from tiles import HeightmapContrastStop

    policy = HeightmapContrastStop(standard_deviation=0, max_level=2)
    assert policy.should_stop(Region(SQUARE, level=2), _stats(standard_deviation=99, heightmap_mean=1.0))
    assert policy.should_stop(Region(SQUARE), _stats(visible_pixel_count=0))


def test_contrast_policy_end_to_end_splits_bright_half_more():
    from tiles import HeightmapContrastStop

    tiles = _grow(_half_bright_context(), HeightmapContrastStop(standard_deviation=40, max_level=4))
    left = [tile for tile in tiles if tile.center[0] < 32]
    right = [tile for tile in tiles if tile.center[0] > 32]
    assert {tile.bbox[2] - tile.bbox[0] for tile in left} == {32}
    assert {tile.bbox[2] - tile.bbox[0] for tile in right} == {4}


@pytest.mark.parametrize("kwargs", [{"standard_deviation": -1}, {"gamma": 0}])
def test_contrast_policy_validates_arguments(kwargs):
    from tiles import HeightmapContrastStop

    with pytest.raises(ValueError):
        HeightmapContrastStop(**kwargs)


def test_target_tile_size_interpolates_with_heightmap():
    from tiles import TargetTileSizeStop

    policy = TargetTileSizeStop(max_size=64, min_size=8)
    assert policy.target_size(0.0) == 64
    assert policy.target_size(1.0) == 8
    assert policy.target_size(0.5) == 36
    assert TargetTileSizeStop(64, 8, gamma=2).target_size(0.5) == 8 + 56 * 0.25


def test_target_tile_size_stops_on_longest_side():
    from tiles import TargetTileSizeStop

    policy = TargetTileSizeStop(max_size=20, min_size=10)
    region = Region(SQUARE)
    assert policy.should_stop(region, _stats(bbox_size=(15, 15), heightmap_mean=0.5))
    assert not policy.should_stop(region, _stats(bbox_size=(5, 16), heightmap_mean=0.5))
    assert policy.should_stop(region, _stats(visible_pixel_count=0, bbox_size=(99, 99)))


def test_target_tile_size_end_to_end_follows_heightmap():
    from tiles import TargetTileSizeStop

    tiles = _grow(_half_bright_context(gradient=False), TargetTileSizeStop(max_size=64, min_size=8))
    sizes = {side for tile in tiles for side in (tile.bbox[2] - tile.bbox[0], tile.bbox[3] - tile.bbox[1])}
    left = {tile.bbox[2] - tile.bbox[0] for tile in tiles if tile.center[0] < 32}
    right = {tile.bbox[2] - tile.bbox[0] for tile in tiles if tile.center[0] > 32}
    assert left == {32}
    assert right == {8}
    assert sizes == {8, 32}


@pytest.mark.parametrize(
    "kwargs", [{"max_size": 10, "min_size": 0}, {"max_size": 5, "min_size": 10}, {"max_size": 9, "gamma": -1}]
)
def test_target_tile_size_validates_arguments(kwargs):
    from tiles import TargetTileSizeStop

    with pytest.raises(ValueError):
        TargetTileSizeStop(**kwargs)


def test_policy_combinators():
    from tiles import AllOf, AnyOf, MaxLevelStop, UniformColorStop

    region = Region(SQUARE, level=3)
    flat, deep = UniformColorStop(5), MaxLevelStop(3)
    noisy = _stats(standard_deviation=50)

    assert AnyOf(flat, deep).should_stop(region, noisy)
    assert not AllOf(flat, deep).should_stop(region, noisy)
    assert AllOf(flat, deep).should_stop(region, _stats(standard_deviation=1))
    assert not AnyOf(flat, deep).should_stop(Region(SQUARE, level=1), noisy)
    with pytest.raises(ValueError):
        AnyOf()


def test_target_size_with_flat_regions_through_any_of():
    from tiles import AnyOf, TargetTileSizeStop, UniformColorStop

    pixels = np.zeros((64, 64, 3), dtype=np.uint8)
    pixels[:, 32:] = (np.arange(32) * 8)[None, :, None]
    context = prepare_image(Image.fromarray(pixels), Image.new("L", (64, 64), 255))
    policy = AnyOf(TargetTileSizeStop(max_size=64, min_size=8), UniformColorStop(0))

    tiles = list(_grow(context, policy))
    flat = [tile for tile in tiles if tile.center[0] < 32]
    detailed = [tile for tile in tiles if tile.center[0] > 32]
    assert {tile.bbox[2] - tile.bbox[0] for tile in flat} == {32}
    assert {tile.bbox[2] - tile.bbox[0] for tile in detailed} == {8}


# --- heightmap range, stochastic and perceptual stop policies ---


@pytest.mark.parametrize("use_tables", [True, False], ids=["tables", "slices"])
def test_stats_report_heightmap_min(monkeypatch, use_tables):
    from tiles import integral
    from tiles.statistics import RegionAnalysis

    if not use_tables:
        monkeypatch.setattr(integral, "INTEGRAL_MAX_PIXELS", 0)
    rng = np.random.default_rng(1)
    context = prepare_image(_transparent_image(), Image.fromarray(rng.integers(0, 256, (30, 40), dtype=np.uint8)))

    for left, top, right, bottom in [(0, 0, 40, 30), (3, 5, 20, 17), (25, 2, 39, 9)]:
        polygon = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=float)
        stats = RegionAnalysis(context, Region(polygon)).stats
        _, heights = _reference_stats(context, left, top, right, bottom)
        if len(heights):
            assert stats.heightmap_min == float(np.min(heights))
            assert stats.heightmap_range == float(np.max(heights) - np.min(heights))


def test_polygon_stats_report_heightmap_min():
    from tiles.statistics import RegionAnalysis

    context = _half_bright_context(gradient=False)
    straddling = np.array([[20, 0], [50, 0], [20, 30]], dtype=float)
    stats = RegionAnalysis(context, Region(straddling)).stats
    assert (stats.heightmap_min, stats.heightmap_max) == (0.0, 1.0)


@pytest.mark.parametrize("as_polygon", [False, True], ids=["rectangle", "polygon"])
def test_oklab_variance_matches_reference(as_polygon):
    from tiles.color_policies import srgb_to_oklab
    from tiles.statistics import RegionAnalysis

    context = prepare_image(_transparent_image())
    left, top, right, bottom = 4, 3, 30, 22
    polygon = np.array([[left, top], [right, top], [right, bottom], [left, bottom]], dtype=float)
    if as_polygon:
        polygon = np.array([[left, top], [right, top], [(left + right) / 2, bottom]], dtype=float)
    analysis = RegionAnalysis(context, Region(polygon))

    selected, _ = analysis.samples
    expected = np.var(srgb_to_oklab(selected), axis=0)
    assert analysis.stats.oklab_variance == pytest.approx(tuple(expected))


def test_oklab_variance_of_an_empty_region_is_zero():
    from tiles.statistics import RegionAnalysis

    context = prepare_image(Image.new("RGBA", (8, 8), (10, 20, 30, 0)))
    polygon = np.array([[0, 0], [8, 0], [8, 8], [0, 8]], dtype=float)
    assert RegionAnalysis(context, Region(polygon)).stats.oklab_variance == (0.0, 0.0, 0.0)


def test_oklab_variance_needs_pixel_data():
    with pytest.raises(ValueError):
        _ = _stats().oklab_variance


def test_range_policy_stops_on_flat_heightmaps():
    from tiles import HeightmapRangeStop

    policy = HeightmapRangeStop(max_range=0.1)
    region = Region(SQUARE)
    assert policy.should_stop(region, _stats(heightmap_min=0.95, heightmap_max=1.0))
    assert not policy.should_stop(region, _stats(heightmap_min=0.0, heightmap_max=0.5))
    assert policy.should_stop(region, _stats(visible_pixel_count=0))
    with pytest.raises(ValueError):
        HeightmapRangeStop(max_range=-1)


def test_range_policy_keeps_heightmap_edges_sharp():
    from tiles import AllOf, HeightmapRangeStop, UniformColorStop

    pixels = np.full((64, 64, 3), 90, dtype=np.uint8)
    heightmap = np.zeros((64, 64), dtype=np.uint8)
    heightmap[:, 40:] = 255
    context = prepare_image(Image.fromarray(pixels), Image.fromarray(heightmap))

    tiles = list(_grow(context, AllOf(UniformColorStop(5), HeightmapRangeStop(0.1))))
    widths = {tile.bbox[2] - tile.bbox[0] for tile in tiles}
    assert widths == {32, 16, 8}
    assert not any(tile.bbox[0] < 40 < tile.bbox[2] for tile in tiles)


def test_stochastic_policy_probability_follows_heightmap():
    from tiles import StochasticStop

    policy = StochasticStop(dark_probability=0.8, bright_probability=0.2)
    assert policy.stop_probability(0.0) == pytest.approx(0.8)
    assert policy.stop_probability(1.0) == pytest.approx(0.2)
    assert policy.stop_probability(0.5) == pytest.approx(0.5)
    assert StochasticStop(0.8, 0.2, gamma=2).stop_probability(0.5) == pytest.approx(0.2 + 0.6 * 0.25)


def test_stochastic_policy_respects_levels_and_needs_a_generator():
    from tiles import StochasticStop

    policy = StochasticStop(dark_probability=1.0, bright_probability=1.0, max_level=3, min_level=1)
    generator = np.random.default_rng(0)
    assert not policy.should_stop(Region(SQUARE, level=0), _stats(), generator)
    assert policy.should_stop(Region(SQUARE, level=1), _stats(), generator)
    assert policy.should_stop(Region(SQUARE, level=3), _stats(), None)
    assert policy.should_stop(Region(SQUARE, level=0), _stats(visible_pixel_count=0), None)
    with pytest.raises(ValueError):
        policy.should_stop(Region(SQUARE, level=1), _stats(), None)


@pytest.mark.parametrize(
    "kwargs", [{"dark_probability": 1.5}, {"bright_probability": -0.1}, {"gamma": 0}, {"min_level": -1}]
)
def test_stochastic_policy_validates_arguments(kwargs):
    from tiles import StochasticStop

    with pytest.raises(ValueError):
        StochasticStop(**kwargs)


def test_stochastic_policy_end_to_end_follows_heightmap():
    from tiles import StochasticStop

    policy = StochasticStop(dark_probability=1.0, bright_probability=0.0, max_level=3, min_level=1)
    tiles = _grow(_half_bright_context(gradient=False), policy)
    assert {tile.bbox[2] - tile.bbox[0] for tile in tiles if tile.center[0] < 32} == {32}
    assert {tile.bbox[2] - tile.bbox[0] for tile in tiles if tile.center[0] > 32} == {8}


def test_stochastic_policy_is_reproducible_from_the_seed():
    from tiles import StochasticStop
    from tiles.partitioners import QuadtreePartitioner
    from tiles.recursion import grow_tree

    context = _half_bright_context()
    policy = StochasticStop(dark_probability=0.6, bright_probability=0.3, max_level=5)

    def boxes(seed):
        return [tile.bbox for tile in grow_tree(context, QuadtreePartitioner(), policy, seed=seed)]

    assert boxes(7) == boxes(7)
    assert any(boxes(seed) != boxes(7) for seed in range(8, 16))


def test_combinators_forward_the_generator():
    from tiles import AllOf, AnyOf, StochasticStop

    policy = StochasticStop(dark_probability=1.0, bright_probability=1.0, max_level=9)
    region = Region(SQUARE, level=1)
    for combination in (AnyOf(policy), AllOf(policy)):
        assert combination.should_stop(region, _stats(), np.random.default_rng(0))


def _two_pixel_stats(first, second):
    from tiles.statistics import RegionAnalysis

    context = prepare_image(Image.fromarray(np.array([[first, second]], dtype=np.uint8)))
    polygon = np.array([[0, 0], [2, 0], [2, 1], [0, 1]], dtype=float)
    return RegionAnalysis(context, Region(polygon)).stats


def test_perceptual_policy_error_is_the_oklab_rms_distance():
    from tiles import PerceptualUniformityStop

    black_white = _two_pixel_stats((0, 0, 0), (255, 255, 255))
    assert PerceptualUniformityStop().error(black_white) == pytest.approx(0.5, abs=1e-3)
    assert PerceptualUniformityStop(lightness_weight=0).error(black_white) == pytest.approx(0.0, abs=1e-3)
    assert PerceptualUniformityStop().error(_two_pixel_stats((7, 8, 9), (7, 8, 9))) == pytest.approx(0.0)


def test_perceptual_policy_decisions():
    from tiles import PerceptualUniformityStop

    region = Region(SQUARE)
    near = _two_pixel_stats((100, 100, 100), (102, 100, 100))
    far = _two_pixel_stats((0, 0, 0), (255, 255, 255))
    policy = PerceptualUniformityStop(tolerance=0.05)
    assert policy.should_stop(region, near)
    assert not policy.should_stop(region, far)
    assert PerceptualUniformityStop(tolerance=0.6).should_stop(region, far)
    assert policy.should_stop(region, _stats(visible_pixel_count=0))


def test_perceptual_policy_depth_is_bounded_whatever_the_tolerance():
    from tiles import PerceptualUniformityStop

    noise = np.random.default_rng(0).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    context = prepare_image(Image.fromarray(noise))

    assert len(_grow(context, PerceptualUniformityStop(tolerance=0, max_level=3))) == 4**3
    assert PerceptualUniformityStop(tolerance=0, max_level=2).should_stop(
        Region(SQUARE, level=2), _two_pixel_stats((0, 0, 0), (255, 255, 255))
    )


@pytest.mark.parametrize("kwargs", [{"tolerance": -1}, {"lightness_weight": -1}])
def test_perceptual_policy_validates_arguments(kwargs):
    from tiles import PerceptualUniformityStop

    with pytest.raises(ValueError):
        PerceptualUniformityStop(**kwargs)


def test_perceptual_policy_end_to_end_keeps_flat_areas_large():
    from tiles import AnyOf, MaxLevelStop, PerceptualUniformityStop

    pixels = np.full((64, 64, 3), 120, dtype=np.uint8)
    pixels[:, 32:] = np.random.default_rng(0).integers(0, 256, (64, 32, 3), dtype=np.uint8)
    context = prepare_image(Image.fromarray(pixels))

    tiles = _grow(context, AnyOf(PerceptualUniformityStop(tolerance=0.05), MaxLevelStop(3)))
    assert {tile.bbox[2] - tile.bbox[0] for tile in tiles if tile.center[0] < 32} == {32}
    assert {tile.bbox[2] - tile.bbox[0] for tile in tiles if tile.center[0] > 32} == {8}


def test_load_images_converts_embedded_icc_profile_to_srgb():
    prophoto = Image.open(Path(__file__).parents[1] / "src" / "images" / "Portrait-3.jpg").info["icc_profile"]
    image = Image.fromarray(np.full((4, 4, 3), (200, 120, 120), dtype=np.uint8))
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "tagged.png"
        image.save(path, icc_profile=prophoto)
        context = load_images(str(path))
    assert "icc_profile" not in context.image.info
    red, green, blue = context.rgba[0, 0, :3].astype(int)
    assert red - green > 80 and red > 200


def test_color_policy_repr_uses_new_names():
    assert repr(MeanRGBColor()) == "MeanRGBColor()"
    assert repr(OKLabChromaColor(vividness=0.25)) == "OKLabChromaColor(vividness=0.25)"
    assert repr(DominantColor(bits=3)) == "DominantColor(bits=3)"
