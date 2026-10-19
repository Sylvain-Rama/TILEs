import numpy as np
import pytest
from PIL import Image

import TILEs
from tiles import MedianRGBColorPolicy


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
