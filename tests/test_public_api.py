import unittest

import numpy as np
from PIL import Image

import TILEs
from tiles import MedianRGBColorPolicy


class PublicApiTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (16, 12), (0, 0, 0))

    def test_partitioners_return_aligned_records(self):
        calls = [
            lambda: TILEs.quadtree(
                self.image,
                max_level=1,
                std_thr=0,
                color_policy=MedianRGBColorPolicy(),
            ),
            lambda: TILEs.recursive_slice(
                self.image,
                max_level=1,
                std_thr=0,
                color_policy=MedianRGBColorPolicy(),
            ),
            lambda: TILEs.voronoitree(
                self.image,
                npoints=5,
                max_level=1,
                std_thr=0,
                rng=np.random.default_rng(1),
                color_policy=MedianRGBColorPolicy(),
            ),
            lambda: TILEs.throw_polys(
                self.image,
                n_points=3,
                rng=np.random.default_rng(1),
                color_policy=MedianRGBColorPolicy(),
            ),
        ]
        for call in calls:
            with self.subTest(call=call):
                result = call()
                lengths = {len(values) for values in result.values()}
                self.assertEqual(len(lengths), 1)

    def test_dither_returns_image(self):
        result = TILEs.dither(self.image)
        self.assertIsInstance(result, Image.Image)
        self.assertEqual(result.size, self.image.size)

    def test_quadtree_accepts_representative_color_policy(self):
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

        self.assertEqual(result["colors"], [(15, 15, 15)])

    def test_default_quadtree_color_remains_legacy_mean(self):
        pixels = np.array(
            [
                [[0, 0, 0], [10, 10, 10]],
                [[20, 20, 20], [100, 100, 100]],
            ],
            dtype=np.uint8,
        )
        image = Image.fromarray(pixels)

        result = TILEs.quadtree(image, max_level=0)

        self.assertEqual(result["colors"], [(32, 32, 32)])


if __name__ == "__main__":
    unittest.main()
