import unittest

import numpy as np
from PIL import Image

import TILEs


class PublicApiTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (16, 12), (0, 0, 0))

    def test_partitioners_return_aligned_records(self):
        calls = [
            lambda: TILEs.quadtree(self.image, max_level=1, std_thr=0),
            lambda: TILEs.recursive_slice(self.image, max_level=1, std_thr=0),
            lambda: TILEs.voronoitree(
                self.image,
                npoints=5,
                max_level=1,
                std_thr=0,
                rng=np.random.default_rng(1),
            ),
            lambda: TILEs.throw_polys(
                self.image,
                n_points=3,
                rng=np.random.default_rng(1),
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


if __name__ == "__main__":
    unittest.main()
