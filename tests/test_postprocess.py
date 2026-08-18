import unittest

import numpy as np

from app import apply_existing_postprocessing


class PostprocessTests(unittest.TestCase):
    def test_astigmatism_changes_pixels_without_mutating_input(self):
        image = np.tile(np.arange(128, dtype=np.uint8), (128, 1))
        params = {"gamma": 1.0, "vignetting": 0.0, "astigmatism": 1.0,
                  "charging": 0.0, "streaks": 0.0}
        processed = apply_existing_postprocessing(image, params, "horizontal")
        self.assertFalse(np.array_equal(processed, image))
        self.assertTrue(np.array_equal(image, np.tile(np.arange(128, dtype=np.uint8), (128, 1))))

    def test_each_exposed_postprocess_filter_changes_a_copy(self):
        image = np.tile(np.arange(128, dtype=np.uint8), (128, 1))
        variants = (
            {"gamma": 0.7, "vignetting": 0.0, "astigmatism": 0.0, "charging": 0.0, "streaks": 0.0},
            {"gamma": 1.0, "vignetting": 0.5, "astigmatism": 0.0, "charging": 0.0, "streaks": 0.0},
            {"gamma": 1.0, "vignetting": 0.0, "astigmatism": 0.0, "charging": 0.5, "streaks": 0.0},
            {"gamma": 1.0, "vignetting": 0.0, "astigmatism": 0.0, "charging": 0.0, "streaks": 1.0},
        )
        for params in variants:
            with self.subTest(params=params):
                processed = apply_existing_postprocessing(image, params, "horizontal")
                self.assertFalse(np.array_equal(processed, image))
                self.assertTrue(np.array_equal(image, np.tile(np.arange(128, dtype=np.uint8), (128, 1))))


if __name__ == "__main__":
    unittest.main()
