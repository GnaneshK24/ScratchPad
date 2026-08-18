import unittest

import numpy as np

from src.localization.visualize import GT_COLOR, PREDICTION_COLOR, render_localization_result


class LocalizationOverlayTests(unittest.TestCase):
    def test_center_coordinates_produce_correct_100px_boxes(self):
        image = np.zeros((1000, 1000), dtype=np.uint8)
        overlay = render_localization_result(image, (500, 500), (200, 200), reference_size=100)
        self.assertTrue(np.array_equal(overlay[450, 450], GT_COLOR))
        self.assertTrue(np.array_equal(overlay[150, 150], PREDICTION_COLOR))

    def test_nearby_centers_do_not_reuse_ground_truth_for_prediction(self):
        image = np.zeros((1000, 1000), dtype=np.uint8)
        overlay = render_localization_result(image, (500, 500), (502, 499), reference_size=100)
        self.assertTrue(np.array_equal(overlay[449, 452], PREDICTION_COLOR))


if __name__ == '__main__':
    unittest.main()
