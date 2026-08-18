import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch
from pathlib import Path

import numpy as np

from src.localization.multi_evaluation import build_combinations, evaluate_variant_pair, metrics, normalize_sample_id
from src.localization.external_ground_truth import parse_external_ground_truth


def variants(prefix, count):
    return [{"label": f"{prefix}{index}", "directory": f"/{prefix}{index}"} for index in range(count)]


class MultiEvaluationTests(unittest.TestCase):
    def test_arbitrary_cartesian_dimensions(self):
        for searches, references in ((1, 1), (1, 4), (4, 1), (2, 3), (4, 4)):
            pairs, skipped = build_combinations(variants("S", searches), variants("R", references))
            self.assertEqual(len(pairs), searches * references)
            self.assertFalse(skipped)

    def test_matched_labels_and_id_normalization(self):
        pairs, skipped = build_combinations(variants("clean", 1) + variants("high", 1), variants("clean", 1), "matched")
        self.assertEqual(len(pairs), 1)
        self.assertEqual(skipped, ["high0"])
        self.assertEqual(normalize_sample_id("search_0007.png"), "7")
        self.assertEqual(normalize_sample_id("reference_7.png"), "7")

    def test_metrics(self):
        result = metrics([{"error_px": 1.0, "confidence": .8}, {"error_px": 6.0, "confidence": .2}])
        self.assertEqual(result["Samples"], 2)
        self.assertEqual(result["Accuracy@5px"], .5)

    def test_completed_combination_is_reused(self):
        record = {"sample_id": "0001", "search": "search_0001.png", "reference": "reference_0001.png",
                  "center_x": 10.0, "center_y": 10.0, "noise_mode": "clean"}
        variants_pair = ({"label": "S", "directory": "/search"}, {"label": "R", "directory": "/reference"})
        with TemporaryDirectory() as directory, \
             patch("src.localization.multi_evaluation._records", side_effect=[(Path("/search"), {"1": record}), (Path("/reference"), {"1": record})]), \
             patch("src.localization.multi_evaluation.cv2.imread", return_value=np.zeros((20, 20), dtype=np.uint8)):
            calls = []
            def localizer(*_):
                calls.append(1); return {"center_x": 10.0, "center_y": 10.0, "confidence": .8}
            rows, details = evaluate_variant_pair(*variants_pair, directory, localizer=localizer)
            self.assertEqual(len(rows), 1); self.assertFalse(details["cached"])
            with patch("src.localization.multi_evaluation._records") as records:
                reused, reused_details = evaluate_variant_pair(*variants_pair, directory, localizer=localizer)
            self.assertEqual(len(reused), 1); self.assertTrue(reused_details["cached"])
            self.assertEqual(len(calls), 1); records.assert_not_called()

    def test_external_gt_overrides_dataset_metadata_for_each_combination(self):
        record = {"sample_id": "0001", "search": "search_0001.png", "reference": "reference_0001.png",
                  "center_x": 10.0, "center_y": 10.0, "noise_mode": "clean"}
        variants_pair = ({"label": "S", "directory": "/search"}, {"label": "R", "directory": "/reference"})
        external = parse_external_ground_truth("ground_truth.csv", b"sample_id,center_x,center_y\n0001,5,7\n")
        with TemporaryDirectory() as directory, \
             patch("src.localization.multi_evaluation._records", side_effect=[(Path("/search"), {"1": record}), (Path("/reference"), {"1": record})]), \
             patch("src.localization.multi_evaluation.cv2.imread", return_value=np.zeros((20, 20), dtype=np.uint8)):
            rows, details = evaluate_variant_pair(*variants_pair, directory,
                                                  localizer=lambda *_: {"center_x": 5.0, "center_y": 7.0, "confidence": .8},
                                                  external_ground_truth=external)
        self.assertEqual(rows[0]["gt_x"], 5.0)
        self.assertEqual(rows[0]["error_px"], 0.0)
        self.assertEqual(details["matched_gt"], 1)


if __name__ == "__main__":
    unittest.main()
