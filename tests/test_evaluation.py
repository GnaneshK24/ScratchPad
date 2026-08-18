import unittest

from app import pr_curve_values, validation_history_summary


def run(search_noise, errors, confidences):
    return {"search_noise": search_noise, "reference_noise": "clean", "search_directory": search_noise,
            "reference_directory": "clean", "skipped": [],
            "rows": [{"Sample": str(index), "error_px": error, "confidence": confidence}
                     for index, (error, confidence) in enumerate(zip(errors, confidences))]}


class EvaluationTests(unittest.TestCase):
    def test_history_metrics_and_pr_inputs_are_real(self):
        runs = [run("clean", [.8, 1.2, 2.1, 3.0], [.95, .80, .65, .50]),
                run("high", [4.0, 8.0, 15.0, 30.0], [.85, .60, .35, .10])]
        summary = validation_history_summary(runs)
        self.assertEqual([row["Samples"] for row in summary], [4, 4])
        self.assertGreater(summary[0]["Acc@2"], summary[1]["Acc@2"])
        recall, precision, auc, warning = pr_curve_values(runs[0]["rows"], tolerance=2)
        self.assertIsNone(warning)
        self.assertEqual(len(recall), len(precision))
        self.assertGreaterEqual(auc, 0.0)


if __name__ == "__main__":
    unittest.main()
