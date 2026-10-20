import unittest

from src.safedrive import detection_metrics, intersection_over_union


class DetectionMetricTests(unittest.TestCase):
    def test_identical_boxes_have_full_overlap(self):
        self.assertEqual(intersection_over_union([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)

    def test_detection_counts_are_matched_one_to_one(self):
        ground_truth = [{"box": [0, 0, 10, 10]}]
        predictions = [
            {"box": [0, 0, 10, 10], "confidence": 0.9},
            {"box": [0, 0, 10, 10], "confidence": 0.8},
        ]
        result = detection_metrics(ground_truth, predictions)
        self.assertEqual((result["TP"], result["FP"], result["FN"]), (1, 1, 0))
        self.assertEqual(result["Precision"], 0.5)
        self.assertEqual(result["Recall"], 1.0)

    def test_missed_ground_truth_increases_false_negative_rate(self):
        result = detection_metrics([{"box": [0, 0, 10, 10]}], [])
        self.assertEqual(result["FNR"], 1.0)


if __name__ == "__main__":
    unittest.main()