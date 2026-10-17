import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from fakes import FakeDetector, person
from patientmon.evaluation import average_precision, evaluate, iou, match_image
from patientmon.labels import write_labels


class MetricsTest(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertAlmostEqual(iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3)
        self.assertEqual(iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)

    def test_average_precision_known_cases(self):
        self.assertAlmostEqual(average_precision([0.9, 0.8], [True, True], 2), 1.0)
        self.assertAlmostEqual(average_precision([0.9], [True], 2), 0.5)  # bỏ sót một nửa
        self.assertEqual(average_precision([], [], 3), 0.0)

    def test_each_ground_truth_matched_at_most_once(self):
        gt = [person(0, 0, 10, 10)]
        flags = [f for _, f in match_image([person(0, 0, 10, 10, 0.9), person(0, 0, 10, 10, 0.8)], gt, 0.5)]
        self.assertEqual(flags, [True, False])


class EvaluateTest(unittest.TestCase):
    def _samples(self, root):
        samples = []
        for i in range(2):
            img, lab = root / f"{i}.jpg", root / f"{i}.txt"
            cv2.imwrite(str(img), np.zeros((100, 100, 3), np.uint8))
            write_labels(lab, [person(30, 30, 70, 70)], 100, 100)
            samples.append((img, lab))
        return samples

    def test_perfect_detector(self):
        with tempfile.TemporaryDirectory() as d:
            m = evaluate(FakeDetector([[person(30, 30, 70, 70)]]), self._samples(Path(d)))
            self.assertEqual((m.precision, m.recall, m.ap50), (1.0, 1.0, 1.0))

    def test_missed_and_false_positive(self):
        with tempfile.TemporaryDirectory() as d:
            detector = FakeDetector([[person(30, 30, 70, 70)], [person(0, 0, 10, 10)]])  # ảnh 2: sai chỗ
            m = evaluate(detector, self._samples(Path(d)))
            self.assertEqual((m.tp, m.n_pred, m.n_gt), (1, 2, 2))
            self.assertAlmostEqual(m.precision, 0.5)
            self.assertAlmostEqual(m.recall, 0.5)

    def test_low_score_prediction_ignored_for_precision_but_counted_in_ap(self):
        with tempfile.TemporaryDirectory() as d:
            detector = FakeDetector([[person(30, 30, 70, 70, score=0.1)]])
            m = evaluate(detector, self._samples(Path(d)), report_conf=0.25)
            self.assertEqual((m.n_pred, m.recall), (0, 0.0))
            self.assertAlmostEqual(m.ap50, 1.0)


if __name__ == "__main__":
    unittest.main()
