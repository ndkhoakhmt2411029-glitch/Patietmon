import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from fakes import FakeDetector, person
from patientmon.contracts import FrameRecord
from patientmon.labels import QUEUE_FILE, _read_queue, label_path, prelabel, read_labels, write_labels


class LabelIoTest(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "a.txt"
            write_labels(path, [person(10, 20, 50, 80, cls=1)], 100, 100)
            (det,) = read_labels(path, 100, 100)
            self.assertEqual(det.cls, 1)
            for got, want in zip(det.box, (10, 20, 50, 80)):
                self.assertAlmostEqual(got, want, places=3)

    def test_box_is_clipped_and_degenerate_boxes_dropped(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "a.txt"
            write_labels(path, [person(-10, -10, 200, 200), person(30, 30, 30, 60)], 100, 100)
            dets = read_labels(path, 100, 100)
            self.assertEqual(len(dets), 1)
            self.assertEqual(dets[0].box, (0.0, 0.0, 100.0, 100.0))

    def test_empty_label_file_means_reviewed_background(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "a.txt"
            write_labels(path, [], 100, 100)
            self.assertTrue(path.exists())
            self.assertEqual(read_labels(path, 100, 100), [])


class PrelabelTest(unittest.TestCase):
    def _make_frames(self, root, n):
        recs = []
        for i in range(n):
            rel = f"v1/v1_f{i:07d}.jpg"
            (root / "v1").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(root / rel), np.zeros((50, 80, 3), np.uint8))
            recs.append(FrameRecord("v1", i, float(i), rel))
        return recs

    def test_queue_sorted_empty_then_uncertain_then_confident_and_resumable(self):
        with tempfile.TemporaryDirectory() as d:
            frames, labels = Path(d) / "frames", Path(d) / "labels"
            recs = self._make_frames(frames, 3)
            detector = FakeDetector([[person(5, 5, 30, 40, score=0.95)], [], [person(5, 5, 30, 40, score=0.3)]])
            queue = prelabel(detector, recs, frames, labels, batch_size=2, confident_score=0.6)
            rows = _read_queue(queue)
            self.assertEqual([r["status"] for r in rows], ["empty", "uncertain", "confident"])
            self.assertTrue(label_path(labels, recs[1]).exists())  # khung trống vẫn có file nhãn rỗng
            calls = detector.calls
            prelabel(detector, recs, frames, labels, batch_size=2)
            self.assertEqual(detector.calls, calls)  # chạy lại không xử lý lại
            self.assertEqual(len(_read_queue(labels / QUEUE_FILE)), 3)


if __name__ == "__main__":
    unittest.main()
