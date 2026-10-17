import unittest

from fakes import person
from patientmon.tracking import ByteTracker


def box(x, y=100, w=50, h=100):
    return (x, y, x + w, y + h)


def run(tracker, frames):
    return [tracker.update(dets) for dets in frames]


class ByteTrackTest(unittest.TestCase):
    def test_single_person_keeps_id(self):
        tracker = ByteTracker()
        out = run(tracker, [[person(*box(10 + 5 * i), score=0.9)] for i in range(30)])
        ids = {t.track_id for frame in out for t in frame}
        self.assertEqual(ids, {1})
        self.assertEqual(len(out[0]), 1)  # khung đầu tiên xuất ngay
        self.assertGreaterEqual(sum(len(f) for f in out), 29)

    def test_two_separate_people_get_different_stable_ids(self):
        tracker = ByteTracker()
        out = run(tracker, [[person(*box(10 + 3 * i), score=0.9), person(*box(400 - 3 * i), score=0.8)] for i in range(20)])
        last = {round(t.box[0]) // 200: t.track_id for t in out[-1]}
        self.assertEqual(len(set(last.values())), 2)
        self.assertEqual(len({t.track_id for f in out for t in f}), 2)

    def test_short_occlusion_keeps_id(self):
        tracker = ByteTracker()
        frames = [[person(*box(10 + 5 * i), score=0.9)] for i in range(10)]
        frames += [[] for _ in range(6)]  # mất phát hiện 6 khung
        frames += [[person(*box(10 + 5 * i), score=0.9)] for i in range(16, 26)]
        ids = {t.track_id for f in run(tracker, frames) for t in f}
        self.assertEqual(ids, {1})

    def test_low_score_detection_rescues_track(self):
        tracker = ByteTracker()
        frames = [[person(*box(10 + 5 * i), score=0.9)] for i in range(8)]
        frames += [[person(*box(10 + 5 * i), score=0.15)] for i in range(8, 12)]  # chỉ còn hộp điểm thấp
        frames += [[person(*box(10 + 5 * i), score=0.9)] for i in range(12, 20)]
        out = run(tracker, frames)
        self.assertEqual({t.track_id for f in out for t in f}, {1})
        self.assertTrue(all(len(f) == 1 for f in out[8:12]))  # vẫn xuất track trong lúc điểm thấp

    def test_low_score_alone_never_starts_a_track(self):
        tracker = ByteTracker()
        out = run(tracker, [[person(*box(10), score=0.15)] for _ in range(10)])
        self.assertTrue(all(len(f) == 0 for f in out))

    def test_long_absence_gets_new_id(self):
        tracker = ByteTracker(buffer=10)
        frames = [[person(*box(100), score=0.9)] for _ in range(10)] + [[] for _ in range(20)]
        frames += [[person(*box(100), score=0.9)] for _ in range(10)]
        out = run(tracker, frames)
        self.assertEqual(out[9][0].track_id, 1)
        self.assertEqual(out[-1][0].track_id, 2)

    def test_reset_restarts_ids(self):
        tracker = ByteTracker()
        run(tracker, [[person(*box(10), score=0.9)] for _ in range(3)])
        tracker.reset()
        self.assertEqual(tracker.update([person(*box(10), score=0.9)])[0].track_id, 1)

    def test_frame_rate_scales_buffer(self):
        self.assertEqual(ByteTracker(buffer=30, frame_rate=15).max_time_lost, 15)
        self.assertEqual(ByteTracker(buffer=30, frame_rate=30).max_time_lost, 30)


if __name__ == "__main__":
    unittest.main()
