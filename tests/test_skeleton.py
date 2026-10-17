import tempfile
import unittest
from pathlib import Path

import numpy as np

from fakes import FakeDetector, FakePose, make_sequence, person, write_image_folder, write_video
from patientmon.contracts import SkeletonSequence
from patientmon.skeleton import extract_skeletons, normalize_window, select_tracks, track_windows
from patientmon.sources import ImageFolderSource, VideoSource
from patientmon.tracking import ByteTracker


def moving_dets(n):
    return [[person(10 + 2 * i, 10, 30 + 2 * i, 50, 0.9)] for i in range(n)]


class ExtractTest(unittest.TestCase):
    def _run(self, source, n, target_fps=None, batch=3):
        factory = lambda fps: ByteTracker(frame_rate=fps)
        return extract_skeletons(source, FakeDetector(moving_dets(n)), factory, FakePose(), target_fps, det_batch=batch)

    def test_video_end_to_end_with_fakes_and_save_load(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "a.avi"
            write_video(path, n_frames=20, fps=10)
            seq = self._run(VideoSource(path, "a"), 20)
            self.assertEqual(set(seq.track_id.tolist()), {1})
            self.assertEqual(len(seq.frame_idx), 20)
            self.assertEqual(seq.keypoints.shape, (20, 17, 3))
            self.assertEqual((seq.width, seq.height, seq.step, seq.fps), (96, 64, 1, 10.0))
            # keypoint nằm trong hộp người
            self.assertTrue(np.all(seq.keypoints[:, :, 0] >= seq.boxes[:, 0:1] - 1e-3))
            seq.save(Path(d) / "out" / "a.npz")
            loaded = SkeletonSequence.load(Path(d) / "out" / "a.npz")
            self.assertEqual(loaded.video_id, "a")
            np.testing.assert_array_equal(loaded.keypoints, seq.keypoints)

    def test_target_fps_subsamples_with_integer_step(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "a.avi"
            write_video(path, n_frames=20, fps=10)
            seq = self._run(VideoSource(path, "a"), 10, target_fps=5)
            self.assertEqual((seq.step, seq.fps), (2, 5.0))
            self.assertEqual(seq.frame_idx.tolist(), list(range(0, 20, 2)))

    def test_image_sequence_source_works_the_same(self):
        with tempfile.TemporaryDirectory() as d:
            write_image_folder(Path(d) / "s", n=8)
            seq = self._run(ImageFolderSource(Path(d) / "s", "s", fps=9), 8)
            self.assertEqual((len(seq.frame_idx), seq.fps), (8, 9.0))

    def test_only_person_class_is_tracked(self):
        with tempfile.TemporaryDirectory() as d:
            write_image_folder(Path(d) / "s", n=6)
            dets = [[person(10, 10, 30, 50, 0.9), person(40, 10, 60, 50, 0.9, cls=1)] for _ in range(6)]
            seq = extract_skeletons(ImageFolderSource(Path(d) / "s", "s", 5), FakeDetector(dets),
                                    lambda fps: ByteTracker(frame_rate=fps), FakePose())
            self.assertEqual(len(seq.frame_idx), 6)


class NormalizeTest(unittest.TestCase):
    def _window(self):
        rng = np.random.RandomState(1)
        kp = np.zeros((16, 17, 3), np.float32)
        kp[:, :, :2] = rng.rand(17, 2) * 50 + 100 + np.sin(np.arange(16))[:, None, None] * 4
        kp[:, :, 2] = 0.9
        return kp

    def test_translation_and_scale_invariant(self):
        kp = self._window()
        moved = kp.copy()
        moved[..., :2] = moved[..., :2] * 2.5 + np.array([300.0, -40.0])
        np.testing.assert_allclose(normalize_window(kp), normalize_window(moved), atol=1e-4)

    def test_window_center_keeps_vertical_motion(self):
        kp = self._window()
        falling = kp.copy()
        falling[:, :, 1] += np.linspace(0, 60, 16)[:, None]  # cả người đi xuống theo thời gian
        out = normalize_window(falling)
        self.assertGreater(out[-1, 11, 1], out[0, 11, 1] + 0.1)  # hông đi xuống vẫn thấy được sau chuẩn hóa

    def test_low_confidence_joint_is_interpolated(self):
        kp = self._window()
        kp[:, 9, :2] = np.array([[100.0 + 10 * i, 50.0] for i in range(16)])
        reference = normalize_window(kp)
        kp[8, 9, :2], kp[8, 9, 2] = [999.0, 999.0], 0.0  # mất giữa chừng, tọa độ rác
        out = normalize_window(kp)
        self.assertTrue(np.isfinite(out).all())
        np.testing.assert_allclose(out[8, 9, :2], reference[8, 9, :2], atol=0.05)
        self.assertEqual(out[8, 9, 2], 0.0)  # độ tin cậy gốc được giữ lại cho mô hình

    def test_all_missing_does_not_crash(self):
        out = normalize_window(np.zeros((8, 17, 3), np.float32))
        self.assertTrue(np.isfinite(out).all())


class WindowsTest(unittest.TestCase):
    def test_window_count_and_positions(self):
        seq = make_sequence("v", n=40)
        starts, wins = track_windows(seq, 1, window=16, stride=8, max_missing=0.5)
        self.assertEqual(starts.tolist(), [0, 8, 16, 24])
        self.assertEqual(wins.shape, (4, 16, 17, 3))

    def test_step_is_respected(self):
        seq = make_sequence("v", n=20, step=2)
        starts, wins = track_windows(seq, 1, window=10, stride=5, max_missing=0.5)
        self.assertEqual(starts.tolist(), [0, 5, 10])
        self.assertTrue(np.allclose(wins[0, 3], seq.keypoints[3]))

    def test_gaps_zero_filled_and_too_gappy_windows_dropped(self):
        seq = make_sequence("v", n=30, drop=set(range(10, 20)))
        starts, wins = track_windows(seq, 1, window=10, stride=10, max_missing=0.3)
        self.assertEqual(starts.tolist(), [0, 20])  # cửa sổ 10-19 mất 100% khung nên bị bỏ
        starts, wins = track_windows(seq, 1, window=12, stride=6, max_missing=0.5)
        self.assertTrue((wins[:, :, :, 2] == 0).any())  # khung mất có độ tin cậy 0

    def test_select_tracks_prefers_longest_and_drops_short(self):
        a, b = make_sequence("v", n=40, track_ids=(1,)), make_sequence("v", n=10, track_ids=(2,))
        both = SkeletonSequence("v", 10.0, 1, 640, 480, np.concatenate([a.frame_idx, b.frame_idx]),
                                np.concatenate([a.track_id, b.track_id]), np.concatenate([a.keypoints, b.keypoints]),
                                np.concatenate([a.boxes, b.boxes]), np.concatenate([a.scores, b.scores]))
        self.assertEqual(select_tracks(both, min_frames=5, max_tracks=1), [1])
        self.assertEqual(sorted(select_tracks(both, min_frames=5, max_tracks=None)), [1, 2])
        self.assertEqual(select_tracks(both, min_frames=20, max_tracks=None), [1])


if __name__ == "__main__":
    unittest.main()
