import tempfile
import unittest
from pathlib import Path

from fakes import write_video
from patientmon.config import FramesCfg
from patientmon.frames import extract_all, load_manifest
from patientmon.sources import VideoSource, sample
from patientmon.video import assign_ids, probe


class VideoTest(unittest.TestCase):
    def test_probe_and_sampling_interval(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "a.avi"
            write_video(path, n_frames=30, fps=10)
            info = probe(path)
            self.assertTrue(info.ok)
            self.assertEqual((info.width, info.height, round(info.fps)), (96, 64, 10))
            idx = [f.idx for f in sample(VideoSource(path, "a"), every_sec=1.0)]
            self.assertEqual(idx, [0, 10, 20])

    def test_static_video_deduplicated_but_first_frame_kept(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "still.avi"
            write_video(path, n_frames=30, fps=10, moving=False)
            self.assertEqual(len(list(sample(VideoSource(path, "still"), 1.0, min_diff=2.0))), 1)

    def test_max_frames_and_unreadable_video(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "a.avi"
            write_video(path, n_frames=50, fps=10)
            self.assertEqual(len(list(sample(VideoSource(path, "a"), 0.5, max_frames=2))), 2)
            bad = Path(d) / "bad.mp4"
            bad.write_bytes(b"not a video")
            self.assertFalse(probe(bad).ok)
            with self.assertRaises(ValueError):
                VideoSource(bad, "bad")

    def test_ids_are_ascii_and_duplicates_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "phòng 1").mkdir()
            a, b = root / "phòng 1" / "té ngã.mp4", root / "x.mp4"
            self.assertEqual(assign_ids([a, b], root)[a], "phong_1__te_nga")
            with self.assertRaises(ValueError):
                assign_ids([root / "a b.mp4", root / "a_b.mp4"], root)


class ExtractTest(unittest.TestCase):
    def test_extract_all_writes_manifest_and_is_resumable(self):
        with tempfile.TemporaryDirectory() as d:
            videos, frames = Path(d) / "videos", Path(d) / "frames"
            (videos / "sub").mkdir(parents=True)
            write_video(videos / "a.avi", 30, 10)
            write_video(videos / "sub" / "b.avi", 20, 10)
            (videos / "broken.mp4").write_bytes(b"junk")
            cfg = FramesCfg(every_sec=1.0, min_diff=0.0)
            first = extract_all(videos, frames, cfg)
            self.assertEqual(sorted({r.video_id for r in first}), ["a", "sub__b"])
            self.assertEqual(len(first), 5)
            self.assertTrue(all((frames / r.rel_path).exists() for r in first))
            before = {p: p.stat().st_mtime_ns for p in frames.rglob("*.jpg")}
            again = extract_all(videos, frames, cfg)
            self.assertEqual(len(again), 5)
            self.assertEqual(before, {p: p.stat().st_mtime_ns for p in frames.rglob("*.jpg")})  # không cắt lại
            self.assertEqual(len(load_manifest(frames)), 5)

    def test_no_videos_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileNotFoundError):
                extract_all(d, Path(d) / "f", FramesCfg())


if __name__ == "__main__":
    unittest.main()
