import tempfile
import unittest
from pathlib import Path

from fakes import write_image_folder, write_video
from patientmon.sources import ImageFolderSource, StillsSource, VideoSource, discover_sources, sample
from patientmon.video import assign_ids


class SourcesTest(unittest.TestCase):
    def test_video_source_step(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "a.avi"
            write_video(path, n_frames=30, fps=10)
            src = VideoSource(path, "a")
            self.assertEqual([f.idx for f in src.frames(step=10)], [0, 10, 20])
            self.assertEqual(len(list(src.frames())), 30)

    def test_image_folder_natural_order_and_fps_required(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / "seq"
            write_image_folder(folder, n=12)  # img_1 ... img_12: thứ tự tự nhiên khác thứ tự chữ
            src = ImageFolderSource(folder, "seq", fps=5)
            self.assertEqual([p.name for p in src.files][:3], ["img_1.jpg", "img_2.jpg", "img_3.jpg"])
            self.assertEqual(src.files[-1].name, "img_12.jpg")
            self.assertEqual([round(f.time_sec, 2) for f in src.frames(step=5)], [0.0, 1.0, 2.0])
            with self.assertRaises(ValueError):
                ImageFolderSource(folder, "seq", fps=None)

    def test_sample_every_sec_and_stills_keep_all(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / "seq"
            write_image_folder(folder, n=10)
            self.assertEqual([f.idx for f in sample(ImageFolderSource(folder, "s", fps=2), every_sec=1.0)], [0, 2, 4, 6, 8])
            self.assertEqual(len(list(sample(StillsSource(folder, "s"), every_sec=1.0))), 10)

    def test_discover_mixed_sources(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            write_video(root / "v" / "clip.avi", 10, 10) if (root / "v").mkdir() is None else None
            write_image_folder(root / "S1" / "walk" / "Camera1", n=3)
            write_image_folder(root / "S1" / "fall" / "Camera1", n=3)
            sources = discover_sources(root, "sequence", image_fps=18)
            ids = sorted(s.id for s in sources)
            self.assertEqual(ids, ["S1__fall__Camera1", "S1__walk__Camera1", "v__clip"])
            self.assertEqual({s.kind for s in sources}, {"video", "sequence"})
            with self.assertRaises(ValueError):  # chuỗi ảnh mà thiếu FPS
                discover_sources(root, "sequence", image_fps=None)
            self.assertTrue(all(s.kind in ("video", "stills") for s in discover_sources(root, "stills")))

    def test_directory_id_keeps_dots(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d) / "run1.5"
            folder.mkdir()
            self.assertEqual(assign_ids([folder], d)[folder], "run1_5")


if __name__ == "__main__":
    unittest.main()
