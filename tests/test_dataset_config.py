import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from fakes import person
from patientmon.config import DetectionConfig
from patientmon.contracts import FrameRecord
from patientmon.dataset import build_yolo_dataset, load_index
from patientmon.labels import label_path, write_labels

RATIOS = {"train": 0.6, "val": 0.2, "test": 0.2}


def make_workspace(root, videos=5, per_video=4, reviewed_every=1):
    frames, labels, records = root / "frames", root / "labels", []
    for v in range(videos):
        for i in range(per_video):
            rel = f"v{v}/v{v}_f{i:07d}.jpg"
            (frames / f"v{v}").mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(frames / rel), np.zeros((40, 60, 3), np.uint8))
            rec = FrameRecord(f"v{v}", i, float(i), rel)
            records.append(rec)
            if i % reviewed_every == 0:
                write_labels(label_path(labels, rec), [person(5, 5, 30, 30)], 60, 40)
    return frames, labels, records


class BuildDatasetTest(unittest.TestCase):
    def test_split_by_video_without_leakage_and_data_yaml(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            frames, labels, records = make_workspace(root)
            out = root / "yolo"
            summary = build_yolo_dataset(records, frames, labels, out, ["person"], RATIOS, seed=0)
            self.assertEqual(sum(s["frames"] for s in summary.values()), 20)
            videos_by_split = {}
            for row in load_index(out):
                videos_by_split.setdefault(row["split"], set()).add(row["video_id"])
            sets = list(videos_by_split.values())
            for i, a in enumerate(sets):
                for b in sets[i + 1:]:
                    self.assertFalse(a & b)  # không video nào nằm ở hai tập
            data = yaml.safe_load((out / "data.yaml").read_text(encoding="utf-8"))
            self.assertEqual(data["names"], {0: "person"})
            self.assertTrue((out / data["train"]).is_dir())

    def test_only_reviewed_frames_are_used(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            frames, labels, records = make_workspace(root, reviewed_every=2)
            summary = build_yolo_dataset(records, frames, labels, root / "yolo", ["person"], RATIOS)
            self.assertEqual(sum(s["frames"] for s in summary.values()), 10)

    def test_group_column_keeps_same_person_together(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            frames, labels, records = make_workspace(root, videos=6)
            meta = {f"v{i}": {"video_id": f"v{i}", "group": f"p{i // 2}"} for i in range(6)}
            out = root / "yolo"
            build_yolo_dataset(records, frames, labels, out, ["person"], RATIOS, meta=meta)
            split_of = {}
            for row in load_index(out):
                split_of.setdefault(row["group"], set()).add(row["split"])
            self.assertTrue(all(len(s) == 1 for s in split_of.values()))

    def test_refuses_to_delete_unrelated_directory_and_empty_labels_raise(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            frames, labels, records = make_workspace(root)
            out = root / "precious"
            out.mkdir()
            (out / "thesis.docx").write_text("x")
            with self.assertRaises(FileExistsError):
                build_yolo_dataset(records, frames, labels, out, ["person"], RATIOS)
            with self.assertRaises(ValueError):
                build_yolo_dataset(records, frames, root / "no_labels", root / "yolo2", ["person"], RATIOS)


class ConfigTest(unittest.TestCase):
    def test_default_yaml_loads(self):
        cfg = DetectionConfig.from_yaml(Path(__file__).parent.parent / "configs" / "config.yaml")
        self.assertEqual(cfg.dataset.names, ["person"])
        self.assertEqual(cfg.prelabel.class_map, {0: 0})
        self.assertEqual(cfg.paths.resolve("frames"), Path("data/frames"))

    def test_unknown_keys_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            for text in ("frames:\n  every_secs: 1\n", "framez:\n  every_sec: 1\n"):
                path = Path(d) / "c.yaml"
                path.write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    DetectionConfig.from_yaml(path)


if __name__ == "__main__":
    unittest.main()
