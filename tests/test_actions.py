import tempfile
import unittest
from pathlib import Path

import numpy as np

from fakes import make_sequence
from patientmon.actions.data import build_window_dataset, load_windows
from patientmon.actions.labeling import Labeler, make_group_fn
from patientmon.actions.metrics import clip_level, confusion_matrix, report
from patientmon.actions.models.graph import COCO_EDGES, bone_parents, hop_distance, spatial_adjacency
from patientmon.config import ActionCfg

CLASSES = ["stand", "walk", "fall"]
RULES = [{"pattern": "walk", "label": "walk"}, {"pattern": "fall", "label": "fall"}, {"pattern": "stand", "label": "stand"}]


class LabelingTest(unittest.TestCase):
    def test_first_matching_rule_wins_case_insensitive(self):
        lab = Labeler(CLASSES, RULES)
        self.assertEqual(lab.label_for("S01__Walk_fall_demo"), 1)  # "walk" đứng trước "fall"
        self.assertEqual(lab.label_for("s02__FALL__t1"), 2)
        self.assertIsNone(lab.label_for("s02__jump__t1"))

    def test_invalid_configuration_rejected(self):
        with self.assertRaises(ValueError):
            Labeler(CLASSES, [])
        with self.assertRaises(ValueError):
            Labeler(CLASSES, [{"pattern": "x", "label": "run"}])

    def test_group_function(self):
        g = make_group_fn(r"subject(\d+)")
        self.assertEqual(g("Subject7__walk__t1"), "7")
        self.assertEqual(g("other__walk"), "other__walk")  # không khớp thì mỗi đoạn là một nhóm
        self.assertEqual(make_group_fn(None)("abc"), "abc")


class MetricsTest(unittest.TestCase):
    def test_confusion_and_report(self):
        cm = confusion_matrix([0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 0], 3)
        np.testing.assert_array_equal(cm, [[1, 1, 0], [0, 2, 0], [1, 0, 1]])
        r = report(cm)
        np.testing.assert_allclose(r["recall"], [0.5, 1.0, 0.5])
        np.testing.assert_allclose(r["precision"], [0.5, 2 / 3, 1.0])
        self.assertAlmostEqual(r["accuracy"], 4 / 6)
        self.assertAlmostEqual(r["macro_f1"], np.mean([0.5, 0.8, 2 / 3]), places=6)

    def test_absent_class_excluded_from_macro(self):
        r = report(confusion_matrix([0, 0, 1], [0, 0, 1], 3))  # lớp 2 không có mẫu
        self.assertAlmostEqual(r["macro_f1"], 1.0)

    def test_clip_level_averages_probabilities(self):
        probs = np.array([[0.9, 0.1], [0.4, 0.6], [0.4, 0.6], [0.2, 0.8]])
        truth, pred, names = clip_level(probs, np.array([0, 0, 0, 1]), ["a", "a", "a", "b"])
        self.assertEqual(names, ["a", "b"])
        self.assertEqual(truth.tolist(), [0, 1])
        # đoạn a: 2/3 cửa sổ nghiêng về lớp 1 nhưng trung bình xác suất (0.567, 0.433) chọn lớp 0 (khác bỏ phiếu đa số)
        self.assertEqual(pred.tolist(), [0, 1])


class GraphTest(unittest.TestCase):
    def test_graph_is_connected_tree_and_partition_is_normalized(self):
        self.assertTrue(np.isfinite(hop_distance(17, COCO_EDGES)).all())
        self.assertEqual(len(COCO_EDGES), 16)
        A = spatial_adjacency()
        self.assertEqual(A.shape, (3, 17, 17))
        self.assertTrue(np.allclose(A.sum(axis=0).sum(axis=0), 1.0))
        self.assertTrue(np.allclose(A[0], np.diag(np.diag(A[0]))))

    def test_bone_parents_form_tree_rooted_at_nose(self):
        parents = bone_parents()
        self.assertEqual(parents[0], 0)
        for v in range(1, 17):  # đi lên theo cha luôn tới gốc
            node, steps = v, 0
            while node != 0 and steps < 20:
                node, steps = parents[node], steps + 1
            self.assertEqual(node, 0)


class BuildWindowsTest(unittest.TestCase):
    def _skeleton_dir(self, root):
        for subject in range(1, 7):
            for action in ("walk", "fall", "stand"):
                make_sequence(f"s{subject:02d}__{action}__t1", n=48, seed=subject).save(root / f"s{subject:02d}__{action}__t1.npz")
        make_sequence("s01__jump__t1", n=48).save(root / "s01__jump__t1.npz")  # không luật nào khớp

    def test_build_split_by_person_without_leakage(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self._skeleton_dir(root / "sk")
            cfg = ActionCfg(classes=CLASSES, label_rules=RULES, group_pattern=r"^s(\d+)", window_frames=16, stride_frames=8,
                            min_track_frames=16)
            summary = build_window_dataset(root / "sk", root / "win", cfg)
            self.assertEqual(set(summary), {"train", "val", "test"})
            data = load_windows(root / "win")
            self.assertEqual(data.X.shape[1:], (16, 17, 3))
            self.assertEqual(set(np.unique(data.y)), {0, 1, 2})
            self.assertTrue(np.isfinite(data.X).all())
            subjects = {}
            for key, split in zip(data.clip_key, data.split):
                subjects.setdefault(key.split("__")[0], set()).add(split)
            self.assertTrue(all(len(v) == 1 for v in subjects.values()))  # mỗi người chỉ ở một tập
            self.assertTrue((root / "win" / "index.csv").exists())
            self.assertEqual(sum(s["windows"] for s in summary.values()), len(data.y))

    def test_no_rule_match_raises_helpful_error(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            make_sequence("zzz", n=48).save(root / "sk" / "zzz.npz")
            cfg = ActionCfg(classes=CLASSES, label_rules=RULES, window_frames=16, min_track_frames=16)
            with self.assertRaises(ValueError):
                build_window_dataset(root / "sk", root / "win", cfg)


if __name__ == "__main__":
    unittest.main()
