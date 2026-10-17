"""Kiểm thử các phần dùng PyTorch. Tự bỏ qua khi chưa cài torch; hãy chạy trên Colab:
    python -m unittest discover -s tests -p "test_models.py" -v"""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from patientmon.actions.models import MODEL_NAMES

PARAMS = {"base_channels": 16, "layers": 4, "rnn_layers": 1, "hidden": 32, "tcn_layers": 2, "dropout": 0.1}


def synthetic_windows(path, n_per_split=24, t=16, seed=0):
    """Hai lớp tách được rõ: lớp 0 đứng yên, lớp 1 cả người trượt xuống theo thời gian."""
    rng = np.random.RandomState(seed)
    X, y, split, keys = [], [], [], []
    for s, name in enumerate(("train", "val", "test")):
        for i in range(n_per_split):
            label = i % 2
            w = rng.randn(t, 17, 3).astype(np.float32) * 0.05
            w[:, :, 2] = 0.9
            if label:
                w[:, :, 1] += np.linspace(0, 2.0, t)[:, None]
            X.append(w), y.append(label), split.append(name), keys.append(f"{name}{i // 4}:1")
    np.savez_compressed(path, X=np.stack(X), y=np.array(y), split=np.array(split), clip_key=np.array(keys),
                        classes=np.array(["still", "slide"]), fps=np.float32(10.0))


@unittest.skipUnless(torch, "chưa cài torch")
class ModelShapesTest(unittest.TestCase):
    def test_every_model_outputs_logits_and_backpropagates(self):
        from patientmon.actions.models import build_model

        x = torch.randn(4, 32, 17, 3)
        for name in MODEL_NAMES:
            model = build_model(name, 5, PARAMS)
            logits = model(x)
            self.assertEqual(tuple(logits.shape), (4, 5), name)
            logits.sum().backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()), name)

    def test_stgcnpp_handles_other_window_lengths(self):
        from patientmon.actions.models import build_model

        model = build_model("stgcnpp", 3, {**PARAMS, "layers": 10, "base_channels": 8})
        for t in (16, 30, 48):
            self.assertEqual(tuple(model(torch.randn(2, t, 17, 3)).shape), (2, 3))

    def test_unknown_model_rejected(self):
        from patientmon.actions.models import build_model

        with self.assertRaises(ValueError):
            build_model("resnet", 3, PARAMS)


@unittest.skipUnless(torch, "chưa cài torch")
class StreamsAndAugmentTest(unittest.TestCase):
    def test_streams(self):
        from patientmon.actions.models.features import to_stream

        x = torch.randn(2, 8, 17, 3)
        for stream in ("joint", "bone", "motion"):
            self.assertEqual(to_stream(x, stream).shape, x.shape)
        bone = to_stream(x, "bone")
        self.assertTrue(torch.all(bone[:, :, 0, :2] == 0))  # khớp gốc không có xương
        self.assertTrue(torch.allclose(bone[:, :, 3, :2], x[:, :, 3, :2] - x[:, :, 1, :2]))  # tai trái - mắt trái
        motion = to_stream(x, "motion")
        self.assertTrue(torch.all(motion[:, -1, :, :2] == 0))
        self.assertTrue(torch.allclose(motion[:, 0, :, :2], x[:, 1, :, :2] - x[:, 0, :, :2]))
        with self.assertRaises(ValueError):
            to_stream(x, "foo")

    def test_augment_keeps_shape_confidence_and_finite(self):
        from patientmon.actions.train import augment

        x = torch.randn(16, 8, 17, 3)
        x[..., 2] = 0.9
        out = augment(x)
        self.assertEqual(out.shape, x.shape)
        self.assertTrue(torch.isfinite(out).all())
        self.assertTrue(torch.allclose(out[..., 2].sort(dim=-1).values, x[..., 2].sort(dim=-1).values))


@unittest.skipUnless(torch, "chưa cài torch")
class TrainEvaluateSmokeTest(unittest.TestCase):
    def _cfg(self, root, model):
        from patientmon.config import Config

        cfg = Config()
        cfg.paths.data_root = str(root)
        cfg.train_action.model = model
        cfg.train_action.epochs, cfg.train_action.batch, cfg.train_action.patience = 12, 16, 50
        for k, v in PARAMS.items():
            setattr(cfg.train_action, k, v)
        cfg.train_action.device = "cpu"
        return cfg

    def test_train_then_evaluate_for_each_model(self):
        from patientmon.actions.data import load_windows
        from patientmon.actions.evaluate import evaluate_checkpoint, format_confusion, format_table
        from patientmon.actions.train import train_action

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "windows").mkdir()
            synthetic_windows(root / "windows" / "windows.npz")
            results = []
            for model in ("tcn", "gru", "stgcnpp"):
                cfg = self._cfg(root, model)
                best = train_action(cfg, model_name=model)
                self.assertTrue(best.exists())
                with (best.parent / "log.csv").open(encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                self.assertGreaterEqual(len(rows), 2)
                self.assertLess(float(rows[-1]["train_loss"]), float(rows[0]["train_loss"]), model)  # loss giảm
                results.append(evaluate_checkpoint(best, load_windows(root / "windows"), "test", device="cpu"))
            table = format_table(results)
            self.assertIn("stgcnpp", table)
            self.assertIn("Recall slide", table)
            self.assertIn("slide", format_confusion(results[0]))
            # bài toán dễ: mô hình đơn giản nhất phải học được
            self.assertGreater(results[0]["windows"]["macro_f1"], 0.8)

    def test_run_directories_do_not_overwrite(self):
        from patientmon.actions.train import train_action

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "windows").mkdir()
            synthetic_windows(root / "windows" / "windows.npz")
            cfg = self._cfg(root, "gru")
            cfg.train_action.epochs = 2
            first, second = train_action(cfg), train_action(cfg)
            self.assertNotEqual(first.parent, second.parent)


if __name__ == "__main__":
    unittest.main()
