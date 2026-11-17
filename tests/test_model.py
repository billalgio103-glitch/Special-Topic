"""Metric correctness and a tiny, repeatable end-to-end training run."""
import json
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from export_tensors import export
from model import dice, roc_auc, roc_curve
from test_tensors import make_data_dir
from train import train


def brute_force_auc(scores, labels):
    pos, neg = scores[labels == 1], scores[labels == 0]
    return float(((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :])).mean())


class Metrics(unittest.TestCase):
    def test_auc_extremes_and_ties(self):
        y = np.array([0, 0, 1, 1])
        self.assertEqual(roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), y), 1.0)
        self.assertEqual(roc_auc(np.array([0.9, 0.8, 0.2, 0.1]), y), 0.0)
        self.assertEqual(roc_auc(np.full(4, 0.3), y), 0.5)
        self.assertIsNone(roc_auc(np.ones(3), np.ones(3)))

    def test_auc_matches_pair_counting(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 300)
        s = np.round(rng.normal(size=300) + y, 1)  # rounding creates ties
        self.assertAlmostEqual(roc_auc(s, y), brute_force_auc(s, y), places=12)

    def test_roc_curve_runs_corner_to_corner(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 2, 5000)
        fpr, tpr = roc_curve(rng.normal(size=5000) + y, y, points=50)
        self.assertEqual((fpr[0], tpr[0], fpr[-1], tpr[-1]), (0, 0, 1, 1))
        self.assertLessEqual(len(fpr), 50)
        self.assertTrue(np.all(np.diff(fpr) >= 0) and np.all(np.diff(tpr) >= 0))

    def test_dice(self):
        self.assertEqual(dice(np.array([1, 1, 0]), np.array([1, 0, 0])), 2 / 3)
        self.assertIsNone(dice(np.zeros(3), np.zeros(3)))


class Training(unittest.TestCase):
    def test_controls_and_repeatable_results(self):
        layout = [("emps:a", "train"), ("emps:b", "validation"), ("emps:c", "train"),
                  ("emps:d", "test"), ("emps:e", "test")]
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:  # Windows keeps mmaps open
            data = Path(tmp)
            make_data_dir(data, layout)
            export(data, size=64, k=2)
            runs = []
            for name in ("a", "b"):
                r = train(data, data / name, epochs=2, batch_size=2, lr=1e-3, width=4, val_fold=0,
                          threads=1, log_every=1, sigma=0.5)
                runs.append((data / name / "results.json").read_text())
            self.assertEqual(runs[0], runs[1])
            self.assertEqual(r["test"]["dummy"]["auc"], 0.5)
            self.assertAlmostEqual(r["test"]["positive_control"]["auc"], r["controls"]["positive_expected_auc"], delta=0.02)
            self.assertEqual([h["epoch"] for h in json.loads(runs[0])["history"]], [1, 2])


if __name__ == "__main__":
    unittest.main()
