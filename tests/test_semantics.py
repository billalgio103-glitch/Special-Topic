"""Small tests for scientific errors that can otherwise yield plausible-looking figures."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_db import object_measurements, split_groups


class MaskSemantics(unittest.TestCase):
    def test_touching_instances_are_not_merged(self):
        labels = np.array([[0, 0, 0, 0], [0, 1, 2, 0], [0, 0, 0, 0]], dtype=np.uint16)
        particles = list(object_measurements(labels, True, None, None))
        components = list(object_measurements(labels, False, None, None))
        self.assertEqual(len(particles), 2)
        self.assertEqual(len(components), 1)
        self.assertEqual(components[0][1], 2)

    def test_anisotropic_calibration_and_border(self):
        labels = np.array([[1, 1, 0], [0, 0, 0], [0, 0, 0]])
        obj = list(object_measurements(labels, True, .02, .03))[0]
        self.assertAlmostEqual(obj[2], 2 * .02 * .03)
        self.assertEqual(obj[-1], 1)

    def test_missing_calibration_is_not_invented(self):
        obj = list(object_measurements(np.array([[1]]), True, None, None))[0]
        self.assertIsNone(obj[2])
        self.assertIsNone(obj[4])

    def test_sparse_instance_ids(self):
        result = list(object_measurements(np.array([[0, 100, 65535]]), True, None, None))
        self.assertEqual([r[0] for r in result], [100, 65535])

    def test_empty_foreground(self):
        self.assertEqual(list(object_measurements(np.zeros((3, 3), dtype=int), False, None, None)), [])

    def test_splits_are_order_independent_and_grouped(self):
        samples = [{"dataset": "emps", "group_id": f"paper:{i}"} for i in range(20)]
        samples.append({"dataset": "emps", "group_id": "paper:0"})
        self.assertEqual(split_groups(samples), split_groups(list(reversed(samples))))
        self.assertEqual(split_groups(samples)["cobalt:unresolved-parent"], "train")
        self.assertEqual(set(split_groups(samples).values()), {"train", "validation", "test"})


if __name__ == "__main__":
    unittest.main()
