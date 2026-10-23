"""Export -> safetensors -> torch round trip on a tiny synthetic EMPS-style database."""
import hashlib
import sys
from pathlib import Path
import sqlite3
import tempfile
import unittest

import numpy as np
from PIL import Image
from safetensors.torch import load_file
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from check_tensors import check
from export_tensors import assign_folds, downsample, export


def make_data_dir(root: Path, groups_by_split):
    """Write PNG images/masks and a schema-valid DB; one image per (group, split)."""
    raw = root / "raw" / "emps"
    raw.mkdir(parents=True)
    con = sqlite3.connect(root / "particles.db")
    con.executescript((ROOT / "schema.sql").read_text())
    con.execute("INSERT INTO sources VALUES ('emps','u','r','MIT',1)")
    rng = np.random.default_rng(0)
    for n, (group, split) in enumerate(groups_by_split):
        image = rng.integers(0, 255, (40, 60), dtype=np.uint8)
        mask = np.zeros((40, 60), dtype=np.uint16)
        mask[10:30, 10:30] = 1
        for name, array in (("i", image), ("m", mask)):
            path = raw / f"{n}{name}.png"
            Image.fromarray(array).save(path)
            con.execute("INSERT INTO assets VALUES (?,?,?,?)", (f"emps/{n}{name}.png", "u", path.stat().st_size,
                        hashlib.sha256(path.read_bytes()).hexdigest()))
        con.execute("INSERT INTO images (image_id,source_id,source_key,image_asset,mask_asset,binary_mask_path,"
                    "binary_mask_sha256,width,height,original_dtype,mask_type,object_kind,calibration_status,group_id,"
                    "split,split_note,intensity_mean,intensity_std,foreground_pixels,object_count,source_metadata_json) "
                    "VALUES (?,'emps',?,?,?,?,'x',60,40,'uint8','instance','particle_instance','none',?,?,'',0,1,400,1,'{}')",
                    (f"emps:{n:03d}", str(n), f"emps/{n}i.png", f"emps/{n}m.png", f"m{n}.png", group, split))
    con.commit()
    con.close()


class Folds(unittest.TestCase):
    rows = [{"source_id": "emps", "group_id": f"g{i % 7}"} for i in range(30)]

    def test_order_independent_and_seeded(self):
        self.assertEqual(assign_folds(self.rows), assign_folds(list(reversed(self.rows))))
        self.assertNotEqual(assign_folds(self.rows, seed=1), assign_folds(self.rows, seed=2))

    def test_every_group_one_fold_and_folds_balanced(self):
        folds = assign_folds(self.rows, k=3)
        self.assertEqual(set(folds), {f"g{i}" for i in range(7)})
        self.assertEqual(sorted(np.bincount(list(folds.values()))), [2, 2, 3])


class Downsample(unittest.TestCase):
    def test_non_square_shapes_and_binary_mask(self):
        image = np.random.default_rng(1).normal(size=(1, 300, 500)).astype(np.float32)
        mask = np.zeros((300, 500), dtype=np.uint8)
        mask[:150] = 1
        x, y = downsample(image, mask, 64)
        self.assertEqual((x.shape, x.dtype, y.shape, y.dtype), ((1, 64, 64), np.float32, (1, 64, 64), np.uint8))
        self.assertEqual(set(np.unique(y)), {0, 1})
        self.assertEqual(int(y.sum()), 32 * 64)


class Export(unittest.TestCase):
    def test_round_trip_isolation_and_byte_identical_rerun(self):
        layout = [("emps:a", "train"), ("emps:a", "train"), ("emps:b", "validation"),
                  ("emps:c", "train"), ("emps:d", "test"), ("emps:d", "test")]
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:  # Windows keeps mmaps open
            data = Path(tmp)
            make_data_dir(data, layout)
            self.assertEqual(export(data, size=16, k=2), {"train": 4, "eval": 2})
            first = {n: (data / "inputs" / f"{n}.safetensors").read_bytes() for n in ("train", "eval")}
            export(data, size=16, k=2)
            for name, payload in first.items():
                self.assertEqual(payload, (data / "inputs" / f"{name}.safetensors").read_bytes())
            report = check(data)
            self.assertEqual(report["files"]["eval"]["groups"], 1)
            t = load_file(data / "inputs" / "train.safetensors")
            x, y = next(iter(DataLoader(TensorDataset(t["images"], t["masks"]), batch_size=4)))
            self.assertEqual(tuple(x.shape), (4, 1, 16, 16))
            self.assertEqual(len(set(zip(t["group"].tolist(), t["fold"].tolist()))), 3)

    def test_group_in_two_splits_is_rejected(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:  # Windows keeps mmaps open
            data = Path(tmp)
            make_data_dir(data, [("emps:a", "train"), ("emps:a", "test")])
            with self.assertRaises(AssertionError):
                export(data, size=16)


if __name__ == "__main__":
    unittest.main()
