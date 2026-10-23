"""Load the exported safetensors exactly as a model would and re-check isolation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3

from safetensors import safe_open
from safetensors.torch import load_file
import torch
from torch.utils.data import DataLoader, TensorDataset

from download import ROOT, digest
from export_tensors import FILE_SPLITS, SOURCES
from verify import fingerprint
from atomic_files import write_text


def read(path):
    with safe_open(path, framework="pt") as f:
        meta = json.loads(f.metadata()["etl"])
    return load_file(path), meta


def check(data_dir, allow_subset=False):
    con = sqlite3.connect(f"file:{(data_dir / 'particles.db').resolve()}?mode=ro", uri=True)
    db = {r[0]: (r[1], r[2], r[3]) for r in con.execute("SELECT image_id, source_id, group_id, split FROM images ORDER BY image_id")}
    db_fingerprint = fingerprint(con)
    con.close()
    report, groups = {"status": "passed", "files": {}}, {}
    for name, splits in FILE_SPLITS.items():
        path = data_dir / "inputs" / f"{name}.safetensors"
        t, meta = read(path)
        n, size = len(meta["image_ids"]), meta["size"]
        assert meta["db_fingerprint"] == db_fingerprint, f"{path} was exported from a different database"
        assert t["images"].shape == (n, 1, size, size) and t["images"].dtype == torch.float32, "Bad image tensor"
        assert t["masks"].shape == (n, 1, size, size) and t["masks"].dtype == torch.uint8, "Bad mask tensor"
        assert torch.isfinite(t["images"]).all(), "Nonfinite image values"
        assert set(t["masks"].unique().tolist()) <= {0, 1}, "Nonbinary mask"
        expected = [i for i, (_, _, split) in db.items() if split in splits]
        assert meta["image_ids"] == expected, f"{name} rows differ from DB splits {splits}"
        # Index tensors must be complete and in range; negative ids would wrap in Python indexing.
        bounds = {"source": len(SOURCES), "group": len(meta["group_ids"])}
        if name == "train":
            bounds["fold"] = meta["k_folds"]
        else:
            assert "fold" not in t, "Fold ids belong to the training file only"
        for key, upper in bounds.items():
            assert key in t, f"Missing {key} tensor in {path}"
            assert t[key].shape == (n,) and t[key].dtype == torch.int64, f"Bad {key} tensor in {path}"
            assert ((t[key] >= 0) & (t[key] < upper)).all(), f"{key} id out of range in {path}"
        for i, image_id in enumerate(meta["image_ids"]):
            source, group, _ = db[image_id]
            assert SOURCES[t["source"][i]] == source, f"Source mismatch: {image_id}"
            assert meta["group_ids"][t["group"][i]] == group, f"Group mismatch: {image_id}"
        if "fold" in t:
            fold_sets = {}
            for g, f in zip(t["group"].tolist(), t["fold"].tolist()):
                fold_sets.setdefault(g, set()).add(f)
            assert all(len(s) == 1 for s in fold_sets.values()), "Group split across folds"
        groups[name] = set(meta["group_ids"])
        if n:
            x, y = next(iter(DataLoader(TensorDataset(t["images"], t["masks"]), batch_size=8, shuffle=False)))
            assert x.shape[1:] == (1, size, size) and y.shape == x.shape, "DataLoader batch shape"
        elif not allow_subset:
            raise AssertionError(f"Empty {name} partition")
        report["files"][name] = {
            "path": path.relative_to(data_dir).as_posix(), "sha256": digest(path), "images": n,
            "per_source": {s: int((t["source"] == i).sum()) for i, s in enumerate(SOURCES)},
            "groups": len(meta["group_ids"]),
            "per_fold": {str(f): int((t["fold"] == f).sum()) for f in range(meta["k_folds"])} if "fold" in t else None}
    assert not groups["train"] & groups["eval"], "Group leakage between train and eval"
    report["db_fingerprint"] = db_fingerprint
    write_text(data_dir / "inputs" / "tensors_validation.json", json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--allow-subset", action="store_true")
    args = p.parse_args()
    print(json.dumps(check(args.data_dir, args.allow_subset), indent=2))


if __name__ == "__main__":
    main()
