"""Check source completeness, raw integrity, split isolation, and database consistency."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image

from download import ROOT, digest
from atomic_files import write_text


def fingerprint(con):
    h = hashlib.sha256()
    for table, order in [("sources", "source_id"), ("assets", "asset_path"),
                         ("images", "image_id"), ("objects", "image_id,label_id")]:
        h.update(table.encode())
        for row in con.execute(f"SELECT * FROM {table} ORDER BY {order}"):
            h.update(json.dumps(list(row), separators=(",", ":"), ensure_ascii=True).encode())
            h.update(b"\n")
    return h.hexdigest()


def verify(data_dir, allow_subset=False):
    path = data_dir / "particles.db"
    con = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    assert con.execute("PRAGMA integrity_check").fetchone()[0] == "ok", "SQLite integrity failure"
    assert not con.execute("PRAGMA foreign_key_check").fetchall(), "Broken foreign key"
    counts = dict(con.execute("SELECT source_id,COUNT(*) FROM images GROUP BY source_id"))
    expected = dict(con.execute("SELECT source_id,expected_samples FROM sources"))
    if not allow_subset:
        assert counts == expected, f"Incomplete dataset: {counts}; expected {expected}"
    assert not con.execute("SELECT group_id FROM images GROUP BY group_id HAVING COUNT(DISTINCT split)>1").fetchall(), "Group leakage"
    assert not con.execute("SELECT a.sha256 FROM images i JOIN assets a ON a.asset_path=i.image_asset WHERE i.array_index IS NULL GROUP BY a.sha256 HAVING COUNT(DISTINCT i.split)>1").fetchall(), "Duplicate image leakage"
    for row in con.execute("SELECT * FROM assets ORDER BY asset_path"):
        raw = data_dir / "raw" / row["asset_path"]
        assert raw.stat().st_size == row["bytes"], f"Raw size mismatch: {raw}"
        assert digest(raw) == row["sha256"], f"Raw checksum mismatch: {raw}"
    for row in con.execute("SELECT * FROM images ORDER BY image_id"):
        path = data_dir / row["binary_mask_path"]
        assert digest(path) == row["binary_mask_sha256"], f"Mask checksum mismatch: {path}"
        with Image.open(path) as im:
            mask = np.asarray(im)
        assert mask.shape == (row["height"], row["width"]), f"Mask shape mismatch: {path}"
        assert set(np.unique(mask)).issubset({0, 255}), f"Nonbinary normalized mask: {path}"
        assert int(np.count_nonzero(mask)) == row["foreground_pixels"], "Wrong foreground count"
        count, area = con.execute("SELECT COUNT(*),COALESCE(SUM(area_px),0) FROM objects WHERE image_id=?", (row["image_id"],)).fetchone()
        assert count == row["object_count"] and area == row["foreground_pixels"], "Object measurements inconsistent"
    report = {"status": "passed", "scope": "subset" if allow_subset else "full",
              "source_counts": counts, "expected_counts": expected,
              "database_fingerprint": fingerprint(con),
              "objects": con.execute("SELECT COUNT(*) FROM objects").fetchone()[0],
              "calibration_counts": [dict(r) for r in con.execute("SELECT source_id,calibration_status,COUNT(*) AS n FROM images GROUP BY source_id,calibration_status")],
              "splits": [dict(r) for r in con.execute("SELECT * FROM dataset_summary ORDER BY source_id,split")]}
    con.close()
    write_text(data_dir / "validation.json", json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--allow-subset", action="store_true")
    args = p.parse_args()
    print(json.dumps(verify(args.data_dir, args.allow_subset), indent=2))


if __name__ == "__main__":
    main()
