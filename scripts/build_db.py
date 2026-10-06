"""Build a complete SQLite database atomically from verified native assets."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile

import numpy as np
from PIL import Image
from scipy import ndimage

from common import load_native
from download import ROOT, digest, selected_samples, valid


def split_groups(samples):
    splits = {}
    for source in ("emps", "hrtem"):
        groups = sorted({s["group_id"] for s in samples if s["dataset"] == source},
                        key=lambda g: hashlib.sha256(("split-v1:" + g).encode()).hexdigest())
        n = len(groups)
        n_test = max(1, round(0.15 * n))
        n_val = max(1, round(0.15 * n))
        for index, group in enumerate(groups):
            splits[group] = "test" if index < n_test else "validation" if index < n_test + n_val else "train"
    # Cobalt contains patches with no published parent-image mapping. Holding its
    # entire source in train avoids claiming independence between related crops.
    splits["cobalt:unresolved-parent"] = "train"
    return splits


def object_measurements(labels, instance, pixel_x, pixel_y):
    if instance:
        numbered = labels.astype(np.int32)
    else:
        numbered, _ = ndimage.label(labels > 0, structure=np.ones((3, 3), dtype=np.uint8))
    counts = np.bincount(numbered.ravel())
    ids = np.flatnonzero(counts[1:]) + 1
    border_ids = set(np.concatenate([numbered[0], numbered[-1], numbered[:, 0], numbered[:, -1]]).tolist())
    for label_id in ids:
        area = int(counts[label_id])
        area_nm = area * pixel_x * pixel_y if pixel_x is not None and pixel_y is not None else None
        yield (int(label_id), area, area_nm, math.sqrt(4 * area / math.pi),
               math.sqrt(4 * area_nm / math.pi) if area_nm is not None else None,
               int(label_id in border_ids))


def write_exports(connection, data_dir):
    exports = data_dir / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    queries = {
        "dataset_summary": "SELECT * FROM dataset_summary ORDER BY source_id, split",
        "size_measurements": "SELECT * FROM size_measurements ORDER BY image_id,label_id",
        "training_manifest": "SELECT image_id,source_id,image_asset,mask_asset,array_index,binary_mask_path,group_id,split,mask_type,pixel_size_x_nm,pixel_size_y_nm,calibration_status FROM images ORDER BY image_id"
    }
    for name, query in queries.items():
        cur = connection.execute(query)
        with (exports / f"{name}.csv").open("w", newline="") as out:
            writer = csv.writer(out)
            writer.writerow([c[0] for c in cur.description])
            writer.writerows(cur)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "sources.lock.json").read_text())
    samples = selected_samples(manifest, args.limit)
    required = {s[k] for s in samples for k in ("image_asset", "mask_asset")}
    assets = [a for a in manifest["assets"] if a["path"] in required]
    splits = split_groups(manifest["samples"])
    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "processed/masks").mkdir(parents=True, exist_ok=True)
    db_path = data_dir / "particles.db"
    # Build outside a cloud-synchronized project directory: a background sync
    # replacing an actively written SQLite inode can invalidate its indexes.
    fd, temporary_name = tempfile.mkstemp(prefix="em-etl-db-", suffix=".db")
    os.close(fd)
    temporary = Path(temporary_name)
    con = sqlite3.connect(temporary)
    try:
        con.executescript((ROOT / "schema.sql").read_text())
        for source in manifest["sources"]:
            con.execute("INSERT INTO sources VALUES (?,?,?,?,?)", (source["id"], source["url"], source["revision"], source["license"], source["expected_samples"]))
        print(f"Verifying {len(assets)} raw files", flush=True)
        for a in assets:
            path = data_dir / "raw" / a["path"]
            if not valid(path, a):
                raise ValueError(f"Missing or corrupted raw asset: {path}; run download.py")
            con.execute("INSERT INTO assets VALUES (?,?,?,?)", (a["path"], a["url"], path.stat().st_size, digest(path)))
        for n, sample in enumerate(samples, 1):
            image, labels, (px, py, status), mask_type, kind = load_native(sample, data_dir / "raw")
            image_id = sample["id"]
            mask_rel = "processed/masks/" + hashlib.sha256(image_id.encode()).hexdigest()[:24] + ".png"
            mask_path = data_dir / mask_rel
            Image.fromarray((labels > 0).astype(np.uint8) * 255).save(mask_path)
            objects = list(object_measurements(labels, kind == "particle_instance", px, py))
            meta = sample["metadata"]
            split_note = ("All cobalt patches kept in train: parent mapping unavailable" if sample["dataset"] == "cobalt"
                          else "Grouped by source paper DOI" if sample["dataset"] == "emps"
                          else "Grouped by acquisition folder")
            values = {
                "image_id": image_id, "source_id": sample["dataset"], "source_key": sample["source_key"],
                "image_asset": sample["image_asset"], "mask_asset": sample["mask_asset"], "array_index": sample.get("array_index"),
                "binary_mask_path": mask_rel, "binary_mask_sha256": digest(mask_path),
                "width": image.shape[1], "height": image.shape[0], "original_dtype": str(image.dtype),
                "mask_type": mask_type, "object_kind": kind, "pixel_size_x_nm": px, "pixel_size_y_nm": py,
                "calibration_status": status, "group_id": sample["group_id"], "split": splits[sample["group_id"]],
                "split_note": split_note, "material": meta.get("Material", meta.get("material")),
                "nominal_particle_size_nm": float(meta["Nanoparticle Size (nm)"]) if meta.get("Nanoparticle Size (nm)") else None,
                "intensity_mean": float(image.mean(dtype=np.float64)), "intensity_std": float(image.std(dtype=np.float64)),
                "foreground_pixels": int(np.count_nonzero(labels)), "object_count": len(objects),
                "source_metadata_json": json.dumps(meta, sort_keys=True)
            }
            columns = ",".join(values)
            placeholders = ",".join("?" for _ in values)
            con.execute(f"INSERT INTO images ({columns}) VALUES ({placeholders})", tuple(values.values()))
            con.executemany("INSERT INTO objects VALUES (?,?,?,?,?,?,?)", [(image_id, *o) for o in objects])
            if n % 20 == 0 or n == len(samples):
                print(f"[{n}/{len(samples)}] parsed {image_id}", flush=True)
        con.commit()
        if con.execute("PRAGMA integrity_check").fetchone()[0] != "ok" or con.execute("PRAGMA foreign_key_check").fetchall():
            raise ValueError("Database integrity check failed")
        if args.limit is None:
            for source in manifest["sources"]:
                count = con.execute("SELECT COUNT(*) FROM images WHERE source_id=?", (source["id"],)).fetchone()[0]
                if count != source["expected_samples"]:
                    raise ValueError(f"Incomplete source {source['id']}: {count}")
        write_exports(con, data_dir)
        con.close()
        incoming = data_dir / "particles.db.partial"
        shutil.move(str(temporary), str(incoming))
        incoming.replace(db_path)
        # Check the published file as well as the still-open working database.
        with sqlite3.connect(db_path) as published:
            if published.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise ValueError("Published database failed integrity check")
            if published.execute("SELECT COUNT(*) FROM images").fetchone()[0] != len(samples):
                raise ValueError("Published database lost rows")
    except BaseException:
        con.close()
        temporary.unlink(missing_ok=True)
        raise
    print(f"Created {db_path} with {len(samples)} images", flush=True)


if __name__ == "__main__":
    main()
