"""Export fixed-size, group-isolated model inputs from the SQLite database.

data/inputs/train.safetensors  DB splits train + validation, with group k-fold ids
data/inputs/eval.safetensors   DB split test (held out; never used for fitting)

Every image is z-scored per image at full resolution (common.training_pair), then
area-averaged to SIZE x SIZE. Masks are area-averaged and thresholded at 0.5.
Splits and folds are assigned per group, so no group appears in two partitions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image
from safetensors.numpy import save

from common import training_pair
from download import ROOT
from verify import fingerprint
from atomic_files import write_bytes

FORMAT_VERSION = "1"
SIZE = 256
K_FOLDS = 5
SEED = 5937
SOURCES = ("emps", "hrtem", "cobalt")
FILE_SPLITS = {"train": ("train", "validation"), "eval": ("test",)}


def downsample(image, mask, size=SIZE):
    """Return float32 image (1,size,size) and uint8 binary mask (1,size,size)."""
    small = Image.fromarray(np.ascontiguousarray(image[0], dtype=np.float32), mode="F").resize((size, size), Image.BOX)
    fraction = Image.fromarray(mask.astype(np.float32), mode="F").resize((size, size), Image.BOX)
    return (np.asarray(small, dtype=np.float32)[None],
            (np.asarray(fraction) >= 0.5).astype(np.uint8)[None])


def assign_folds(rows, k=K_FOLDS, seed=SEED):
    """Map group_id -> fold. Per source, seeded hash order, then round-robin."""
    folds = {}
    for source in SOURCES:
        groups = sorted({r["group_id"] for r in rows if r["source_id"] == source},
                        key=lambda g: hashlib.sha256(f"fold-v1:{seed}:{g}".encode()).hexdigest())
        for index, group in enumerate(groups):
            folds[group] = index % k
    return folds


def check_isolation(partitions):
    train, held_out = partitions["train"], partitions["eval"]
    assert not {r["image_id"] for r in train} & {r["image_id"] for r in held_out}, "Image in train and eval"
    assert not {r["group_id"] for r in train} & {r["group_id"] for r in held_out}, "Group in train and eval"


def build_file(rows, data_dir, folds, metadata, size=SIZE):
    groups = sorted({r["group_id"] for r in rows})
    group_index = {g: i for i, g in enumerate(groups)}
    images = np.zeros((len(rows), 1, size, size), dtype=np.float32)
    masks = np.zeros((len(rows), 1, size, size), dtype=np.uint8)
    for n, row in enumerate(rows):
        images[n], masks[n] = downsample(*training_pair(row, data_dir), size)
        if (n + 1) % 50 == 0 or n + 1 == len(rows):
            print(f"  [{n + 1}/{len(rows)}] {row['image_id']}", flush=True)
    tensors = {"images": images, "masks": masks,
               "source": np.array([SOURCES.index(r["source_id"]) for r in rows], dtype=np.int64),
               "group": np.array([group_index[r["group_id"]] for r in rows], dtype=np.int64)}
    if folds is not None:
        tensors["fold"] = np.array([folds[r["group_id"]] for r in rows], dtype=np.int64)
    metadata = dict(metadata, image_ids=[r["image_id"] for r in rows], group_ids=groups, sources=list(SOURCES))
    # One sorted-JSON metadata key keeps the file header byte-identical across runs.
    return save(tensors, metadata={"etl": json.dumps(metadata, sort_keys=True)})


def export(data_dir, size=SIZE, k=K_FOLDS, seed=SEED, allow_subset=False):
    db_path = data_dir / "particles.db"
    con = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = [dict(r) for r in con.execute(
        "SELECT image_id, source_id, group_id, split, image_asset, mask_asset, array_index "
        "FROM images ORDER BY image_id")]
    db_fingerprint = fingerprint(con)
    con.close()
    partitions = {name: [r for r in rows if r["split"] in splits] for name, splits in FILE_SPLITS.items()}
    check_isolation(partitions)
    folds = assign_folds(partitions["train"], k, seed)
    for name, part in partitions.items():
        if not part and not allow_subset:
            raise SystemExit(f"Empty {name} partition; use --allow-subset only for development subsets")
    common_meta = {"format_version": FORMAT_VERSION, "db_fingerprint": db_fingerprint, "size": size,
                   "seed": seed, "k_folds": k, "split_mapping": FILE_SPLITS}
    out = data_dir / "inputs"
    for name, part in partitions.items():
        print(f"Exporting {len(part)} images to {out / name}.safetensors", flush=True)
        payload = build_file(part, data_dir, folds if name == "train" else None, dict(common_meta, partition=name), size)
        write_bytes(out / f"{name}.safetensors", payload)
    return {name: len(part) for name, part in partitions.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--size", type=int, default=SIZE)
    p.add_argument("--k-folds", type=int, default=K_FOLDS)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--allow-subset", action="store_true")
    args = p.parse_args()
    print(json.dumps(export(args.data_dir, args.size, args.k_folds, args.seed, args.allow_subset)))


if __name__ == "__main__":
    main()
