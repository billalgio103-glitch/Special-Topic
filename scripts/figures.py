"""Preliminary descriptive figures from SQLite, plus source/mask QC examples."""
import argparse
import io
from pathlib import Path
import sqlite3

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import load_native
from download import ROOT
from atomic_files import write_bytes


def save_figure(fig, path, dpi):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi)
    write_bytes(path, buffer.getvalue())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = p.parse_args()
    out = args.data_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(args.data_dir / "particles.db")
    con.row_factory = sqlite3.Row
    sources = [r[0] for r in con.execute("SELECT DISTINCT source_id FROM images ORDER BY source_id")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for j, split in enumerate(["train", "validation", "test"]):
        values = [con.execute("SELECT COUNT(*) FROM images WHERE source_id=? AND split=?", (s, split)).fetchone()[0] for s in sources]
        axes[0].bar(np.arange(len(sources)) + (j - 1) * .25, values, .25, label=split)
    axes[0].set(xticks=np.arange(len(sources)), xticklabels=sources, ylabel="Images", title="Source and grouped split counts")
    axes[0].legend()
    for s in sources:
        fractions = [r[0] for r in con.execute("SELECT 1.0*foreground_pixels/(width*height) FROM images WHERE source_id=?", (s,))]
        axes[1].hist(fractions, bins=np.linspace(0, 1, 21), histtype="step", linewidth=2, label=s)
    axes[1].set(xlabel="Foreground area fraction", ylabel="Images", title="Foreground coverage")
    axes[1].legend()
    save_figure(fig, out / "dataset_overview.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(len(sources), 2, figsize=(8, 4 * len(sources)), squeeze=False, constrained_layout=True)
    for row, source in enumerate(sources):
        sample = dict(con.execute("SELECT * FROM images WHERE source_id=? ORDER BY image_id LIMIT 1", (source,)).fetchone())
        image, labels, *_ = load_native(sample, args.data_dir / "raw")
        lo, hi = np.percentile(image, [1, 99])
        axes[row, 0].imshow(image, cmap="gray", vmin=lo, vmax=hi)
        axes[row, 0].set_title(source + ": input (display contrast only)")
        axes[row, 1].imshow(image, cmap="gray", vmin=lo, vmax=hi)
        axes[row, 1].contour(labels > 0, levels=[.5], colors=["#ff542e"], linewidths=.5)
        axes[row, 1].set_title("Foreground boundary overlay")
        for axis in axes[row]:
            axis.axis("off")
    save_figure(fig, out / "mask_alignment_qc.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(sources), figsize=(12, 4), squeeze=False, constrained_layout=True)
    for axis, source in zip(axes[0], sources):
        values = [r[0] for r in con.execute("SELECT equivalent_diameter_px FROM size_measurements WHERE source_id=?", (source,))]
        if values:
            axis.hist(values, bins=40, color="#23678a")
            axis.set_yscale("log")
        kind = "Particle instances" if source == "emps" else "Connected components"
        axis.set(title=f"{source}: {kind}", xlabel="Equivalent diameter (pixels)", ylabel="Objects (log scale)")
    fig.suptitle("Border objects excluded; pixels are not comparable physical sizes across sources")
    save_figure(fig, out / "object_sizes_pixels.png", dpi=180)
    plt.close(fig)
    con.close()
    print(f"Wrote figures to {out}")


if __name__ == "__main__":
    main()
