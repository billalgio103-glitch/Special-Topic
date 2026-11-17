"""Training and held-out evaluation figures, drawn only from data/model outputs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from safetensors.numpy import load_file

from download import ROOT
from figures import save_figure

SOURCES = ("emps", "hrtem", "cobalt")
LABELS = {"unet": "U-Net", "dummy": "Negative control (dummy)", "positive_control": "Positive control (noisy truth)"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = p.parse_args()
    model_dir = args.data_dir / "model"
    out = model_dir / "figures"
    out.mkdir(parents=True, exist_ok=True)
    results = json.loads((model_dir / "results.json").read_text())

    history = results["history"]
    epochs = [h["epoch"] for h in history]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    ax1.plot(epochs, [h["train_loss"] for h in history], "o-", label="train")
    if all("val_loss" in h for h in history):
        ax1.plot(epochs, [h["val_loss"] for h in history], "o-", label="validation")
    ax1.set(xlabel="epoch", ylabel="BCE + soft Dice loss", title="Loss")
    ax1.legend()
    val_auc = [(h["epoch"], h["val_auc"]) for h in history if h.get("val_auc") is not None]
    if val_auc:
        ax2.plot(*zip(*val_auc), "o-", color="tab:green")
        ax2.axvline(results["best_epoch"], color="gray", ls="--", lw=1, label=f"selected epoch {results['best_epoch']}")
        ax2.legend()
    ax2.set(xlabel="epoch", ylabel="pixel ROC AUC", title=f"Validation fold {results['config']['val_fold']}")
    fig.tight_layout()
    save_figure(fig, out / "training_curve.png", 150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.6, 4.4))
    for name, label in LABELS.items():
        m = results["test"][name]
        ax.plot(m["roc"]["fpr"], m["roc"]["tpr"], label=f"{label}: AUC {m['auc']:.3f}")
    ax.plot([0, 1], [0, 1], color="lightgray", lw=1)
    ax.set(xlabel="false positive rate", ylabel="true positive rate", title="Held-out test pixels",
           xlim=(0, 1), ylim=(0, 1), aspect="equal")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    save_figure(fig, out / "roc.png", 150)
    plt.close(fig)

    ex = load_file(model_dir / "examples.safetensors")
    n = len(ex["images"])
    fig, axes = plt.subplots(n, 3, figsize=(6.6, 2.2 * n), squeeze=False)
    for i in range(n):
        for ax, image, title in zip(axes[i], (ex["images"][i, 0], ex["masks"][i, 0], ex["probs"][i, 0]),
                                    ("input", "true mask", "predicted probability")):
            ax.imshow(image, cmap="gray", vmin=0 if title != "input" else None, vmax=1 if title != "input" else None)
            ax.set_title(f"{SOURCES[ex['source'][i]]}: {title}" if title == "input" else title, fontsize=8)
            ax.axis("off")
    fig.tight_layout()
    save_figure(fig, out / "examples.png", 120)
    plt.close(fig)
    print(f"Wrote figures to {out}")


if __name__ == "__main__":
    main()
