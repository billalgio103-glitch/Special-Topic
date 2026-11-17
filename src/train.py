"""Train the U-Net on train.safetensors, then score the held-out eval.safetensors.

Validation during training uses one group fold of the training file (`fold`), so
the held-out eval file is touched exactly once, after training. The test metric is
pixel ROC AUC, reported next to two controls:

negative  "dummy" model predicting the training foreground fraction everywhere;
          it cannot rank pixels, so its AUC must be exactly 0.5.
positive  the true mask plus seeded Gaussian noise (sigma); its expected AUC is
          Phi(1 / (sigma * sqrt(2))), about 0.921 for sigma = 0.5.

If either control misses its expected value, the metric code is wrong and the
run fails instead of writing a report.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time

import numpy as np
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from download import ROOT, digest
from atomic_files import write_text
from model import UNet, dice, roc_auc, roc_curve

SEED = 5937
SOURCES = ("emps", "hrtem", "cobalt")


def read(path):
    with safe_open(path, framework="pt") as f:
        meta = json.loads(f.metadata()["etl"])
    return load_file(path), meta


def loss_fn(logits, target):
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    soft_dice = 1 - (2 * (p * target).sum() + 1) / (p.sum() + target.sum() + 1)
    return bce + soft_dice


@torch.no_grad()
def predict(model, images, batch_size):
    model.eval()
    return torch.cat([torch.sigmoid(model(images[i:i + batch_size])) for i in range(0, len(images), batch_size)])


def sparkline(values):
    bars = " ▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    return "".join(bars[1 + round(7 * (v - lo) / (hi - lo)) if hi > lo else 4] for v in values)


def train(data_dir, out_dir, epochs, batch_size, lr, width, val_fold, threads, log_every, sigma, seed=SEED):
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    inputs = data_dir / "inputs"
    train_t, train_meta = read(inputs / "train.safetensors")
    test_t, test_meta = read(inputs / "eval.safetensors")
    assert train_meta["db_fingerprint"] == test_meta["db_fingerprint"], "train/eval come from different databases"
    is_val = train_t["fold"] == val_fold
    x_fit, y_fit = train_t["images"][~is_val], train_t["masks"][~is_val].float()
    x_val, y_val = train_t["images"][is_val], train_t["masks"][is_val].float()
    x_test, y_test = test_t["images"], test_t["masks"].float()
    assert len(x_fit) and len(x_test), "Need training and held-out images"
    print(f"fit {len(x_fit)} | validation (fold {val_fold}) {len(x_val)} | held-out test {len(x_test)}", flush=True)

    loader = DataLoader(TensorDataset(x_fit, y_fit), batch_size=batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    model = UNet(width)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    history, best = [], {"auc": -1.0, "epoch": None, "state": None}
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for step, (x, y) in enumerate(loader, 1):
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if step % log_every == 0 or step == len(loader):
                print(f"epoch {epoch}/{epochs} step {step}/{len(loader)} loss {loss.item():.4f}", flush=True)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses))}
        if len(x_val):
            p_val = predict(model, x_val, batch_size)
            row["val_loss"] = float(loss_fn(torch.logit(p_val.clamp(1e-6, 1 - 1e-6)), y_val))
            row["val_auc"] = roc_auc(p_val.numpy(), y_val.numpy())
        history.append(row)
        score = row.get("val_auc") if row.get("val_auc") is not None else epoch  # no validation: keep last epoch
        if score > best["auc"]:
            best = {"auc": score, "epoch": epoch, "state": copy.deepcopy(model.state_dict())}
        print(f"== epoch {epoch}: train loss {row['train_loss']:.4f}"
              + (f" | val loss {row['val_loss']:.4f} | val AUC {row['val_auc']:.4f}" if row.get("val_auc") is not None else "")
              + f" | {time.monotonic() - started:.0f}s", flush=True)
    print("train loss curve: " + sparkline([h["train_loss"] for h in history]), flush=True)
    val_aucs = [h["val_auc"] for h in history if h.get("val_auc") is not None]
    if val_aucs:
        print("val AUC curve:    " + sparkline(val_aucs), flush=True)

    model.load_state_dict(best["state"])
    p_test = predict(model, x_test, batch_size).numpy()
    labels = y_test.numpy()
    prior = float(y_fit.mean())
    dummy = np.full_like(p_test, prior)
    positive = labels + np.random.default_rng(seed).normal(0, sigma, labels.shape).astype(np.float32)
    expected_positive = 0.5 * (1 + math.erf(1 / (sigma * math.sqrt(2)) / math.sqrt(2)))

    def evaluate(scores):
        result = {"auc": roc_auc(scores, labels), "roc": dict(zip(("fpr", "tpr"), roc_curve(scores, labels))),
                  "per_source_auc": {}}
        for i, source in enumerate(SOURCES):
            rows = (test_t["source"] == i).numpy()
            if rows.any():
                result["per_source_auc"][source] = roc_auc(scores[rows], labels[rows])
        return result

    models = {"unet": evaluate(p_test), "dummy": evaluate(dummy), "positive_control": evaluate(positive)}
    models["unet"]["dice_at_0.5"] = dice(p_test >= 0.5, labels)
    if abs(models["dummy"]["auc"] - 0.5) > 1e-9:
        raise SystemExit(f"Negative control AUC {models['dummy']['auc']} != 0.5: metric is broken")
    if abs(models["positive_control"]["auc"] - expected_positive) > 0.02:
        raise SystemExit(f"Positive control AUC {models['positive_control']['auc']} far from {expected_positive:.3f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, out_dir / "model.safetensors")
    # A few held-out examples per source for the report's qualitative figure.
    examples = [int(i) for s in range(len(SOURCES)) for i in torch.nonzero(test_t["source"] == s).flatten()[:2]]
    save_file({"images": x_test[examples].contiguous(), "masks": y_test[examples].contiguous(),
               "probs": torch.from_numpy(p_test[examples]).contiguous(),
               "source": test_t["source"][examples].contiguous()}, out_dir / "examples.safetensors")
    results = {
        "inputs": {name: {"sha256": digest(inputs / f"{name}.safetensors"), "images": n}
                   for name, n in (("train", len(train_t["images"])), ("eval", len(x_test)))},
        "db_fingerprint": train_meta["db_fingerprint"],
        "config": {"seed": seed, "epochs": epochs, "batch_size": batch_size, "lr": lr, "width": width,
                   "val_fold": val_fold, "threads": threads, "loss": "BCE + soft Dice",
                   "torch": torch.__version__, "image_size": train_meta["size"]},
        "split_sizes": {"fit": len(x_fit), "validation": len(x_val), "test": len(x_test)},
        "train_foreground_fraction": prior,
        "best_epoch": best["epoch"],
        "history": history,
        "test": models,
        "controls": {"negative": "constant training foreground fraction; AUC must be 0.5",
                     "positive": f"true mask + N(0, {sigma}^2) noise; expected AUC {expected_positive:.4f}",
                     "positive_expected_auc": expected_positive, "sigma": sigma},
    }
    write_text(out_dir / "results.json", json.dumps(results, indent=2) + "\n")
    print(f"held-out test pixel AUC  unet {models['unet']['auc']:.4f} | dummy {models['dummy']['auc']:.4f}"
          f" | positive control {models['positive_control']['auc']:.4f}", flush=True)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--width", type=int, default=16)
    p.add_argument("--val-fold", type=int, default=1)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--sigma", type=float, default=0.5)
    a = p.parse_args()
    train(a.data_dir, a.out_dir or a.data_dir / "model", a.epochs, a.batch_size, a.lr, a.width, a.val_fold,
          a.threads, a.log_every, a.sigma)


if __name__ == "__main__":
    main()
