"""Small U-Net for binary particle segmentation, plus pixel-level ROC metrics."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
import torch
from torch import nn


def block(c_in, c_out):
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True),
                         nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class UNet(nn.Module):
    """Three-level U-Net. Nearest-neighbour upsampling keeps CPU training deterministic."""

    def __init__(self, width=16):
        super().__init__()
        w = width
        self.enc1, self.enc2, self.mid = block(1, w), block(w, 2 * w), block(2 * w, 4 * w)
        self.dec2, self.dec1 = block(6 * w, 2 * w), block(3 * w, w)
        self.pool, self.up = nn.MaxPool2d(2), nn.Upsample(scale_factor=2, mode="nearest")
        self.head = nn.Conv2d(w, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        m = self.mid(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up(m), e2], 1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], 1))
        return self.head(d1)  # logits


def roc_auc(scores, labels):
    """Exact ROC AUC (Mann-Whitney U with average ranks for ties)."""
    scores, labels = np.ravel(scores), np.ravel(labels).astype(bool)
    n_pos = int(labels.sum())
    n_neg = labels.size - n_pos
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = rankdata(scores)
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def roc_curve(scores, labels, points=201):
    """FPR/TPR at every distinct threshold, thinned to at most `points` for plotting."""
    scores, labels = np.ravel(scores), np.ravel(labels).astype(bool)
    order = np.argsort(-scores, kind="stable")
    s, y = scores[order], labels[order]
    last = np.r_[np.flatnonzero(np.diff(s)), s.size - 1]  # last index of each distinct score
    tpr = np.r_[0.0, np.cumsum(y)[last] / max(y.sum(), 1)]
    fpr = np.r_[0.0, np.cumsum(~y)[last] / max((~y).sum(), 1)]
    keep = np.unique(np.linspace(0, fpr.size - 1, min(points, fpr.size)).round().astype(int))
    return [round(float(v), 6) for v in fpr[keep]], [round(float(v), 6) for v in tpr[keep]]


def dice(binary, labels):
    binary, labels = np.ravel(binary).astype(bool), np.ravel(labels).astype(bool)
    total = binary.sum() + labels.sum()
    return float(2 * (binary & labels).sum() / total) if total else None
