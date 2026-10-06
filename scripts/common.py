"""Read native assets without resizing, clipping, or discarding original precision."""
from __future__ import annotations

from pathlib import Path
import h5py
import numpy as np
from PIL import Image
from ncempy.io import dm


def load_native(sample: dict, raw_dir: Path):
    image_path = raw_dir / sample["image_asset"]
    label_path = raw_dir / sample["mask_asset"]
    dataset = sample.get("dataset", sample.get("source_id"))
    calibration = (None, None, "unavailable")
    if dataset == "emps":
        with Image.open(image_path) as im:
            image = np.asarray(im.convert("L"))
        with Image.open(label_path) as im:
            labels = np.asarray(im)
        if labels.ndim != 2 or not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f"Expected integer instance mask: {label_path}")
        calibration = (None, None, "scale_bar_requires_manual_calibration")
        mask_type, object_kind = "instance", "particle_instance"
    elif dataset == "hrtem":
        native = dm.dmReader(str(image_path))
        image = np.asarray(native["data"])
        units = native["pixelUnit"]
        factors = {"nm": 1.0, "pm": 0.001, "um": 1000.0, "µm": 1000.0,
                   "μm": 1000.0, "m": 1e9, "A": 0.1, "Å": 0.1}
        converted = [float(v) * factors[str(u)] for v, u in zip(native["pixelSize"], units)]
        if len(converted) != 2 or not all(np.isfinite(v) and v > 0 for v in converted):
            raise ValueError(f"Invalid spatial calibration: {image_path}")
        calibration = (converted[1], converted[0], "dm3_header")
        with Image.open(label_path) as im:
            mask = np.asarray(im)
        if mask.ndim != 2 or not set(np.unique(mask)).issubset({0, 1, 255}):
            raise ValueError(f"Unexpected HRTEM mask values: {label_path}")
        labels = mask > 0
        mask_type, object_kind = "binary", "connected_component"
    elif dataset == "cobalt":
        index = sample["array_index"]
        with h5py.File(image_path, "r") as f:
            image = f["images"][index]
        with h5py.File(label_path, "r") as f:
            one_hot = f["labels"][index]
        if one_hot.shape != (*image.shape, 2):
            raise ValueError(f"Unexpected one-hot shape {one_hot.shape}; image {image.shape}")
        if not np.isfinite(one_hot).all() or not np.isin(one_hot, [0, 1]).all():
            raise ValueError("Cobalt label is not one-hot binary data")
        if not np.all(one_hot.sum(axis=-1) == 1):
            raise ValueError("Cobalt one-hot channels do not sum to one")
        # The authors' notebook displays labels[i,:,:,1] as the foreground.
        labels = one_hot[..., 1] == 1
        calibration = (None, None, "range_only_67_to_86_pm_no_per_image_mapping")
        mask_type, object_kind = "one_hot", "connected_component"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    if image.ndim != 2 or image.shape != labels.shape:
        raise ValueError(f"Image/mask misalignment for {image_path}: {image.shape} != {labels.shape}")
    if not np.isfinite(image).all():
        raise ValueError(f"Nonfinite image data: {image_path}")
    return image, labels, calibration, mask_type, object_kind


def training_pair(sample: dict, data_dir: Path):
    """Return full-resolution float32 image (1,H,W) and uint8 binary target (H,W).

    Per-image z-scoring has no train/test fitting leakage. Crop/augment only after
    selecting a split; do not shuffle patches from one parent into other splits.
    """
    image, labels, *_ = load_native(sample, data_dir / "raw")
    image = image.astype(np.float32)
    std = float(image.std(dtype=np.float64))
    image = (image - float(image.mean(dtype=np.float64))) / (std if std > 0 else 1)
    return image[None, ...], (labels > 0).astype(np.uint8)
