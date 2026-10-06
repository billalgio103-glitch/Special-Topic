"""Download exactly the files in the source lock, checking content before reuse."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path, algorithm: str = "sha256") -> str:
    if algorithm == "git-blob-sha1":
        h = hashlib.sha1()
        h.update(f"blob {path.stat().st_size}\0".encode())
    else:
        h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def valid(path: Path, item: dict) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    if item.get("bytes") is not None and path.stat().st_size != item["bytes"]:
        return False
    if item.get("checksum"):
        return digest(path, item["checksum_type"]) == item["checksum"]
    receipt = path.with_suffix(path.suffix + ".sha256")
    return receipt.is_file() and digest(path) == receipt.read_text().strip()


def download(item: dict, data_dir: Path) -> tuple[str, str]:
    path = data_dir / "raw" / item["path"]
    if valid(path, item):
        return item["path"], "cached"
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        descriptor, temporary_name = tempfile.mkstemp(prefix="em-etl-", suffix=".partial")
        os.close(descriptor)
        partial = Path(temporary_name)
        try:
            req = urllib.request.Request(item["url"], headers={"User-Agent": "Special-Topic-ETL/1.0"})
            with urllib.request.urlopen(req, timeout=180) as response, partial.open("wb") as out:
                expected = response.headers.get("Content-Length")
                for block in iter(lambda: response.read(4 * 1024 * 1024), b""):
                    out.write(block)
            if expected and partial.stat().st_size != int(expected):
                raise ValueError(f"Incomplete transfer: {item['path']}")
            if item.get("bytes") is not None and partial.stat().st_size != item["bytes"]:
                raise ValueError(f"Wrong byte count: {item['path']}")
            if item.get("checksum") and digest(partial, item["checksum_type"]) != item["checksum"]:
                raise ValueError(f"Checksum mismatch: {item['path']}")
            checksum = digest(partial)
            # Stream into the OS temporary directory, then publish once. This
            # avoids syncing incomplete multi-GB files in cloud workspaces.
            local_partial = path.with_suffix(path.suffix + ".partial")
            shutil.move(str(partial), str(local_partial))
            local_partial.replace(path)
            path.with_suffix(path.suffix + ".sha256").write_text(checksum + "\n")
            return item["path"], "downloaded"
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("unreachable")


def selected_samples(manifest: dict, limit: int | None) -> list[dict]:
    samples = manifest["samples"]
    if limit is None:
        return samples
    result = []
    for dataset in ("emps", "hrtem", "cobalt"):
        result.extend([s for s in samples if s["dataset"] == dataset][:limit])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int, help="Development only: first N samples per source")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "sources.lock.json").read_text())
    if any(not a.get("checksum") or not a.get("bytes") for a in manifest["assets"]):
        raise SystemExit("Source lock is incomplete: every asset needs a checksum and byte count")
    samples = selected_samples(manifest, args.limit)
    required = {s[k] for s in samples for k in ("image_asset", "mask_asset")}
    assets = [a for a in manifest["assets"] if a["path"] in required]
    args.data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checking/downloading {len(assets)} assets for {len(samples)} samples", flush=True)
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, a, args.data_dir): a for a in assets}
        for n, future in enumerate(as_completed(futures), 1):
            try:
                name, status = future.result()
                if n % 25 == 0 or status == "downloaded" and not name.endswith(".png"):
                    print(f"[{n}/{len(assets)}] {status}: {name}", flush=True)
            except Exception as exc:
                errors.append(f"{futures[future]['path']}: {exc}")
                print("ERROR " + errors[-1], flush=True)
    if errors:
        raise SystemExit("Download failed; no partial database will be built.\n" + "\n".join(errors))
    print("All selected assets verified.", flush=True)


if __name__ == "__main__":
    main()
