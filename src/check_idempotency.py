"""Rebuild DB and tensors twice from the same raw assets; require identical results."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

from download import ROOT, digest
from verify import fingerprint
from atomic_files import write_text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    build = [sys.executable, str(ROOT / "src/build_db.py"), "--data-dir", str(args.data_dir)]
    export = [sys.executable, str(ROOT / "src/export_tensors.py"), "--data-dir", str(args.data_dir)]
    if args.limit is not None:
        build += ["--limit", str(args.limit)]
        export += ["--allow-subset"]
    results = []
    for _ in range(2):
        subprocess.run(build, check=True)
        subprocess.run(export, check=True)
        with closing(sqlite3.connect(args.data_dir / "particles.db")) as con:
            results.append({"fingerprint": fingerprint(con), "images": con.execute("SELECT COUNT(*) FROM images").fetchone()[0],
                            "objects": con.execute("SELECT COUNT(*) FROM objects").fetchone()[0],
                            "tensors_sha256": {name: digest(args.data_dir / "inputs" / f"{name}.safetensors")
                                               for name in ("train", "eval")}})
    if results[0] != results[1]:
        raise SystemExit(f"Idempotency failed: {results}")
    report = {"status": "passed", "scope": "subset" if args.limit is not None else "full", "runs": results}
    write_text(args.data_dir / "idempotency.json", json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
