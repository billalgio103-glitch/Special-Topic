"""Rebuild twice from the same raw assets; require equal complete table contents."""
import argparse
import json
from pathlib import Path
import sqlite3
import subprocess
import sys

from download import ROOT
from verify import fingerprint
from atomic_files import write_text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    command = [sys.executable, str(ROOT / "scripts/build_db.py"), "--data-dir", str(args.data_dir)]
    if args.limit is not None:
        command += ["--limit", str(args.limit)]
    results = []
    for _ in range(2):
        subprocess.run(command, check=True)
        with sqlite3.connect(args.data_dir / "particles.db") as con:
            results.append({"fingerprint": fingerprint(con), "images": con.execute("SELECT COUNT(*) FROM images").fetchone()[0],
                            "objects": con.execute("SELECT COUNT(*) FROM objects").fetchone()[0]})
    if results[0] != results[1]:
        raise SystemExit(f"Idempotency failed: {results}")
    report = {"status": "passed", "scope": "subset" if args.limit is not None else "full", "runs": results}
    write_text(args.data_dir / "idempotency.json", json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
