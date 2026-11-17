"""Compile report/report.typ into report/report.pdf from the pipeline outputs."""
import argparse
from datetime import datetime, timezone
from pathlib import Path

import typst

from download import ROOT
from atomic_files import write_bytes


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--output", type=Path, default=ROOT / "report" / "report.pdf")
    args = p.parse_args()
    data = "/" + args.data_dir.resolve().relative_to(ROOT).as_posix()  # Typst paths are relative to ROOT
    pdf = typst.compile(str(ROOT / "report" / "report.typ"), root=str(ROOT), sys_inputs={"data": data},
                        ignore_system_fonts=True,  # embedded fonts only: same PDF on every machine
                        timestamp=int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()))
    write_bytes(args.output, pdf)
    print(f"Wrote {args.output} ({len(pdf) // 1024} KB)")


if __name__ == "__main__":
    main()
