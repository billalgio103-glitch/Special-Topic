"""Publish finished files with atomic replacement, including across filesystems."""
import errno
import os
from pathlib import Path
import shutil
import tempfile


def publish_file(temporary: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(temporary, destination)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        fd, name = tempfile.mkstemp(prefix=".em-etl-incoming-", dir=destination.parent)
        staged = Path(name)
        try:
            with os.fdopen(fd, "wb") as target, temporary.open("rb") as source:
                shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
                target.flush()
                os.fsync(target.fileno())
            os.replace(staged, destination)
            temporary.unlink()
        finally:
            staged.unlink(missing_ok=True)


def write_bytes(destination: Path, contents: bytes):
    fd, name = tempfile.mkstemp(prefix="em-etl-output-")
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as out:
            out.write(contents)
            out.flush()
            os.fsync(out.fileno())
        publish_file(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def write_text(destination: Path, contents: str):
    write_bytes(destination, contents.encode("utf-8"))
