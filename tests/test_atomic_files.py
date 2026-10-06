"""Protect completed outputs when the container and bind mount use different devices."""
import errno
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import atomic_files


class AtomicPublication(unittest.TestCase):
    def test_cross_device_publish_replaces_complete_content(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source", root / "target"
            source.write_bytes(b"complete new data")
            target.write_bytes(b"previous data")
            original = atomic_files.os.replace
            calls = 0

            def cross_device_once(a, b):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise OSError(errno.EXDEV, "different filesystems")
                return original(a, b)

            with patch.object(atomic_files.os, "replace", side_effect=cross_device_once):
                atomic_files.publish_file(source, target)
            self.assertEqual(target.read_bytes(), b"complete new data")
            self.assertFalse(source.exists())
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["target"])

    def test_failed_cross_device_copy_preserves_previous_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, target = root / "source", root / "target"
            source.write_bytes(b"new data")
            target.write_bytes(b"previous data")
            with patch.object(atomic_files.os, "replace", side_effect=OSError(errno.EXDEV, "different filesystems")), \
                 patch.object(atomic_files.shutil, "copyfileobj", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    atomic_files.publish_file(source, target)
            self.assertEqual(target.read_bytes(), b"previous data")
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["source", "target"])
