import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from atomic_artifacts import ArtifactWriteError, write_json


class AtomicArtifactTests(unittest.TestCase):
    def test_windows_lock_retries_without_overwriting_previous_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "metadata.json"
            target.write_text('{"version": 1}', encoding="utf-8")
            replace = os.replace
            attempts = []

            def locked_twice(source, destination):
                attempts.append(Path(source))
                if len(attempts) <= 2:
                    self.assertEqual(json.loads(target.read_text()), {"version": 1})
                    raise PermissionError("simulated Windows reader lock")
                return replace(source, destination)

            with patch("atomic_artifacts.os.replace", side_effect=locked_twice):
                write_json(target, {"version": 2}, attempts=3, delay=0)
            self.assertEqual(json.loads(target.read_text()), {"version": 2})
            self.assertEqual(len(attempts), 3)
            self.assertEqual(list(Path(temporary).iterdir()), [target])

    def test_permanent_lock_is_fatal_not_a_model_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "metadata.json"
            target.write_text('{"version": 1}', encoding="utf-8")
            with patch("atomic_artifacts.os.replace", side_effect=PermissionError("locked")) as replace:
                with self.assertRaises(ArtifactWriteError):
                    write_json(target, {"version": 2}, attempts=3, delay=0)
            self.assertEqual(replace.call_count, 3)
            self.assertEqual(json.loads(target.read_text()), {"version": 1})
            self.assertFalse(issubclass(ArtifactWriteError, Exception))
            self.assertEqual(list(Path(temporary).iterdir()), [target])
