"""Durable IO overlay: sync temporary file contents before atomic replacement."""

import json
import os
from pathlib import Path
import tempfile
import time

from atomic_artifacts import ArtifactWriteError


def write_bytes(path, data, *, attempts=50, delay=0.1):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix="." + path.name + ".",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        for attempt in range(attempts):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt + 1 == attempts:
                    raise
                time.sleep(delay)
    except OSError as error:
        raise ArtifactWriteError(
            f"Durable write failed for {path.name} ({type(error).__name__})"
        ) from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_json(path, data, **kwargs):
    write_bytes(
        path,
        (json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n").encode(
            "utf8"
        ),
        **kwargs,
    )


def append(path, value):
    try:
        with Path(path).open("ab") as stream:
            stream.write((json.dumps(value, ensure_ascii=False) + "\n").encode("utf8"))
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as error:
        raise ArtifactWriteError(
            f"Durable journal failed ({type(error).__name__})"
        ) from error
