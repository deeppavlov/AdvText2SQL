"""Bounded atomic JSON writes, including transient Windows reader locks."""

import json
import os
from pathlib import Path
import tempfile
import time


class ArtifactWriteError(BaseException):
    """Stop the benchmark instead of turning an artifact failure into a model label."""


def write_json(path, data, *, attempts=50, delay=0.1):
    path = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                         prefix=f".{path.name}.", suffix=".tmp",
                                         dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(data, stream, indent=2, ensure_ascii=False, default=str)
            stream.write("\n")
        for attempt in range(attempts):
            try:
                os.replace(temporary, path)
                return
            except PermissionError:
                if attempt + 1 == attempts:
                    raise
                time.sleep(delay)
    except OSError as error:
        raise ArtifactWriteError(f"Cannot save {path.name} after bounded IO retries ({type(error).__name__})") from error
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
