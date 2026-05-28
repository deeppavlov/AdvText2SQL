"""Pytest configuration: make `src/` importable and register markers."""
from __future__ import annotations

import sys
from pathlib import Path


# Add `src/` to sys.path so tests can import `adv_text2sql` without install
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers", "integration: integration tests requiring DB access via SSH tunnel"
    )
