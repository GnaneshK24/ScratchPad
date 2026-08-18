"""Shared deterministic identifiers for dataset-validation records."""
from __future__ import annotations

import re
from pathlib import Path


def normalize_sample_id(value: str) -> str:
    """Match common ``search_0001``/``reference_0001`` names by their final ID."""
    digits = re.findall(r"\d+", str(value))
    return str(int(digits[-1])) if digits else str(value).strip().casefold()


def normalize_filename(value: str) -> str:
    """Normalize a supplied filename without weakening it to an ID match."""
    return Path(str(value).replace("\\", "/")).name.casefold()
