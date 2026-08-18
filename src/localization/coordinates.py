"""Single source of truth for the 1000px-search / 100px-reference contract."""
from __future__ import annotations

from typing import Sequence


def half_reference(reference_size: int = 100) -> float:
    return reference_size / 2.0


def search_center_to_top_left(x: float, y: float, reference_size: int = 100) -> tuple[float, float]:
    """Convert a search-image centre to the template's top-left pixel."""
    half = half_reference(reference_size)
    return float(x) - half, float(y) - half


def top_left_to_center(x: float, y: float, reference_size: int = 100) -> tuple[float, float]:
    """Convert a template top-left pixel to a search-image centre."""
    half = half_reference(reference_size)
    return float(x) + half, float(y) + half


def center_to_bbox(x: float, y: float, reference_size: int = 100) -> list[float]:
    left, top = search_center_to_top_left(x, y, reference_size)
    return [left, top, left + reference_size, top + reference_size]


def validate_center(x: float, y: float, search_size: int = 1000, reference_size: int = 100) -> None:
    half = half_reference(reference_size)
    if not (half <= float(x) <= search_size - half and half <= float(y) <= search_size - half):
        raise ValueError(f'center {(x, y)} must be within [{half}, {search_size-half}] for a {reference_size}px reference')


def normalize_coordinate(x: float, search_size: int = 1000) -> float:
    return float(x) / float(search_size)


def denormalize_coordinate(x: float, search_size: int = 1000) -> float:
    return float(x) * float(search_size)


def reference_to_search_coordinates(point: Sequence[float], top_left: Sequence[float]) -> tuple[float, float]:
    """Map a reference-local point to the search image; no resizing is implied."""
    return float(point[0]) + float(top_left[0]), float(point[1]) + float(top_left[1])
