"""Shared implementation of the challenge's centre tie rule."""
from __future__ import annotations

import numpy as np


def resolve_equivalent_peak(peaks, search_center, tolerance, template_half_size=50.):
    """Choose the search-centre-nearest peak only among equivalent best peaks.

    ``peaks`` must be sorted highest-score first and use template top-left
    coordinates.  This utility intentionally has no ground-truth dependency.
    """
    if not peaks:
        raise ValueError('Cannot resolve an empty peak list.')
    best_score = float(peaks[0][2])
    equivalent = [
        {'x': float(x + template_half_size), 'y': float(y + template_half_size), 'score': float(score)}
        for x, y, score in peaks
        if best_score - float(score) <= tolerance
    ]
    chosen = min(equivalent, key=lambda item: np.hypot(item['x'] - search_center[0], item['y'] - search_center[1]))
    return equivalent, chosen
