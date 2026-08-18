import numpy as np

from src.localization.diagnose import _dominant_period


def test_dominant_period_detects_repeating_response_profile():
    x = np.arange(160, dtype=np.float64)
    response_map = np.tile(np.cos(2 * np.pi * x / 16), (120, 1)).astype(np.float32)

    assert _dominant_period(response_map, axis=0) == 16

