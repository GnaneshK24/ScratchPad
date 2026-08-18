"""Regression coverage for the public localize.py argument contract."""
from __future__ import annotations

import cv2
import numpy as np

import localize


def test_predict_passes_search_then_reference_to_production_matcher(tmp_path, monkeypatch):
    """The public flags must not invert the production API's image order."""
    reference = np.full((100, 100), 17, dtype=np.uint8)
    search = np.full((1000, 1000), 93, dtype=np.uint8)
    reference_path = tmp_path / "reference.png"
    search_path = tmp_path / "search.png"
    assert cv2.imwrite(str(reference_path), reference)
    assert cv2.imwrite(str(search_path), search)

    received = {}

    def fake_production_localize(search_image, reference_image):
        received["search"] = search_image
        received["reference"] = reference_image
        return {"center_x": 123.25, "center_y": 456.75, "confidence": 0.9}

    monkeypatch.setattr(localize, "production_localize", fake_production_localize)

    result = localize.predict(str(reference_path), str(search_path))

    assert np.array_equal(received["search"], search)
    assert np.array_equal(received["reference"], reference)
    assert result["center_x"] == 123.25
    assert result["center_y"] == 456.75
