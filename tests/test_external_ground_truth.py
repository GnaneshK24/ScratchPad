"""Focused coverage for evaluator-provided Dataset Validation ground truth."""
from __future__ import annotations

import csv
import json

import cv2
import numpy as np
import pytest

from app import validation_pair_records
from src.localization.external_ground_truth import (GroundTruthValidationError,
                                                    parse_external_ground_truth,
                                                    resolve_external_ground_truth)


def _plain_pair(tmp_path, identifiers=("000001",), shape=(24, 32)):
    search_dir, reference_dir = tmp_path / "search", tmp_path / "reference"
    search_dir.mkdir(); reference_dir.mkdir()
    for identifier in identifiers:
        image = np.zeros(shape, dtype=np.uint8)
        assert cv2.imwrite(str(search_dir / f"search_{identifier}.png"), image)
        assert cv2.imwrite(str(reference_dir / f"reference_{identifier}.png"), image)
    return search_dir, reference_dir


def test_valid_csv_is_parsed_with_required_fields():
    records = parse_external_ground_truth(
        "ground_truth.csv", b"sample_id,center_x,center_y\n000001,12.4,18.2\n"
    )
    assert records[0]["sample_id"] == "000001"
    assert records[0]["center_x"] == 12.4


def test_valid_json_preserves_optional_metadata():
    records = parse_external_ground_truth(
        "meta.json", json.dumps({"samples": [{"sample_id": "000001", "center_x": 12, "center_y": 18,
                                                   "noise_mode": "medium", "rotation_deg": 2.0}]}).encode()
    )
    assert records[0]["noise_mode"] == "medium"
    assert records[0]["rotation_deg"] == 2.0


def test_csv_with_filenames_prefers_exact_filename_match():
    records = parse_external_ground_truth(
        "ground_truth.csv",
        b"sample_id,reference_filename,search_filename,center_x,center_y\nwrong,reference_000001.png,search_000001.png,12,18\n",
    )
    resolved = resolve_external_ground_truth(records, "1", "search_000001.png", "reference_000001.png")
    assert resolved is records[0]


def test_missing_gt_record_is_reported_without_guessing(tmp_path):
    search_dir, reference_dir = _plain_pair(tmp_path, ("000001", "000002"))
    records = parse_external_ground_truth(
        "ground_truth.csv", b"sample_id,center_x,center_y\n000001,12,18\n"
    )
    pairs, _, summary = validation_pair_records(str(search_dir), str(reference_dir), records)
    assert [pair["gt_record"] is not None for pair in pairs] == [True, False]
    assert summary["matched_gt"] == 1
    assert summary["missing_gt"] == 1
    assert summary["unused_gt"] == 0


def test_extra_gt_record_is_reported_without_failing_pairs(tmp_path):
    search_dir, reference_dir = _plain_pair(tmp_path)
    records = parse_external_ground_truth(
        "ground_truth.csv", b"sample_id,center_x,center_y\n000001,12,18\n999999,12,18\n"
    )
    _, _, summary = validation_pair_records(str(search_dir), str(reference_dir), records)
    assert summary["matched_gt"] == 1
    assert summary["unused_gt"] == 1


def test_duplicate_sample_id_is_rejected():
    with pytest.raises(GroundTruthValidationError, match="duplicate sample_id"):
        parse_external_ground_truth(
            "ground_truth.csv", b"sample_id,center_x,center_y\n000001,12,18\n1,13,19\n"
        )


def test_invalid_coordinate_is_rejected():
    with pytest.raises(GroundTruthValidationError, match="non-numeric"):
        parse_external_ground_truth("ground_truth.csv", b"sample_id,center_x,center_y\n000001,nope,18\n")


def test_coordinate_outside_search_bounds_is_rejected(tmp_path):
    search_dir, reference_dir = _plain_pair(tmp_path)
    records = parse_external_ground_truth("ground_truth.csv", b"sample_id,center_x,center_y\n000001,32,18\n")
    with pytest.raises(GroundTruthValidationError, match="outside Search width 32"):
        validation_pair_records(str(search_dir), str(reference_dir), records)


def test_normalized_id_matches_search_and_reference_filenames(tmp_path):
    search_dir, reference_dir = _plain_pair(tmp_path)
    records = parse_external_ground_truth("ground_truth.csv", b"sample_id,center_x,center_y\n000001,12,18\n")
    pairs, _, _ = validation_pair_records(str(search_dir), str(reference_dir), records)
    assert pairs[0]["sample_id"] == "1"
    assert pairs[0]["gt_record"]["center_y"] == 18.0


def test_existing_generated_dataset_metadata_still_resolves_without_upload(tmp_path):
    search_dir, reference_dir = tmp_path / "search", tmp_path / "reference"
    search_dir.mkdir(); reference_dir.mkdir()
    image = np.zeros((1000, 1000), dtype=np.uint8)
    assert cv2.imwrite(str(search_dir / "search_000.png"), image)
    assert cv2.imwrite(str(reference_dir / "ref_000.png"), image)
    with (tmp_path / "ground_truth.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("pair_id", "reference_file", "search_file", "center_x", "center_y", "noise_mode"))
        writer.writeheader()
        writer.writerow({"pair_id": "000", "reference_file": "ref_000.png", "search_file": "search_000.png",
                         "center_x": 512, "center_y": 488, "noise_mode": "clean"})
    pairs, _, summary = validation_pair_records(str(tmp_path), str(tmp_path))
    assert pairs[0]["gt_record"]["center_x"] == 512.0
    assert summary["external_gt"] is False
    assert summary["has_noise_metadata"] is True
