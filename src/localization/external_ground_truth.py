"""Validation-only parsing and deterministic resolution of evaluator GT files."""
from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path

from .identity import normalize_filename, normalize_sample_id


class GroundTruthValidationError(ValueError):
    """Raised with a user-actionable message when an evaluator GT file is invalid."""


def _clean_record(record: dict, index: int) -> dict:
    values = {str(key).strip().casefold(): value for key, value in record.items() if key is not None}
    label = values.get("sample_id") or values.get("search_filename") or values.get("reference_filename") or f"row {index}"
    if "center_x" not in values or "center_y" not in values:
        raise GroundTruthValidationError(f"Ground truth validation failed: {label} is missing center_x or center_y.")
    try:
        center_x, center_y = float(values["center_x"]), float(values["center_y"])
    except (TypeError, ValueError) as exc:
        raise GroundTruthValidationError(
            f"Ground truth validation failed: {label} has non-numeric center_x or center_y."
        ) from exc
    if not (math.isfinite(center_x) and math.isfinite(center_y)):
        raise GroundTruthValidationError(
            f"Ground truth validation failed: {label} has non-finite center_x or center_y."
        )
    sample_id = str(values.get("sample_id") or "").strip()
    search_filename = str(values.get("search_filename") or "").strip()
    reference_filename = str(values.get("reference_filename") or "").strip()
    if not sample_id and not search_filename and not reference_filename:
        raise GroundTruthValidationError(
            f"Ground truth validation failed: row {index} needs sample_id, search_filename, or reference_filename."
        )
    cleaned = dict(record)
    cleaned.update({
        "sample_id": sample_id or None,
        "search_filename": search_filename or None,
        "reference_filename": reference_filename or None,
        "center_x": center_x,
        "center_y": center_y,
        "_normalized_sample_id": normalize_sample_id(sample_id) if sample_id else None,
        "_normalized_search_filename": normalize_filename(search_filename) if search_filename else None,
        "_normalized_reference_filename": normalize_filename(reference_filename) if reference_filename else None,
    })
    return cleaned


def parse_external_ground_truth(filename: str, content: bytes) -> list[dict]:
    """Parse the documented CSV or JSON GT formats without saving uploads to disk."""
    suffix = Path(filename).suffix.casefold()
    if suffix not in {".csv", ".json"}:
        raise GroundTruthValidationError("Ground truth validation failed: upload a .csv or .json file.")
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise GroundTruthValidationError("Ground truth validation failed: file must be UTF-8 text.") from exc
    try:
        if suffix == ".csv":
            source_records = list(csv.DictReader(io.StringIO(text)))
        else:
            payload = json.loads(text)
            source_records = payload.get("samples") if isinstance(payload, dict) else None
            if not isinstance(source_records, list):
                raise GroundTruthValidationError("Ground truth validation failed: JSON must contain a samples array.")
    except csv.Error as exc:
        raise GroundTruthValidationError(f"Ground truth validation failed: invalid CSV ({exc}).") from exc
    except json.JSONDecodeError as exc:
        raise GroundTruthValidationError(f"Ground truth validation failed: invalid JSON ({exc.msg}).") from exc
    if not source_records:
        raise GroundTruthValidationError("Ground truth validation failed: no records were found.")
    if not all(isinstance(item, dict) for item in source_records):
        raise GroundTruthValidationError("Ground truth validation failed: every record must be an object/CSV row.")

    records = [_clean_record(item, index + 1) for index, item in enumerate(source_records)]
    seen_ids: set[str] = set()
    seen_filename_keys: set[tuple[str | None, str | None]] = set()
    for record in records:
        sample_id = record["_normalized_sample_id"]
        if sample_id:
            if sample_id in seen_ids:
                raise GroundTruthValidationError(
                    f"Ground truth validation failed: duplicate sample_id {record['sample_id']}."
                )
            seen_ids.add(sample_id)
        filename_key = (record["_normalized_search_filename"], record["_normalized_reference_filename"])
        if any(filename_key):
            if filename_key in seen_filename_keys:
                raise GroundTruthValidationError(
                    "Ground truth validation failed: duplicate search/reference filename record."
                )
            seen_filename_keys.add(filename_key)
    return records


def resolve_external_ground_truth(records: list[dict], sample_id: str,
                                  search_filename: str, reference_filename: str) -> dict | None:
    """Resolve one GT record by exact filenames, then by an ID-only record.

    A record that explicitly carries a filename never falls back to an ID for a
    different image; this prevents silently reusing GT across changed geometry.
    """
    search_key, reference_key = normalize_filename(search_filename), normalize_filename(reference_filename)
    exact = []
    for record in records:
        supplied_search = record.get("_normalized_search_filename")
        supplied_reference = record.get("_normalized_reference_filename")
        if supplied_search or supplied_reference:
            if ((not supplied_search or supplied_search == search_key) and
                    (not supplied_reference or supplied_reference == reference_key)):
                exact.append(record)
    if len(exact) > 1:
        raise GroundTruthValidationError(
            f"Ground truth validation failed: multiple filename records match {search_filename}."
        )
    if exact:
        return exact[0]
    normalized_id = normalize_sample_id(sample_id)
    by_id = [record for record in records
             if not record.get("_normalized_search_filename") and not record.get("_normalized_reference_filename")
             and record.get("_normalized_sample_id") == normalized_id]
    if len(by_id) > 1:
        raise GroundTruthValidationError(
            f"Ground truth validation failed: multiple ID-only records match sample_id {sample_id}."
        )
    return by_id[0] if by_id else None
