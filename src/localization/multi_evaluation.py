"""Variant-agnostic N-by-M evaluation built on the existing dataset/localizer contracts."""
from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from .config import LocalizationConfig
from .dataset import SEMLocalizationDataset
from .external_ground_truth import GroundTruthValidationError, resolve_external_ground_truth
from .inference import localize as production_localize
from .identity import normalize_sample_id


def build_combinations(search_variants, reference_variants, mode="all"):
    if mode == "matched":
        references = {str(item["label"]).casefold(): item for item in reference_variants}
        pairs = [(item, references[str(item["label"]).casefold()]) for item in search_variants
                 if str(item["label"]).casefold() in references]
        skipped = [item["label"] for item in search_variants if str(item["label"]).casefold() not in references]
        return pairs, skipped
    return [(search, reference) for search in search_variants for reference in reference_variants], []


def metrics(rows):
    rows = [row for row in rows if row.get("error_px") is not None]
    if not rows:
        return {"Samples": 0}
    errors = np.asarray([float(row["error_px"]) for row in rows])
    confidence = [float(row["confidence"]) for row in rows if row.get("confidence") not in (None, "")]
    result = {"Samples": len(rows), **{f"Accuracy@{p}px": float(np.mean(errors <= p)) for p in (1, 2, 3, 4, 5, 10, 20, 50)},
              "Mean Error": float(errors.mean()), "Median Error": float(np.median(errors)),
              "P90 Error": float(np.percentile(errors, 90)), "P95 Error": float(np.percentile(errors, 95)),
              "Max Error": float(errors.max()), "False Localization Rate": float(np.mean(errors > 10)),
              "Mean Confidence": float(np.mean(confidence)) if confidence else None}
    return result


def _records(directory, allow_plain_images=False):
    root = Path(directory).expanduser()
    if (root / "annotations.json").is_file() or (root / "ground_truth.csv").is_file():
        records = SEMLocalizationDataset(root, LocalizationConfig()).records
    elif allow_plain_images and root.is_dir():
        images = sorted((path for path in root.iterdir()
                         if path.is_file() and path.suffix.casefold() in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}),
                        key=lambda path: path.name.casefold())
        if not images:
            raise ValueError(f"No supported image files in {root}")
        records = [{"sample_id": normalize_sample_id(path.name), "search": path.name, "reference": path.name,
                    "noise_mode": None, "process_type": "external"} for path in images]
    else:
        raise FileNotFoundError(f"Expected annotations.json or ground_truth.csv at dataset root: {root}")
    indexed = {}
    for record in records:
        sample_id = normalize_sample_id(record["sample_id"])
        if sample_id in indexed:
            raise ValueError(f"Duplicate normalized sample ID {sample_id} in {root}")
        indexed[sample_id] = record
    return root, indexed


def _csv_path(output_dir, search_label, reference_label):
    safe = lambda value: re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value)).strip("_") or "variant"
    return Path(output_dir) / "multi_dataset" / f"{safe(search_label)}__{safe(reference_label)}.csv"


def _read_rows(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _write_rows(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def evaluate_variant_pair(search_variant, reference_variant, output_dir, force=False,
                          localizer=production_localize, progress=None, external_ground_truth=None):
    """Evaluate exact shared IDs and write each completed combination immediately."""
    destination = _csv_path(output_dir, search_variant["label"], reference_variant["label"])
    complete_marker = destination.with_suffix(".complete")
    if destination.exists() and complete_marker.exists() and not force and external_ground_truth is None:
        rows = _read_rows(destination)
        return rows, {"cached": True, "missing_search": [], "missing_reference": [], "path": str(destination)}
    search_root, search_records = _records(search_variant["directory"], allow_plain_images=external_ground_truth is not None)
    reference_root, reference_records = _records(reference_variant["directory"], allow_plain_images=external_ground_truth is not None)
    matching = sorted(set(search_records) & set(reference_records), key=lambda value: (len(value), value))
    missing_search = sorted(set(reference_records) - set(search_records))
    missing_reference = sorted(set(search_records) - set(reference_records))
    rows = [] if force or external_ground_truth is not None or not destination.exists() else _read_rows(destination)
    completed_ids = {normalize_sample_id(row["sample_id"]) for row in rows}
    resolved_gt, used_gt = {}, set()
    if external_ground_truth is not None:
        for sample_id in matching:
            search_record, reference_record = search_records[sample_id], reference_records[sample_id]
            gt = resolve_external_ground_truth(external_ground_truth, sample_id,
                                               search_record["search"], reference_record["reference"])
            if gt is None:
                resolved_gt[sample_id] = None
                continue
            search = cv2.imread(str(search_root / search_record["search"]), cv2.IMREAD_GRAYSCALE)
            if search is None:
                raise ValueError(f"Ground truth validation failed: cannot read Search image {search_record['search']}.")
            height, width = search.shape[:2]
            if not (0 <= float(gt["center_x"]) < width):
                raise GroundTruthValidationError(
                    f"Ground truth validation failed: sample_id {sample_id} has center_x={gt['center_x']}, outside Search width {width}."
                )
            if not (0 <= float(gt["center_y"]) < height):
                raise GroundTruthValidationError(
                    f"Ground truth validation failed: sample_id {sample_id} has center_y={gt['center_y']}, outside Search height {height}."
                )
            resolved_gt[sample_id] = gt
            used_gt.add(id(gt))
    for index, sample_id in enumerate(matching, start=1):
        if sample_id in completed_ids:
            if progress:
                progress(index, len(matching), search_variant, reference_variant)
            continue
        search_record, reference_record = search_records[sample_id], reference_records[sample_id]
        search = cv2.imread(str(search_root / search_record["search"]), cv2.IMREAD_GRAYSCALE)
        reference = cv2.imread(str(reference_root / reference_record["reference"]), cv2.IMREAD_GRAYSCALE)
        if search is None or reference is None:
            continue
        prediction = localizer(search, reference)
        gt = (resolved_gt.get(sample_id) if external_ground_truth is not None
              else {"center_x": search_record["center_x"], "center_y": search_record["center_y"],
                    "noise_mode": search_record.get("noise_mode")})
        gx = float(gt["center_x"]) if gt is not None else None
        gy = float(gt["center_y"]) if gt is not None else None
        error = (float(np.hypot(float(prediction["center_x"]) - gx, float(prediction["center_y"]) - gy))
                 if gt is not None else None)
        rows.append({"search_variant": search_variant["label"], "reference_variant": reference_variant["label"],
                     "sample_id": sample_id, "search_path": search_record["search"], "reference_path": reference_record["reference"],
                     "gt_x": gx, "gt_y": gy, "pred_x": prediction["center_x"], "pred_y": prediction["center_y"],
                     "error_px": error, "confidence": prediction.get("confidence"),
                     **{f"success_at_{p}px": (error <= p if error is not None else None) for p in range(1, 6)},
                     "search_metadata_noise": (gt or {}).get("noise_mode") or search_record.get("noise_mode"),
                     "reference_metadata_noise": reference_record.get("noise_mode"),
                     "result": "Missing Ground Truth" if error is None else "Scored"})
        _write_rows(destination, rows)
        if progress:
            progress(index, len(matching), search_variant, reference_variant)
    complete_marker.parent.mkdir(parents=True, exist_ok=True)
    complete_marker.write_text("complete\n", encoding="utf-8")
    return rows, {"cached": False, "missing_search": missing_search, "missing_reference": missing_reference,
                  "matched_gt": sum(row.get("error_px") is not None for row in rows),
                  "missing_gt": sum(row.get("error_px") is None for row in rows),
                  "unused_gt": len(external_ground_truth or []) - len(used_gt), "path": str(destination)}


def write_experiment(output_dir, search_variants, reference_variants, mode, summaries):
    root = Path(output_dir) / "multi_dataset"; root.mkdir(parents=True, exist_ok=True)
    config = {"search_variants": search_variants, "reference_variants": reference_variants, "comparison_mode": mode,
              "timestamp": datetime.now().isoformat(), "combinations": len(summaries)}
    (root / "experiment_config.json").write_text(__import__("json").dumps(config, indent=2), encoding="utf-8")
    _write_rows(root / "summary.csv", summaries)
