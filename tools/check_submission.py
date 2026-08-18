"""Non-destructive public-submission smoke check.

This utility is outside the runtime API. By default it creates a temporary
five-pair FinFET dataset, validates the public command-line contract, and
removes that temporary directory when complete.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
COORDINATE_LINE = re.compile(r"^[+-]?\d+(?:\.\d+)?,[+-]?\d+(?:\.\d+)?\r?\n?$")
INFERENCE_SOURCES = (
    ROOT / "localize.py",
    ROOT / "src" / "localization" / "inference.py",
    ROOT / "src" / "localization" / "classical_matcher.py",
    ROOT / "src" / "localization" / "center_rule.py",
)
FORBIDDEN_INFERENCE_TERMS = ("annotations.json", "ground_truth.csv", "metadata_", "benchmark_data")


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)


def _read_rows(dataset_dir: Path, expected_count: int) -> list[dict[str, str]]:
    ground_truth = dataset_dir / "ground_truth.csv"
    if not ground_truth.is_file():
        raise AssertionError("Generator did not write ground_truth.csv.")
    with ground_truth.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != expected_count:
        raise AssertionError(f"Expected {expected_count} ground-truth rows, found {len(rows)}.")
    for row in rows:
        pair_id = int(row["pair_id"])
        reference = dataset_dir / "reference" / row["reference_file"]
        search = dataset_dir / "search" / row["search_file"]
        if not reference.is_file() or not search.is_file():
            raise AssertionError(f"Missing pair files for pair_id={pair_id}.")
        reference_image = cv2.imread(str(reference), cv2.IMREAD_GRAYSCALE)
        search_image = cv2.imread(str(search), cv2.IMREAD_GRAYSCALE)
        if reference_image is None or search_image is None:
            raise AssertionError(f"Undecodable pair files for pair_id={pair_id}.")
        if reference_image.shape != (1000, 1000) or search_image.shape != (1000, 1000):
            raise AssertionError(f"Unexpected image shape for pair_id={pair_id}.")
        x, y = float(row["center_x"]), float(row["center_y"])
        if not (0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0):
            raise AssertionError(f"Out-of-range GT center for pair_id={pair_id}: {(x, y)}")
    return rows


def _cli_prediction(reference: Path, search: Path) -> tuple[float, float]:
    completed = _run([sys.executable, "localize.py", "--reference", str(reference), "--search", str(search)])
    if not COORDINATE_LINE.fullmatch(completed.stdout):
        raise AssertionError(f"localize.py stdout was not exactly x,y: {completed.stdout!r}")
    x_text, y_text = completed.stdout.strip().split(",")
    x, y = float(x_text), float(y_text)
    if not (math.isfinite(x) and math.isfinite(y)):
        raise AssertionError("localize.py returned non-finite coordinates.")
    return x, y


def _assert_no_gt_dependency() -> None:
    for source in INFERENCE_SOURCES:
        text = source.read_text(encoding="utf-8").lower()
        leaked = [term for term in FORBIDDEN_INFERENCE_TERMS if term in text]
        if leaked:
            raise AssertionError(f"Inference source {source.relative_to(ROOT)} mentions forbidden inputs: {leaked}")


def _assert_ui_parity(dataset_dir: Path, rows: list[dict[str, str]], predictions: dict[str, tuple[float, float]]) -> None:
    # Import only for a verification call; this does not start Streamlit.
    import app

    first_three = [str(row["pair_id"]) for row in rows[:3]]
    for sample_id in first_three:
        ui_rows, skipped = app.noise_comparison_rows(str(dataset_dir), str(dataset_dir), sample_id=sample_id)
        if skipped or len(ui_rows) != 1:
            raise AssertionError(f"UI validation could not process pair_id={sample_id}: skipped={skipped}")
        ui = ui_rows[0]
        cli_x, cli_y = predictions[sample_id]
        if abs(float(ui["Pred X"]) - cli_x) > 1e-6 or abs(float(ui["Pred Y"]) - cli_y) > 1e-6:
            raise AssertionError(f"UI/CLI mismatch for pair_id={sample_id}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run non-destructive public submission checks.")
    parser.add_argument("--num-pairs", type=int, default=5, help="Number of fresh FinFET pairs to check (default: 5).")
    parser.add_argument("--seed", type=int, default=42, help="Generation seed (default: 42).")
    parser.add_argument("--dataset-dir", type=Path, help="Validate an existing generated dataset instead of making a temporary one.")
    args = parser.parse_args()
    if args.num_pairs < 1:
        parser.error("--num-pairs must be at least 1.")

    _assert_no_gt_dependency()
    temporary_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.dataset_dir:
        dataset_dir = args.dataset_dir.resolve()
    else:
        temporary_dir = tempfile.TemporaryDirectory(prefix="sem_submission_check_")
        dataset_dir = Path(temporary_dir.name) / "dataset"
        _run([
            sys.executable, "generate_dataset.py", "--architecture", "finfet",
            "--num-pairs", str(args.num_pairs), "--output-dir", str(dataset_dir), "--seed", str(args.seed),
        ])

    try:
        rows = _read_rows(dataset_dir, args.num_pairs)
        predictions = {}
        errors = []
        for row in rows:
            pair_id = str(row["pair_id"])
            predicted_x, predicted_y = _cli_prediction(
                dataset_dir / "reference" / row["reference_file"],
                dataset_dir / "search" / row["search_file"],
            )
            predictions[pair_id] = (predicted_x, predicted_y)
            errors.append(math.hypot(predicted_x - float(row["center_x"]), predicted_y - float(row["center_y"])))

        # Compare one CLI prediction directly with the exact same production API.
        from src.localization.inference import localize as production_localize

        first = rows[0]
        direct = production_localize(
            cv2.imread(str(dataset_dir / "search" / first["search_file"]), cv2.IMREAD_GRAYSCALE),
            cv2.imread(str(dataset_dir / "reference" / first["reference_file"]), cv2.IMREAD_GRAYSCALE),
        )
        cli_x, cli_y = predictions[str(first["pair_id"])]
        if abs(float(direct["center_x"]) - cli_x) > 1e-6 or abs(float(direct["center_y"]) - cli_y) > 1e-6:
            raise AssertionError("Public CLI and direct production matcher disagree.")

        _assert_ui_parity(dataset_dir, rows, predictions)
        print(json.dumps({
            "status": "passed",
            "dataset": str(dataset_dir),
            "pairs": len(rows),
            "cli_direct_parity": "passed",
            "ui_cli_parity_pairs": min(3, len(rows)),
            "median_error_px": sorted(errors)[len(errors) // 2],
            "max_error_px": max(errors),
        }, indent=2))
    finally:
        if temporary_dir is not None:
            temporary_dir.cleanup()


if __name__ == "__main__":
    main()
