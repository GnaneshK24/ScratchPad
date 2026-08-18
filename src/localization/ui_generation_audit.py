"""End-to-end audit of Streamlit's FinFET Generate Pairs workflow.

This intentionally imports the functions that the Streamlit controls call:
``app.generate_one``, ``app.save_pair``, ``app.localize``, and
``app.noise_comparison_rows``.  It is therefore a repeatable, non-interactive
audit of the UI path rather than a second generator or matcher implementation.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app  # noqa: E402  - import after the repository root is available


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def _require_new_empty_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise SystemExit(f"Refusing to reuse a non-empty audit directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _run_center_rule_audit(dataset_dir: Path, output_root: Path) -> dict:
    command = [
        sys.executable,
        "-m",
        "src.localization.center_rule_diagnose",
        "--dataset-dir",
        str(dataset_dir),
        "--output-dir",
        str(output_root),
    ]
    completed = subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True)
    (output_root / "center_rule_audit_console.txt").write_text(
        completed.stdout + completed.stderr, encoding="utf-8"
    )
    return json.loads((output_root / "diagnostics" / "center_rule_consistency_summary.json").read_text(encoding="utf-8"))


def _cli_prediction(search_path: Path, reference_path: Path) -> dict:
    completed = subprocess.run(
        [sys.executable, "src/inference.py", "--search_path", str(search_path), "--reference_path", str(reference_path)],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return json.loads(completed.stdout)


def _ui_vs_cli(dataset_dir: Path, pairs: list[dict], count: int) -> list[dict]:
    comparisons = []
    for pair in pairs[:count]:
        sample_id = int(pair["sample_id"])
        search_path = dataset_dir / f"search_{sample_id:04d}.png"
        reference_path = dataset_dir / f"reference_{sample_id:04d}.png"
        ui = app.localize(pair["search"], pair["reference"])
        cli = _cli_prediction(search_path, reference_path)
        equal = all(math.isclose(float(ui[key]), float(cli[key]), abs_tol=1e-9, rel_tol=0.0)
                    for key in ("center_x", "center_y", "score", "confidence"))
        comparisons.append({
            "sample_id": sample_id,
            "ui_center_x": float(ui["center_x"]),
            "ui_center_y": float(ui["center_y"]),
            "ui_score": float(ui["score"]),
            "ui_confidence": float(ui["confidence"]),
            "cli_center_x": float(cli["center_x"]),
            "cli_center_y": float(cli["center_y"]),
            "cli_score": float(cli["score"]),
            "cli_confidence": float(cli["confidence"]),
            "exact_numeric_match": equal,
        })
    return comparisons


def _validation_report(rows: list[dict], skipped: list[str]) -> dict:
    values = np.asarray([float(row["error_px"]) for row in rows], dtype=float)
    return {
        "matching_sample_ids": len(rows),
        "skipped": len(skipped),
        "accuracy_at_1px": float(np.mean(values <= 1)),
        "accuracy_at_2px": float(np.mean(values <= 2)),
        "accuracy_at_5px": float(np.mean(values <= 5)),
        "accuracy_at_10px": float(np.mean(values <= 10)),
        "mean_error_px": float(values.mean()),
        "median_error_px": float(np.median(values)),
        "p95_error_px": float(np.percentile(values, 95)),
        "max_error_px": float(values.max()),
        "error_over_20px": float(np.mean(values > 20)),
        "error_over_50px": float(np.mean(values > 50)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the Streamlit FinFET Generate Pairs path.")
    parser.add_argument("--output-dir", required=True, help="New, empty parent directory for the audit batch.")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=9200)
    parser.add_argument("--sample-id", type=int, default=0)
    parser.add_argument("--cli-check-count", type=int, default=3)
    args = parser.parse_args()
    if args.count < 1 or args.cli_check_count < 1:
        raise SystemExit("--count and --cli-check-count must be positive")

    output_root = Path(args.output_dir).expanduser().resolve()
    _require_new_empty_directory(output_root)
    clean_dir = output_root / "clean"

    # These two calls are the exact methods reached by Generate Pair.  The
    # bare Streamlit session-state assignment supplies only the UI-selected
    # output directory; no generation, GT, or saving logic is duplicated.
    app.st.session_state.output_directory = str(output_root)
    pairs = []
    started = time.perf_counter()
    for offset in range(args.count):
        pair = app.generate_one("FinFET", args.seed + offset, "clean", args.sample_id + offset)
        app.save_pair(pair)
        pairs.append(pair)
    generation_seconds = time.perf_counter() - started

    manifest = json.loads((clean_dir / "annotations.json").read_text(encoding="utf-8"))
    if len(manifest.get("pairs", [])) != args.count:
        raise RuntimeError(f"Expected {args.count} saved manifest pairs, found {len(manifest.get('pairs', []))}")
    required_files = [
        clean_dir / f"{kind}_{int(pair['sample_id']):04d}.png"
        for pair in pairs
        for kind in ("search", "reference")
    ]
    missing_files = [str(path) for path in required_files if not path.is_file()]
    if missing_files:
        raise RuntimeError(f"Saved UI batch has missing images: {missing_files}")

    contract = _run_center_rule_audit(clean_dir, output_root)
    if contract["center_rule_consistent_samples"] != args.count:
        raise RuntimeError(f"Center-rule audit failed: {contract}")

    # This invokes the exact Dataset Validation worker used by the Streamlit UI.
    rows, skipped = app.noise_comparison_rows(str(clean_dir), str(clean_dir))
    validation = _validation_report(rows, skipped)
    if validation["matching_sample_ids"] != args.count or validation["skipped"] != 0:
        raise RuntimeError(f"UI Dataset Validation did not find exactly the generated pairs: {validation}")
    display_rows = [{key: value for key, value in row.items() if key not in {"pair", "result"}} for row in rows]
    with (output_root / "ui_dataset_validation_rows.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(display_rows[0]))
        writer.writeheader()
        writer.writerows(display_rows)

    comparisons = _ui_vs_cli(clean_dir, pairs, min(args.cli_check_count, len(pairs)))
    if not all(row["exact_numeric_match"] for row in comparisons):
        raise RuntimeError("CLI and UI predictions diverged on an audited generated pair")
    with (output_root / "ui_vs_cli_predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(comparisons)

    report = {
        "generator_call": "app.generate_one('FinFET', seed, 'clean', sample_id)",
        "persistence_call": "app.save_pair(pair)",
        "dataset_validation_call": "app.noise_comparison_rows(clean_dir, clean_dir)",
        "samples_requested": args.count,
        "samples_saved": len(manifest["pairs"]),
        "generation_seconds": generation_seconds,
        "center_rule_contract": contract,
        "dataset_validation": validation,
        "ui_cli_comparisons": comparisons,
        "verdict": "PASS",
    }
    (output_root / "ui_generation_audit.json").write_text(
        json.dumps(report, indent=2, default=_json_default), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
