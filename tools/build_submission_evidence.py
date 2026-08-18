"""Build a reproducible, non-training evaluation evidence package.

This tool intentionally uses the existing FinFET generator and the same
``predict`` function used by ``localize.py``.  It never changes matcher
configuration, labels, candidate ranking, or centre-rule behaviour.
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dataset_generator import FinFETSEMDatasetGenerator  # noqa: E402
from localize import predict  # noqa: E402
from src.localization.visualize import save_localization_result  # noqa: E402

NOISE_MODES = ("clean", "low", "medium", "high")
TOLERANCES = (0.25, 0.5, 0.75, 1, 2, 3, 4, 5, 10, 20, 50)


def configure_console_encoding() -> None:
    """Allow the established generator's UTF-8 progress messages on Windows."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, OSError):
            pass


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_ground_truth(dataset_dir: Path) -> list[dict]:
    with (dataset_dir / "ground_truth.csv").open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def integrity_gate(dataset_dir: Path, expected: int) -> list[dict]:
    rows = read_ground_truth(dataset_dir)
    if len(rows) != expected:
        raise AssertionError(f"{dataset_dir}: expected {expected} rows, found {len(rows)}")
    ids = set()
    for row in rows:
        sample_id = str(row["pair_id"])
        if sample_id in ids:
            raise AssertionError(f"{dataset_dir}: duplicate sample ID {sample_id}")
        ids.add(sample_id)
        ref = dataset_dir / "reference" / row["reference_file"]
        search = dataset_dir / "search" / row["search_file"]
        if not ref.is_file() or not search.is_file():
            raise AssertionError(f"{dataset_dir}: missing reference/search for {sample_id}")
        ref_image = cv2.imread(str(ref), cv2.IMREAD_GRAYSCALE)
        search_image = cv2.imread(str(search), cv2.IMREAD_GRAYSCALE)
        if ref_image is None or search_image is None:
            raise AssertionError(f"{dataset_dir}: unreadable image for {sample_id}")
        if ref_image.shape != (1000, 1000) or search_image.shape != (1000, 1000):
            raise AssertionError(f"{dataset_dir}: unexpected image shape for {sample_id}")
        x, y = float(row["center_x"]), float(row["center_y"])
        if not (0 <= x <= search_image.shape[1] and 0 <= y <= search_image.shape[0]):
            raise AssertionError(f"{dataset_dir}: GT outside search bounds for {sample_id}")
    return rows


def generate_pool(pool: Path, pairs_per_noise: int, seed: int) -> list[tuple[str, Path, list[dict]]]:
    records = []
    for offset, mode in enumerate(NOISE_MODES):
        destination = pool / mode
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite existing evaluation pool: {destination}")
        generator = FinFETSEMDatasetGenerator(
            input_dir=str(ROOT / "finfet_base_images"),
            seed=seed + offset,
            noise_mode=mode,
        )
        generator.generate_dataset(output_count=pairs_per_noise, output_dir=str(destination))
        records.append((mode, destination, integrity_gate(destination, pairs_per_noise)))
    return records


def evaluate_pool(groups: list[tuple[str, Path, list[dict]]]) -> list[dict]:
    evaluated = []
    for mode, root, ground_truth in groups:
        for gt in ground_truth:
            ref = root / "reference" / gt["reference_file"]
            search = root / "search" / gt["search_file"]
            result = predict(str(ref), str(search))
            gx, gy = float(gt["center_x"]), float(gt["center_y"])
            px, py = float(result["center_x"]), float(result["center_y"])
            error = float(np.hypot(px - gx, py - gy))
            evaluated.append({
                "sample_id": f"{mode}_{int(gt['pair_id']):03d}",
                "search_image_path": relative(search),
                "GTX": gx, "GTY": gy, "Output_X": px, "Output_Y": py,
                "reference_image_path": relative(ref), "error_px": error,
                "confidence": float(result["confidence"]), "noise_mode": mode,
                "rotation_deg": float(gt["rotation_angle"]),
            })
    return evaluated


def metric_summary(rows: list[dict]) -> dict:
    errors = np.asarray([float(row["error_px"]) for row in rows], dtype=float)
    confidences = np.asarray([float(row["confidence"]) for row in rows], dtype=float)
    values = {
        "samples": int(len(rows)),
        **{f"accuracy_at_{t:g}px": float(np.mean(errors <= t)) for t in TOLERANCES},
        "mean_error_px": float(np.mean(errors)),
        "median_error_px": float(np.median(errors)),
        "p90_error_px": float(np.percentile(errors, 90)),
        "p95_error_px": float(np.percentile(errors, 95)),
        "max_error_px": float(np.max(errors)),
        "mean_confidence": float(np.mean(confidences)),
        "catastrophic_error_rate_gt_50px": float(np.mean(errors > 50)),
    }
    return values


def choose_representative(rows: list[dict], count: int = 30) -> list[dict]:
    """Stratify by actual noise profile and evenly cover each error range."""
    quotas = {"clean": 8, "low": 8, "medium": 7, "high": 7}
    if sum(quotas.values()) != count:
        raise AssertionError("Selection quotas must equal requested count")
    selected = []
    for mode in NOISE_MODES:
        candidates = sorted((row for row in rows if row["noise_mode"] == mode), key=lambda row: row["error_px"])
        positions = np.linspace(0, len(candidates) - 1, quotas[mode]).round().astype(int)
        selected.extend(candidates[index] for index in positions)
    if len({row["sample_id"] for row in selected}) != count:
        raise AssertionError("Representative selection unexpectedly duplicated a sample")
    return sorted(selected, key=lambda row: (NOISE_MODES.index(row["noise_mode"]), row["error_px"], row["sample_id"]))


def difficulty(row: dict) -> str:
    error = float(row["error_px"])
    if error <= 0.5:
        return "sub-pixel success"
    if error <= 5:
        return "within 5 px tolerance"
    return "failure at 5 px tolerance"


def confidence_threshold(rows: list[dict]) -> float:
    return float(np.median([float(row["confidence"]) for row in rows]))


def save_graphs(rows: list[dict], output: Path, threshold: float) -> None:
    output.mkdir(parents=True, exist_ok=True)
    errors = np.asarray([float(row["error_px"]) for row in rows])
    confidences = np.asarray([float(row["confidence"]) for row in rows])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plotted = False
    for tolerance in (1, 2, 5):
        truth = errors <= tolerance
        if not np.any(truth):
            continue
        xs = np.r_[np.inf, np.unique(confidences)[::-1], -np.inf]
        precision, recall = [], []
        for value in xs:
            positive = confidences >= value
            tp = np.sum(positive & truth)
            precision.append(float(tp / positive.sum()) if positive.any() else 1.0)
            recall.append(float(tp / truth.sum()))
        ax.plot(recall, precision, marker=".", label=f"correct at {tolerance} px")
        plotted = True
    if not plotted:
        ax.text(.5, .5, "No positive examples at selected tolerances", ha="center", va="center")
    ax.set(xlabel="Recall", ylabel="Precision", title="Precision–recall from frozen matcher confidence", xlim=(0, 1), ylim=(0, 1.05))
    ax.grid(alpha=.25); ax.legend(loc="lower left"); fig.tight_layout(); fig.savefig(output / "precision_recall.png", dpi=130); plt.close(fig)

    tolerances = np.asarray(TOLERANCES)
    accuracy = [np.mean(errors <= value) for value in tolerances]
    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.plot(tolerances, np.asarray(accuracy) * 100, marker="o")
    ax.set(xlabel="Tolerance (pixels)", ylabel="Correct localizations (%)", title="Accuracy versus localization tolerance", xscale="symlog", ylim=(0, 105))
    ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(output / "accuracy_vs_tolerance.png", dpi=130); plt.close(fig)

    for tolerance in range(1, 6):
        actual = errors <= tolerance
        predicted = confidences >= threshold
        matrix = np.array([[np.sum(actual & predicted), np.sum(actual & ~predicted)], [np.sum(~actual & predicted), np.sum(~actual & ~predicted)]])
        fig, ax = plt.subplots(figsize=(4.6, 4)); image = ax.imshow(matrix, cmap="Blues")
        for (y, x), value in np.ndenumerate(matrix): ax.text(x, y, str(int(value)), ha="center", va="center", fontsize=14)
        ax.set(xticks=(0, 1), yticks=(0, 1), xticklabels=("positive", "negative"), yticklabels=("correct", "incorrect"), xlabel="Predicted from confidence", ylabel=f"Actual: error ≤ {tolerance} px", title=f"Tolerance {tolerance} px; confidence ≥ {threshold:.3f}")
        fig.colorbar(image, ax=ax, shrink=.8); fig.tight_layout(); fig.savefig(output / f"confusion_matrix_{tolerance}px.png", dpi=130); plt.close(fig)

    grouped = {mode: [row for row in rows if row["noise_mode"] == mode] for mode in NOISE_MODES}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tolerance, marker in ((1, "o"), (2, "s"), (5, "^")):
        ax.plot(NOISE_MODES, [100 * np.mean([row["error_px"] <= tolerance for row in grouped[mode]]) for mode in NOISE_MODES], marker=marker, label=f"Accuracy@{tolerance}px")
    ax.set(ylabel="Correct localizations (%)", title="Noise-stress accuracy"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(output / "noise_stress_accuracy.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5)); means = [np.mean([row["error_px"] for row in grouped[mode]]) for mode in NOISE_MODES]; medians = [np.median([row["error_px"] for row in grouped[mode]]) for mode in NOISE_MODES]
    ax.plot(NOISE_MODES, means, marker="o", label="Mean error"); ax.plot(NOISE_MODES, medians, marker="s", label="Median error")
    ax.set(ylabel="Error (pixels)", title="Noise-stress localization error"); ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(output / "noise_stress_error.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5)); ax.hist(errors, bins=min(20, max(6, len(errors) // 2)), color="#4c78a8", edgecolor="white")
    for label, value, color in (("median", np.median(errors), "#54a24b"), ("P90", np.percentile(errors, 90), "#e45756"), ("P95", np.percentile(errors, 95), "#f2cf5b")):
        ax.axvline(value, color=color, linestyle="--", label=f"{label}: {value:.3f}")
    ax.set(xlabel="Euclidean localization error (pixels)", ylabel="Samples", title="Full-pool error distribution"); ax.legend(); fig.tight_layout(); fig.savefig(output / "error_distribution.png", dpi=130); plt.close(fig)


def copy_selected(selected: list[dict], evidence: Path, pool: Path) -> list[dict]:
    root = evidence / "selected_30"
    for folder in (root / "reference", root / "search", root / "overlays"):
        folder.mkdir(parents=True, exist_ok=True)
    metadata = []
    for number, row in enumerate(selected, 1):
        pair = f"pair_{number:03d}.png"
        source_ref = ROOT / row["reference_image_path"]
        source_search = ROOT / row["search_image_path"]
        destination_ref = root / "reference" / pair
        destination_search = root / "search" / pair
        shutil.copy2(source_ref, destination_ref); shutil.copy2(source_search, destination_search)
        result = {"center_x": row["Output_X"], "center_y": row["Output_Y"]}
        save_localization_result(cv2.imread(str(source_search), cv2.IMREAD_GRAYSCALE), root / "overlays" / pair, (row["GTX"], row["GTY"]), result, row["confidence"], row["error_px"])
        record = {
            "sample_id": row["sample_id"], "reference_image": f"reference/{pair}", "search_image": f"search/{pair}",
            "GTX": row["GTX"], "GTY": row["GTY"], "Output_X": row["Output_X"], "Output_Y": row["Output_Y"],
            "error_px": row["error_px"], "confidence": row["confidence"], "noise_mode": row["noise_mode"],
            "rotation_deg": row["rotation_deg"], "difficulty": difficulty(row),
            "challenge_description": f"{row['noise_mode']} generator profile; rotation {row['rotation_deg']:.2f}°; measured error {row['error_px']:.3f} px.",
        }
        metadata.append(record)
    fields = list(metadata[0])
    write_csv(root / "metadata.csv", metadata, fields)
    lines = ["# Selected-pair comments", "", "The deterministic selection is stratified by noise mode and evenly spaced error rank within each mode; it includes both low-error and high-error examples rather than the easiest 30.", ""]
    for number, row in enumerate(metadata, 1):
        lines.extend([f"## Pair {number:02d} — {row['sample_id']}", "", f"Noise level: {row['noise_mode']}; rotation: {float(row['rotation_deg']):.2f}°", f"GT: ({float(row['GTX']):.6f}, {float(row['GTY']):.6f}); prediction: ({float(row['Output_X']):.6f}, {float(row['Output_Y']):.6f})", f"Error: {float(row['error_px']):.6f} px; confidence: {float(row['confidence']):.6f}", "", f"Challenge: {row['challenge_description']}", f"Result: {row['difficulty']}.", ""])
    (root / "pair_comments.md").write_text("\n".join(lines), encoding="utf-8")
    return metadata


def rgb_regression(row: dict) -> dict:
    with tempfile.TemporaryDirectory(prefix="rgb_compat_") as temp:
        temp = Path(temp); reference = cv2.imread(str(ROOT / row["reference_image_path"]), cv2.IMREAD_GRAYSCALE); search = cv2.imread(str(ROOT / row["search_image_path"]), cv2.IMREAD_GRAYSCALE)
        rgb_ref, rgb_search = temp / "reference_rgb.png", temp / "search_rgb.png"
        cv2.imwrite(str(rgb_ref), cv2.cvtColor(reference, cv2.COLOR_GRAY2BGR)); cv2.imwrite(str(rgb_search), cv2.cvtColor(search, cv2.COLOR_GRAY2BGR))
        gray = predict(str(ROOT / row["reference_image_path"]), str(ROOT / row["search_image_path"])); rgb = predict(str(rgb_ref), str(rgb_search))
        delta = float(np.hypot(float(gray["center_x"]) - float(rgb["center_x"]), float(gray["center_y"]) - float(rgb["center_y"])))
    return {"tested": True, "supported": bool(delta <= 1e-6), "prediction_delta_px": delta, "method": "Equivalent three-channel PNG paths are decoded through localize.py's grayscale image loader."}


def write_reports(evidence: Path, rows: list[dict], selected: list[dict], selected_meta: list[dict], seed: int, threshold: float, rgb: dict) -> None:
    results = evidence / "results"; results.mkdir(parents=True, exist_ok=True)
    full = metric_summary(rows); chosen = metric_summary(selected)
    by_noise = {mode: metric_summary([row for row in rows if row["noise_mode"] == mode]) for mode in NOISE_MODES}
    payload = {"full_evaluation_pool": full, "selected_30": chosen, "noise_stress": by_noise, "evaluation": {"generator": "FinFETSEMDatasetGenerator used by generate_dataset.py", "localizer": "localize.predict -> src.localization.inference.localize", "seed_base": seed, "pairs_per_noise_mode": len(rows) // len(NOISE_MODES), "confidence_threshold_for_confusion_matrices": threshold, "selection_policy": "Stratified by clean/low/medium/high, then evenly spaced by observed error rank within each stratum.", "interpretation": "Synthetic center-rule consistency evidence only; it is not an external real-image benchmark."}, "rgb_compatibility": rgb}
    (results / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# Measured localization performance", "", "## Interpretation", "", "These measurements use a fresh fixed-seed synthetic FinFET pool generated by the repository's frozen public generator and labels from its corrected centre-rule contract. They demonstrate measured consistency with that synthetic contract; they do **not** establish accuracy on independently labelled real SEM imagery.", "", "## Full evaluation pool", "", "| Metric | Value |", "| --- | ---: |"]
    for key, value in full.items(): lines.append(f"| {key} | {value:.6f} |" if isinstance(value, float) else f"| {key} | {value} |")
    lines.extend(["", "## Selected 30 examples", "", "The selected 30 are intentionally diverse examples, not an unbiased benchmark. Their results are reported separately.", "", "| Metric | Value |", "| --- | ---: |"])
    for key, value in chosen.items(): lines.append(f"| {key} | {value:.6f} |" if isinstance(value, float) else f"| {key} | {value} |")
    lines.extend(["", "## Noise stress", "", "| Mode | Samples | Accuracy@1 px | Accuracy@2 px | Accuracy@5 px | Mean error | Median error | P95 error |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"])
    for mode, values in by_noise.items(): lines.append(f"| {mode} | {values['samples']} | {values['accuracy_at_1px']:.3f} | {values['accuracy_at_2px']:.3f} | {values['accuracy_at_5px']:.3f} | {values['mean_error_px']:.3f} | {values['median_error_px']:.3f} | {values['p95_error_px']:.3f} |")
    lines.extend(["", "## Confidence-derived classification figures", "", f"For each tolerance, actual positive means `error ≤ tolerance`; predicted positive means `confidence ≥ {threshold:.6f}` (the full-pool median confidence). These matrices characterize confidence calibration against localization correctness.", "", f"RGB input compatibility: {'supported' if rgb['supported'] else 'not supported'} for an equivalent three-channel PNG path; coordinate delta = {rgb['prediction_delta_px']:.6f} px."])
    (results / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    requirements = [line for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]
    (results / "environment.md").write_text("# Environment\n\n" + f"- Python: `{sys.version}`\n- Platform: `{platform.platform()}`\n- Production/runtime environment: [`requirements.txt`](../../requirements.txt)\n- Complete development-environment record: [`requirements-freeze.txt`](../../requirements-freeze.txt)\n\n## Condensed runtime dependencies\n\n```text\n" + "\n".join(requirements) + "\n```\n\nTorch, torchvision, kornia, kornia_rs, and LightGlue are not production runtime dependencies.\n", encoding="utf-8")
    worst = max(selected_meta, key=lambda row: float(row["error_px"]))
    failure = evidence / "failure_cases"; failure.mkdir(parents=True, exist_ok=True)
    shutil.copy2(evidence / "selected_30" / "overlays" / Path(worst["search_image"]).name, failure / "failure_case_01.png")
    (failure / "FAILURE_ANALYSIS.md").write_text(f"# Highest-error selected case\n\n- Sample ID: `{worst['sample_id']}`\n- Reference: [`../selected_30/{worst['reference_image']}`](../selected_30/{worst['reference_image']})\n- Search: [`../selected_30/{worst['search_image']}`](../selected_30/{worst['search_image']})\n- Ground truth: ({float(worst['GTX']):.6f}, {float(worst['GTY']):.6f})\n- Prediction: ({float(worst['Output_X']):.6f}, {float(worst['Output_Y']):.6f})\n- Error: {float(worst['error_px']):.6f} px\n- Confidence: {float(worst['confidence']):.6f}\n- Generator metadata: noise={worst['noise_mode']}, rotation={float(worst['rotation_deg']):.2f}°\n\nThis is the selected subset's genuine highest measured error. It is a failure at every tolerance below its measured error; whether it is a failure at 5 px is stated by its recorded difficulty label. Periodic FinFET geometry creates visually similar candidate regions, while the actual noise profile and rotation further reduce contrast. No matcher, candidate, ranking, or label was changed for this analysis.\n", encoding="utf-8")
    readme = f"# Submission evaluation evidence\n\nThis package records a fixed-seed FinFET synthetic evaluation generated with the existing public generator and evaluated with the exact production path used by `localize.py`.\n\n- Full evaluation pool: {len(rows)} pairs ({len(rows) // 4} per actual generator noise mode), seed base `{seed}`\n- Selected examples: 30 pairs, deterministically stratified by noise mode and observed error rank\n- Full-pool Accuracy@1 px: {full['accuracy_at_1px']:.2%}; Accuracy@5 px: {full['accuracy_at_5px']:.2%}\n- Selected-30 Accuracy@1 px: {chosen['accuracy_at_1px']:.2%}; Accuracy@5 px: {chosen['accuracy_at_5px']:.2%}\n\n## Evidence files\n\n- [Result CSV](results/localization_results.csv) and [measured metrics](results/metrics.md)\n- [Environment](results/environment.md)\n- [Selected pair metadata](selected_30/metadata.csv), [comments](selected_30/pair_comments.md), and overlays (green = GT; red = prediction)\n- [Graphs](graphs/) including PR, tolerance, confidence-calibration matrices, noise stress, and error distribution\n- [Failure analysis](failure_cases/FAILURE_ANALYSIS.md)\n- [Scoring utility status](scoring_utility/README.md)\n\n## Interpretation\n\nThe full-pool figure is the honest measured result for this fixed synthetic pool. It is a test of consistency with the generator's corrected centre-rule labels, not an independently labelled real-SEM benchmark. The selected 30 are deliberately diverse examples and are reported separately.\n"
    readme = readme.replace("- [Scoring utility status](scoring_utility/README.md)\n", "")
    reproducibility_note = (
        "The full source pool is intentionally not committed. The result CSV retains "
        "reproducible relative `evidence_pool/<mode>/...` paths and the fixed seed; "
        "the committed selected-30 metadata points to the included copies."
    )
    readme = readme.replace("\n\n## Evidence files", f"\n\n{reproducibility_note}\n\n## Evidence files")
    (evidence / "README.md").write_text(readme, encoding="utf-8")


def main() -> None:
    configure_console_encoding()
    parser = argparse.ArgumentParser(description="Build fixed-seed FinFET submission evidence without changing production localization.")
    parser.add_argument("--pairs-per-noise", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--pool-dir", type=Path, default=ROOT / "evidence_pool")
    parser.add_argument("--evidence-dir", type=Path, default=ROOT / "submission_evidence")
    args = parser.parse_args()
    if args.pairs_per_noise < 8:
        parser.error("--pairs-per-noise must be at least 8 for the 30-pair stratified selection")
    if args.evidence_dir.exists() or args.pool_dir.exists():
        parser.error("Refusing to overwrite an existing evidence directory or evaluation pool")
    groups = generate_pool(args.pool_dir, args.pairs_per_noise, args.seed)
    rows = evaluate_pool(groups)
    results = args.evidence_dir / "results"
    fields = ["search_image_path", "GTX", "GTY", "Output_X", "Output_Y", "sample_id", "reference_image_path", "error_px", "confidence", "noise_mode", "rotation_deg"]
    write_csv(results / "localization_results.csv", rows, fields)
    selected = choose_representative(rows)
    selected_metadata = copy_selected(selected, args.evidence_dir, args.pool_dir)
    threshold = confidence_threshold(rows)
    save_graphs(rows, args.evidence_dir / "graphs", threshold)
    rgb = rgb_regression(selected[0])
    write_reports(args.evidence_dir, rows, selected, selected_metadata, args.seed, threshold, rgb)
    print(json.dumps({"full_evaluation_pool": metric_summary(rows), "selected_30": metric_summary(selected), "rgb_compatibility": rgb}, indent=2))


if __name__ == "__main__":
    main()
