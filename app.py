"""Streamlit UI for existing SEM generation, localization, and evaluation paths."""
from __future__ import annotations

import csv
import gc
import hashlib
import json
import subprocess
import sys
import time
from io import BytesIO
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Keep the generator's established direct-module import contract intact.
SOURCE_ROOT = Path(__file__).resolve().parent / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from dataset_generator import FinFETSEMDatasetGenerator, SyntheticDatasetGenerator
from src.localization.classical_matcher import ClassicalSEMLocalizer
from src.localization.config import LocalizationConfig
from src.localization.dataset import SEMLocalizationDataset
from src.localization.external_ground_truth import (GroundTruthValidationError,
                                                    parse_external_ground_truth,
                                                    resolve_external_ground_truth)
from src.localization.multi_evaluation import (build_combinations, evaluate_variant_pair,
                                               metrics as multi_metrics, normalize_sample_id,
                                               write_experiment)
from src.localization.visualize import render_localization_result


FOOTPRINT_PX = 100
NOISE_MODES = ("clean", "low", "medium", "high")
IMAGE_TYPES = ["png", "jpg", "jpeg", "tiff", "bmp"]
POSTPROCESS_DEFAULTS = {
    "postprocess_gamma": 1.0,
    "postprocess_vignetting": 0.20,
    "postprocess_astigmatism": 1.0,
    "postprocess_charging": 0.0,
    "postprocess_streaks": 0.0,
}


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def ground_truth(metadata: dict) -> tuple[float, float]:
    center = metadata["ground_truth_center"]
    return float(center["x"]), float(center["y"])


def localize(search: np.ndarray, reference: np.ndarray) -> dict:
    """Call the production localizer and retain its measured execution time."""
    started = time.perf_counter()
    result = dict(ClassicalSEMLocalizer().localize(search, reference))
    result.setdefault("runtime_s", time.perf_counter() - started)
    return result


def errors(result: dict, gt: tuple[float, float]) -> tuple[float, float, float]:
    dx = abs(float(result["center_x"]) - gt[0])
    dy = abs(float(result["center_y"]) - gt[1])
    return dx, dy, float(np.hypot(dx, dy))


def result_status(error_px: float | None) -> str:
    if error_px is None:
        return "Missing Ground Truth"
    if error_px <= 2:
        return "PASS @2px"
    if error_px <= 5:
        return "PASS @5px"
    if error_px <= 10:
        return "PASS @10px"
    if error_px <= 20:
        return "PASS @20px"
    return "FAILED"


def generate_one(architecture: str, seed: int, noise_mode: str, sample_id: int) -> dict:
    """Use the unmodified project generators for one exact image pair."""
    if architecture == "FinFET":
        generator = FinFETSEMDatasetGenerator(seed=seed, noise_mode=noise_mode)
        reference, search, metadata = generator.generate_image_pair(sample_id)
        del generator
        gc.collect()
    else:
        generator = SyntheticDatasetGenerator(architecture="dram", seed=seed + sample_id)
        reference, search, metadata = generator.create_image_pair(pair_idx=sample_id)
        metadata.setdefault("noise_mode", "procedural")
    return {"architecture": architecture, "noise_mode": noise_mode,
            "sample_id": sample_id, "search": search, "reference": reference,
            "metadata": metadata}


def generate_all_noise(seed: int, sample_id: int, on_variant=None) -> list[dict]:
    """Generate existing FinFET presets while resetting RNG for shared geometry."""
    generator = FinFETSEMDatasetGenerator(seed=seed, noise_mode="clean")
    pairs = []
    for index, noise_mode in enumerate(NOISE_MODES, start=1):
        # Geometry choices occur before the preset-dependent noise stage. Resetting
        # the existing generator's RNG keeps those choices identical across presets.
        np.random.seed(seed)
        generator.noise_mode = noise_mode
        reference, search, metadata = generator.generate_image_pair(sample_id)
        pairs.append({"architecture": "FinFET", "noise_mode": noise_mode,
                      "sample_id": sample_id, "search": search, "reference": reference,
                      "metadata": metadata})
        if on_variant is not None:
            on_variant(index, len(NOISE_MODES), noise_mode)
    del generator
    gc.collect()
    return pairs


def high_noise_quality(search: np.ndarray) -> tuple[dict, str | None]:
    """Report structure-only quality signals; never use localization success as a gate."""
    image = search.astype(np.float32)
    gradients = cv2.Sobel(image, cv2.CV_32F, 1, 0) ** 2 + cv2.Sobel(image, cv2.CV_32F, 0, 1) ** 2
    metrics = {"std": float(image.std()), "gradient_energy": float(gradients.mean()),
               "non_saturated_fraction": float(np.mean((image > 3) & (image < 252)))}
    warning = None
    if metrics["std"] < 5 or metrics["gradient_energy"] < 2 or metrics["non_saturated_fraction"] < 0.10:
        warning = "High-noise quality warning: structural contrast may be unusually weak; the image was not regenerated."
    return metrics, warning


def postprocess_values() -> dict:
    return {key.removeprefix("postprocess_"): st.session_state[key] for key in POSTPROCESS_DEFAULTS}


def apply_corner_rounding(image: np.ndarray, rounding_px: float) -> np.ndarray:
    """Correctly size the existing morphology-based corner-rounding operation.

    The original helper used ``radius`` as a kernel side length and imposed a
    3-pixel minimum, making several slider values collapse to the same 3x3
    operation.  Here the UI value is a radius, so the elliptical footprint is
    explicitly ``(2r + 1)`` and grows monotonically.
    """
    radius = int(round(rounding_px))
    if radius <= 0:
        return image.copy()
    image_float = image.astype(np.float32)
    # Percentile thresholding fails when a valid bright feature covers more
    # than ~18% of the patch (the percentile itself becomes 255). Otsu keeps
    # the same intensity-mask approach but obtains an actual feature mask.
    _, mask = cv2.threshold(image.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if not mask.any():
        return image.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    rounded = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    rounded = cv2.morphologyEx(rounded, cv2.MORPH_CLOSE, kernel, iterations=1)
    rounded = cv2.GaussianBlur(rounded.astype(np.float32), (0, 0), max(0.3, radius / 3.0))
    return np.clip(cv2.addWeighted(image_float, 0.82, rounded, 0.18, 0), 0, 255).astype(np.uint8)


def apply_existing_postprocessing(image: np.ndarray, params: dict, direction: str) -> np.ndarray:
    """Apply only pre-existing generator filter implementations to an existing image."""
    result = image.copy()
    # These helpers do not depend on generator state; calling them unbound avoids
    # loading base layouts and, crucially, does not regenerate geometry.
    if params["astigmatism"] > 0:
        result = FinFETSEMDatasetGenerator._apply_astigmatism_blur(
            None, result, direction=direction, strength=params["astigmatism"])
    if params["vignetting"] > 0:
        result = FinFETSEMDatasetGenerator._apply_vignetting(None, result, strength=params["vignetting"])
    if params["charging"] > 0:
        result = FinFETSEMDatasetGenerator._apply_charging_effect(None, result, intensity=params["charging"])
    if params["streaks"] > 0:
        count = max(1, round(params["streaks"] * result.shape[0] / 100.0))
        # Fix the realization for a stable preview; the filter itself is unchanged.
        state = np.random.get_state()
        np.random.seed(2026)
        try:
            result = FinFETSEMDatasetGenerator._apply_charging_streaks(
                None, result, direction=direction, num_streaks=count)
        finally:
            np.random.set_state(state)
    if params["gamma"] != 1.0:
        result = FinFETSEMDatasetGenerator._adjust_contrast_gamma(None, result, gamma=params["gamma"])
    return result


def filtered_pair(pair: dict, params: dict) -> dict:
    """Derive a non-cumulative preview from the preserved original pixel arrays."""
    source_search = pair.get("original_search", pair["search"])
    source_reference = pair.get("original_reference", pair["reference"])
    output = dict(pair)
    output["original_search"] = source_search
    output["original_reference"] = source_reference
    output["search"] = apply_existing_postprocessing(source_search, params, "vertical")
    output["reference"] = apply_existing_postprocessing(source_reference, params, "horizontal")
    output["postprocess_params"] = params
    return output


def save_filtered_pair(pair: dict) -> Path:
    folder = output_path() / str(pair["noise_mode"]).lower()
    folder.mkdir(parents=True, exist_ok=True)
    sample_id = int(pair["sample_id"])
    cv2.imwrite(str(folder / f"search_{sample_id:04d}_filtered.png"), pair["search"])
    cv2.imwrite(str(folder / f"reference_{sample_id:04d}_filtered.png"), pair["reference"])
    payload = {"architecture": pair["architecture"], "noise_mode": pair["noise_mode"],
               "sample_id": sample_id, "ground_truth_center": pair["metadata"]["ground_truth_center"],
               "postprocess": pair["postprocess_params"],
               "search": f"search_{sample_id:04d}_filtered.png",
               "reference": f"reference_{sample_id:04d}_filtered.png"}
    (folder / f"metadata_{sample_id:04d}_filtered.json").write_text(
        json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return folder


def reset_postprocess() -> None:
    for key, value in POSTPROCESS_DEFAULTS.items():
        st.session_state[key] = value
    st.session_state.postprocess_preview = None


def output_path() -> Path:
    return Path(st.session_state.output_directory).expanduser()


def save_pair(pair: dict) -> Path:
    folder = output_path() / str(pair["noise_mode"]).lower()
    folder.mkdir(parents=True, exist_ok=True)
    sample_id = int(pair["sample_id"])
    cv2.imwrite(str(folder / f"search_{sample_id:04d}.png"), pair["search"])
    cv2.imwrite(str(folder / f"reference_{sample_id:04d}.png"), pair["reference"])
    metadata = dict(pair["metadata"])
    metadata.update({"architecture": pair["architecture"], "noise_mode": pair["noise_mode"],
                     "search": f"search_{sample_id:04d}.png",
                     "reference": f"reference_{sample_id:04d}.png"})
    (folder / f"metadata_{sample_id:04d}.json").write_text(
        json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8")
    # Keep each noise folder directly consumable by the repository's existing
    # dataset loader; this is its documented annotations.json format.
    manifest_path = folder / "annotations.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {"pairs": []}
    record = {"pair_id": sample_id, "reference_path": f"reference_{sample_id:04d}.png",
              "search_path": f"search_{sample_id:04d}.png",
              "ground_truth_center": metadata["ground_truth_center"],
              "noise_mode": pair["noise_mode"], "architecture": pair["architecture"]}
    manifest["pairs"] = [item for item in manifest.get("pairs", []) if str(item.get("pair_id")) != str(sample_id)]
    manifest["pairs"].append(record)
    manifest_path.write_text(json.dumps(manifest, indent=2, default=_json_default), encoding="utf-8")
    return folder


def save_generated_prediction(pair: dict, result: dict, error_px: float) -> Path:
    folder = save_pair(pair)
    sample_id = int(pair["sample_id"])
    gt = ground_truth(pair["metadata"])
    overlay = render_localization_result(pair["search"], gt, result,
                                          result.get("confidence"), error_px,
                                          reference_size=FOOTPRINT_PX)
    cv2.imwrite(str(folder / f"prediction_{sample_id:04d}.png"), overlay)
    dx, dy, _ = errors(result, gt)
    payload = {"architecture": pair["architecture"], "noise_level": pair["noise_mode"],
               "ground_truth": {"x": gt[0], "y": gt[1]},
               "prediction": result, "error": {"x": dx, "y": dy, "euclidean_px": error_px}}
    (folder / f"prediction_{sample_id:04d}.json").write_text(
        json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return folder


def render_pair(pair: dict, result: dict | None = None, error_px: float | None = None) -> None:
    left, right = st.columns(2)
    with left:
        st.caption("Reference")
        st.image(pair["reference"], clamp=True, use_container_width=True)
    with right:
        gt_center = pair.get("metadata", {}).get("ground_truth_center")
        gt = (float(gt_center["x"]), float(gt_center["y"])) if gt_center else None
        st.caption("Search" if result is None else ("Search — green GT, red prediction" if gt else "Search — red prediction"))
        image = pair["search"] if result is None else render_localization_result(
            pair["search"], gt, result, result.get("confidence"), error_px,
            reference_size=FOOTPRINT_PX)
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image, clamp=True, use_container_width=True)


def render_filter_preview(before: dict, after: dict) -> None:
    ref_before, ref_after = st.columns(2)
    with ref_before:
        st.caption("Reference Before")
        st.image(before.get("original_reference", before["reference"]), clamp=True, use_container_width=True)
    with ref_after:
        st.caption("Reference After")
        st.image(after["reference"], clamp=True, use_container_width=True)
    search_before, search_after = st.columns(2)
    with search_before:
        st.caption("Search Before")
        st.image(before.get("original_search", before["search"]), clamp=True, use_container_width=True)
    with search_after:
        st.caption("Search After")
        st.image(after["search"], clamp=True, use_container_width=True)


def legacy_postprocess_section(pairs: list[dict]) -> None:
    """Compact UI for the subset of existing filters that can act on saved pixels."""
    with st.expander("Post Processing", expanded=False):
        st.caption("Works on the current generated pixels only — it never regenerates layout or runs localization. "
                   "Original files remain unchanged; filtered files are saved with `_filtered` suffixes.")
        active = st.selectbox("Current Noise Level", range(len(pairs)),
                              format_func=lambda index: f"{str(pairs[index]['noise_mode']).title()} — sample {pairs[index]['sample_id']}",
                              key="postprocess_active_pair")
        with st.expander("Geometry", expanded=False):
            st.caption("No geometry post-processing filter is exposed for existing pixels.")
        with st.expander("SEM Imaging Physics", expanded=False):
            st.slider("Astigmatism strength (existing backend)", 0.5, 2.0, step=0.05,
                      key="postprocess_astigmatism")
            st.caption("Beam spot size is not independently exposed by the current implementation.")
        with st.expander("Acquisition Noise", expanded=False):
            st.slider("Charging gradient strength (existing backend)", 0.0, 1.0, step=0.01,
                      key="postprocess_charging")
            st.slider("Charging streaks (per 100 rows)", 0.0, 5.0, step=0.1,
                      key="postprocess_streaks")
            st.caption("Dose, raster drift, row jitter, streak intensity, speckle, and salt-and-pepper controls are not implemented separately.")
        with st.expander("Distortion & Polygon Scaling", expanded=False):
            st.slider("Vignette strength (existing backend)", 0.0, 1.0, step=0.01,
                      key="postprocess_vignetting")
            st.slider("Gamma (existing backend contrast curve)", 0.4, 2.5, step=0.05,
                      key="postprocess_gamma")
            st.caption("Linewidth/CD bias and barrel/pincushion distortion are not implemented in the current renderer.")
        with st.expander("Die Layout", expanded=False):
            st.caption("Array block size, separator-strip width, and boundary-straddling probability are generation-layout parameters not implemented for post-processing.")

        preview, apply_current, apply_all, reset = st.columns(4)
        params = postprocess_values()
        if preview.button("Preview Filter"):
            st.session_state.postprocess_preview = {"index": active, "pair": filtered_pair(pairs[active], params),
                                                    "params": params}
        if apply_current.button("Apply Filter to Current Level"):
            updated = filtered_pair(pairs[active], params)
            pairs = list(pairs)
            pairs[active] = updated
            st.session_state.generated_pairs = pairs
            st.session_state.filtered_generated_pairs = pairs
            folder = save_filtered_pair(updated)
            st.session_state.postprocess_preview = {"index": active, "pair": updated, "params": params}
            st.success(f"Saved filtered images and metadata to {folder}")
        if apply_all.button("Apply Filter to All", disabled=len(pairs) < 2):
            updated_pairs = [filtered_pair(pair, params) for pair in pairs]
            st.session_state.generated_pairs = updated_pairs
            st.session_state.filtered_generated_pairs = updated_pairs
            for pair in updated_pairs:
                save_filtered_pair(pair)
            st.session_state.postprocess_preview = {"index": active, "pair": updated_pairs[active], "params": params}
            st.success(f"Saved filtered images for all {len(updated_pairs)} generated levels to {output_path()}")
        reset.button("Reset Filters", on_click=reset_postprocess)
        preview_record = st.session_state.postprocess_preview
        if preview_record:
            st.divider()
            st.caption("Processed Preview")
            original = pairs[preview_record["index"]]
            render_filter_preview(original, preview_record["pair"])


def postprocess_section(pairs: list[dict]) -> None:
    """Post-processing controls shown only inside the Generate Pairs workflow."""
    with st.container(border=True):
        st.subheader("Post Processing")
        if not pairs:
            st.caption("Generate a pair to enable post-processing.")
            return
        st.caption("Uses the current in-session pixels only. Preview never saves or regenerates geometry.")
        if st.session_state.get("postprocess_active_pair", 0) not in range(len(pairs)):
            st.session_state.pop("postprocess_active_pair", None)
        active = st.selectbox("Active generated pair", range(len(pairs)),
                              format_func=lambda index: f"{str(pairs[index]['noise_mode']).title()} — sample {pairs[index]['sample_id']}",
                              key="postprocess_active_pair")
        with st.expander("SEM Imaging", expanded=False):
            st.slider("Astigmatism strength", 0.5, 2.0, step=0.05, key="postprocess_astigmatism")
        with st.expander("Acquisition", expanded=False):
            st.slider("Charging gradient strength", 0.0, 1.0, step=0.01, key="postprocess_charging")
            st.slider("Charging streaks (per 100 rows)", 0.0, 5.0, step=0.1, key="postprocess_streaks")
        with st.expander("Appearance / Distortion", expanded=False):
            st.slider("Vignette strength", 0.0, 1.0, step=0.01, key="postprocess_vignetting")
            st.slider("Gamma", 0.4, 2.5, step=0.05, key="postprocess_gamma")
        params = postprocess_values()
        if st.button("Preview Filter", use_container_width=True):
            st.session_state.postprocess_preview = {"index": active, "pair": filtered_pair(pairs[active], params),
                                                    "params": params}
        if st.button("Apply Filter", use_container_width=True):
            updated = filtered_pair(pairs[active], params)
            updated_pairs = list(pairs)
            updated_pairs[active] = updated
            st.session_state.generated_pairs = updated_pairs
            st.session_state.filtered_generated_pairs = updated_pairs
            folder = save_filtered_pair(updated)
            st.session_state.postprocess_preview = {"index": active, "pair": updated, "params": params}
            st.success(f"Saved filtered images to {folder}")
        st.button("Reset Filters", on_click=reset_postprocess, use_container_width=True)


def show_result(result: dict, gt: tuple[float, float] | None = None) -> float | None:
    st.write(f"Prediction: ({float(result['center_x']):.2f}, {float(result['center_y']):.2f})")
    if gt is None:
        st.info("Ground truth not provided — localization error cannot be calculated.")
        error_px = None
    else:
        dx, dy, error_px = errors(result, gt)
        st.write(f"Ground Truth: ({gt[0]:.2f}, {gt[1]:.2f})")
        st.write(f"X Error: {dx:.2f} px · Y Error: {dy:.2f} px · Pixel Error: {error_px:.2f} px")
        status = result_status(error_px)
        (st.error if status == "FAILED" else st.success)(status)
    values = {"Confidence": result.get("confidence"), "Scale": result.get("scale"),
              "Rotation": result.get("rotation"), "Runtime (s)": result.get("runtime_s")}
    st.caption(" · ".join(f"{name}: {value:.4g}" if isinstance(value, (int, float)) else f"{name}: N/A"
                            for name, value in values.items()))
    return error_px


def scored_validation_rows(rows: list[dict]) -> list[dict]:
    """Exclude rows without GT from accuracy/error summaries and graphs."""
    return [row for row in rows if row.get("error_px") is not None and np.isfinite(float(row["error_px"]))]


def validation_summary(rows: list[dict]) -> None:
    rows = scored_validation_rows(rows)
    if not rows:
        st.info("No matched ground-truth records are available for accuracy/error metrics.")
        return
    values = np.asarray([row["error_px"] for row in rows], dtype=float)
    confidence = [row["confidence"] for row in rows if row["confidence"] is not None]
    metrics = {"Samples": len(rows), "Accuracy@2px": np.mean(values <= 2),
               "Accuracy@5px": np.mean(values <= 5), "Accuracy@10px": np.mean(values <= 10),
               "Accuracy@20px": np.mean(values <= 20), "Mean Error": values.mean(),
               "Median Error": np.median(values), "Mean Confidence": np.mean(confidence) if confidence else None}
    columns = st.columns(3)
    for index, (label, value) in enumerate(metrics.items()):
        if isinstance(value, float):
            shown = f"{value:.2%}" if label.startswith("Accuracy") else f"{value:.2f}"
        else:
            shown = "N/A" if value is None else str(value)
        columns[index % 3].metric(label, shown)


def decode_upload(upload) -> np.ndarray | None:
    if upload is None:
        return None
    image = cv2.imdecode(np.frombuffer(upload.getvalue(), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("The uploaded file could not be decoded as an image.")
    return image


def save_manual_prediction() -> Path:
    record = st.session_state.manual_prediction_result
    folder = output_path() / "manual_prediction" / datetime.now().strftime("%Y%m%d_%H%M%S")
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / "search.png"), record["search"])
    cv2.imwrite(str(folder / "reference.png"), record["reference"])
    cv2.imwrite(str(folder / "prediction.png"), record["overlay"])
    payload = {"ground_truth": record["ground_truth"], "prediction": record["result"],
               "error_px": record["error_px"]}
    (folder / "prediction.json").write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return folder


def run_evaluator(dataset_dir: str) -> dict:
    run_dir = output_path() / "dataset_validation" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "src.localization.evaluate", "--dataset-dir", dataset_dir,
               "--output-dir", str(run_dir), "--visualize-count", "20"]
    completed = subprocess.run(command, cwd=Path(__file__).resolve().parent,
                               capture_output=True, text=True, timeout=3600)
    if completed.returncode:
        raise RuntimeError(completed.stderr or completed.stdout or "Existing evaluator failed.")
    with (run_dir / "metrics.json").open(encoding="utf-8") as file:
        metrics = json.load(file)
    with (run_dir / "predictions.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    return {"dataset_dir": str(Path(dataset_dir)), "run_dir": run_dir, "metrics": metrics,
            "rows": rows, "console": completed.stdout}


def dataset_manifest_problem(dataset_dir: str) -> str | None:
    """Mirror the existing loader's root-level manifest requirement in the UI."""
    root = Path(dataset_dir).expanduser()
    if not root.is_dir():
        return f"Dataset directory does not exist: {root}"
    if (root / "annotations.json").is_file() or (root / "ground_truth.csv").is_file():
        return None
    return (f"{root} is not a dataset root. Select the folder that directly contains "
            "annotations.json or ground_truth.csv (for example, data/finfet).")


def _plain_image_records(root: Path, role: str) -> list[dict]:
    """Expose an evaluator's image directory only when external GT was supplied."""
    images = sorted((path for path in root.iterdir()
                     if path.is_file() and path.suffix.lstrip(".").casefold() in IMAGE_TYPES),
                    key=lambda path: path.name.casefold())
    if not images:
        raise ValueError(f"{role} directory contains no supported image files: {root}")
    records = []
    seen = set()
    for path in images:
        normalized_id = normalize_sample_id(path.name)
        if normalized_id in seen:
            raise ValueError(f"{role} directory has duplicate normalized sample ID {normalized_id}: {path.name}")
        seen.add(normalized_id)
        records.append({"sample_id": normalized_id, role.lower(): path.name,
                        "noise_mode": None, "process_type": "external"})
    return records


def _validation_records(directory: str, role: str, allow_plain_images: bool) -> tuple[Path, list[dict]]:
    root = Path(directory).expanduser()
    problem = dataset_manifest_problem(str(root))
    if problem is None:
        return root, SEMLocalizationDataset(root, LocalizationConfig()).records
    if not allow_plain_images:
        raise ValueError(problem)
    if not root.is_dir():
        raise ValueError(problem)
    records = _plain_image_records(root, role)
    # Give both sides a common field name used by the matching worker.
    return root, [{**record, "search": record.get("search", record.get("reference")),
                   "reference": record.get("reference", record.get("search"))} for record in records]


def _normalized_record_map(records: list[dict], role: str) -> dict[str, dict]:
    indexed = {}
    for record in records:
        identifier = normalize_sample_id(record["sample_id"])
        if identifier in indexed:
            raise ValueError(f"{role} dataset has duplicate normalized sample ID {identifier}.")
        indexed[identifier] = record
    return indexed


def validation_pair_records(search_dir: str, reference_dir: str,
                            external_gt: list[dict] | None = None) -> tuple[list[dict], list[str], dict]:
    """Resolve paired records and validate optional evaluator GT before matching begins."""
    allow_plain_images = external_gt is not None
    search_root, search_list = _validation_records(search_dir, "Search", allow_plain_images)
    reference_root, reference_list = _validation_records(reference_dir, "Reference", allow_plain_images)
    search_records = _normalized_record_map(search_list, "Search")
    reference_records = _normalized_record_map(reference_list, "Reference")
    ids = sorted(set(search_records) & set(reference_records), key=lambda value: (len(value), value))
    if not ids:
        raise ValueError("No matching normalized sample IDs were found between Search and Reference directories.")
    skipped = sorted((set(search_records) ^ set(reference_records)), key=lambda value: (len(value), value))
    pairs, used_gt = [], set()
    for identifier in ids:
        search_record, reference_record = search_records[identifier], reference_records[identifier]
        if external_gt is None:
            gt_record = {"center_x": float(search_record["center_x"]), "center_y": float(search_record["center_y"]),
                         "noise_mode": search_record.get("noise_mode")}
        else:
            gt_record = resolve_external_ground_truth(external_gt, identifier,
                                                       search_record["search"], reference_record["reference"])
            if gt_record is not None:
                search = cv2.imread(str(search_root / search_record["search"]), cv2.IMREAD_GRAYSCALE)
                if search is None:
                    raise ValueError(f"Ground truth validation failed: cannot read Search image {search_record['search']}.")
                height, width = search.shape[:2]
                x, y = float(gt_record["center_x"]), float(gt_record["center_y"])
                if not (0.0 <= x < width):
                    raise GroundTruthValidationError(
                        f"Ground truth validation failed: sample_id {identifier} has center_x={x}, outside Search width {width}."
                    )
                if not (0.0 <= y < height):
                    raise GroundTruthValidationError(
                        f"Ground truth validation failed: sample_id {identifier} has center_y={y}, outside Search height {height}."
                    )
                used_gt.add(id(gt_record))
        pairs.append({"sample_id": identifier, "search_root": search_root, "reference_root": reference_root,
                      "search_record": search_record, "reference_record": reference_record, "gt_record": gt_record})
    summary = {"external_gt": external_gt is not None,
               "matched_gt": sum(pair["gt_record"] is not None for pair in pairs),
               "missing_gt": sum(pair["gt_record"] is None for pair in pairs),
               "unused_gt": len(external_gt or []) - len(used_gt),
               "has_noise_metadata": bool([pair for pair in pairs if pair["gt_record"] is not None]) and all(
                   bool(pair["gt_record"].get("noise_mode")) for pair in pairs if pair["gt_record"] is not None)}
    return pairs, skipped, summary


def noise_dataset_roots(dataset_dir: str) -> dict[str, Path]:
    """Find existing per-noise dataset roots without inventing a new file format."""
    root = Path(dataset_dir).expanduser()
    found = {}
    for mode in NOISE_MODES:
        for candidate in (root / mode, root / f"output_{mode}"):
            if (candidate / "annotations.json").is_file() or (candidate / "ground_truth.csv").is_file():
                found[mode] = candidate
                break
    return found


def noise_comparison_rows(search_dir: str, reference_dir: str, sample_id: str | None = None,
                          on_progress=None, external_gt: list[dict] | None = None,
                          return_gt_summary: bool = False):
    """Match exact IDs, then use supplied GT only for post-prediction evaluation."""
    pairs, skipped, gt_summary = validation_pair_records(search_dir, reference_dir, external_gt)
    if sample_id:
        selected = normalize_sample_id(sample_id)
        pairs = [pair for pair in pairs if pair["sample_id"] == selected]
    search_label, reference_label = Path(search_dir).expanduser().name, Path(reference_dir).expanduser().name
    rows = []
    for completed, pair_record in enumerate(pairs, start=1):
        identifier = pair_record["sample_id"]
        search_record, reference_record = pair_record["search_record"], pair_record["reference_record"]
        search = cv2.imread(str(pair_record["search_root"] / search_record["search"]), cv2.IMREAD_GRAYSCALE)
        reference = cv2.imread(str(pair_record["reference_root"] / reference_record["reference"]), cv2.IMREAD_GRAYSCALE)
        if search is None or reference is None:
            skipped.append(identifier)
            if on_progress is not None:
                on_progress(completed, len(pairs))
            continue
        result = localize(search, reference)
        gt_record = pair_record["gt_record"]
        gt = (float(gt_record["center_x"]), float(gt_record["center_y"])) if gt_record is not None else None
        error_px = errors(result, gt)[2] if gt is not None else None
        metadata = {"ground_truth_center": {"x": gt[0], "y": gt[1]}} if gt is not None else {}
        noise_mode = (gt_record or {}).get("noise_mode") or search_record.get("noise_mode")
        rows.append({"Sample": identifier, "Search Noise": search_label, "Reference Noise": reference_label,
                     "Search Noise Level": search_record.get("noise_mode") or search_label,
                     "Reference Noise Level": reference_record.get("noise_mode") or reference_label,
                     "Noise Mode": noise_mode,
                     "Architecture": search_record.get("process_type", "unknown"),
                     "GT X": gt[0] if gt is not None else None, "GT Y": gt[1] if gt is not None else None,
                     "Pred X": result["center_x"], "Pred Y": result["center_y"], "error_px": error_px,
                     "confidence": result.get("confidence"), "Result": result_status(error_px),
                     "GT Rotation (deg)": (gt_record or {}).get("rotation_deg"),
                     **{f"Success@{tolerance}px": (error_px <= tolerance if error_px is not None else None)
                        for tolerance in (1, 2, 3, 4, 5, 10)},
                     "pair": {"search": search, "reference": reference,
                              "metadata": metadata},
                     "result": result})
        if on_progress is not None:
            on_progress(completed, len(pairs))
    if return_gt_summary:
        return rows, skipped, gt_summary
    return rows, skipped


def comparison_metrics(rows: list[dict]) -> dict:
    rows = [row for row in rows if row.get("error_px") is not None]
    if not rows:
        return {"Samples": 0, "Accuracy@2px": 0.0, "Accuracy@5px": 0.0,
                "Accuracy@10px": 0.0, "Mean Error": float("nan"), "Median Error": float("nan"),
                "Mean Confidence": float("nan")}
    values = np.asarray([row["error_px"] for row in rows], dtype=float)
    confidence = [row["confidence"] for row in rows if row["confidence"] is not None]
    return {"Samples": len(rows), "Accuracy@2px": float(np.mean(values <= 2)),
            "Accuracy@5px": float(np.mean(values <= 5)), "Accuracy@10px": float(np.mean(values <= 10)),
            "Mean Error": float(values.mean()), "Median Error": float(np.median(values)),
            "Mean Confidence": float(np.mean(confidence)) if confidence else float("nan")}


def display_dataset_sample(record: dict, row: dict) -> None:
    root = Path(record["dataset_dir"])
    search = cv2.imread(str(root / row["search_path"]), cv2.IMREAD_GRAYSCALE)
    reference = cv2.imread(str(root / row["reference_path"]), cv2.IMREAD_GRAYSCALE)
    if search is None or reference is None:
        st.warning("The image paths in this evaluator record could not be loaded.")
        return
    result = {"center_x": float(row["pred_x"]), "center_y": float(row["pred_y"]),
              "confidence": float(row["confidence"]), "scale": float(row["scale"]),
              "rotation": float(row["rotation"])}
    pair = {"search": search, "reference": reference,
            "metadata": {"ground_truth_center": {"x": float(row["gt_x"]), "y": float(row["gt_y"])} }}
    render_pair(pair, result, float(row["error_px"]))
    show_result(result, ground_truth(pair["metadata"]))
    st.caption(f"Noise mode: {row.get('noise_mode') or 'N/A'}")


def display_noise_comparison(rows: list[dict], skipped: list[str]) -> None:
    if skipped:
        st.warning(f"Skipped {len(skipped)} unmatched or unreadable sample ID(s): {', '.join(skipped[:10])}")
    if not rows:
        st.info("No matching sample IDs were available for this noise comparison.")
        return
    validation_summary(rows)
    st.dataframe([{key: value for key, value in row.items() if key not in {"pair", "result"}}
                  for row in rows], use_container_width=True)
    selected = st.selectbox("Inspect matched noise pair", range(len(rows)),
                            format_func=lambda index: str(rows[index]["Sample"]), key="noise_pair_viewer")
    row = rows[selected]
    render_pair(row["pair"], row["result"], row["error_px"])
    gt_center = row["pair"].get("metadata", {}).get("ground_truth_center")
    gt = (float(gt_center["x"]), float(gt_center["y"])) if gt_center else None
    show_result(row["result"], gt)


def initialize_state() -> None:
    defaults = {"generated_pairs": [], "generated_validation_results": [], "uploaded_search": None,
                "uploaded_reference": None, "manual_ground_truth": None, "manual_prediction_result": None,
                "dataset_validation_results": None, "output_directory": "generated_output",
                "postprocess_preview": None, "filtered_generated_pairs": [],
                "noise_comparison_results": None, "noise_comparison_matrix": None,
                "selected_search_noise": "clean", "selected_reference_noise": "clean",
                "validation_runs": [], "current_evaluation_graph": None,
                "search_variants": [{"label": "Search 1", "directory": "data/finfet"}],
                "reference_variants": [{"label": "Reference 1", "directory": "data/finfet"}],
                "multi_evaluation_results": None}
    defaults.update(POSTPROCESS_DEFAULTS)
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def generation_tab() -> None:
    st.subheader("Generate Pairs")
    architecture = st.selectbox("Architecture", ("FinFET", "DRAM"), key="generation_architecture")
    st.text_input("Output Directory", key="output_directory")
    seed, sample_id, sample_count = st.columns(3)
    with seed:
        generation_seed = int(st.number_input("Seed", min_value=0, value=42, step=1))
    with sample_id:
        pair_id = int(st.number_input("Sample ID", min_value=0, value=0, step=1))
    with sample_count:
        number_of_samples = int(st.number_input("Number of Samples", min_value=0, value=1, step=1))
    if architecture == "FinFET":
        noise_mode = st.selectbox("Noise Level", NOISE_MODES, format_func=str.title)
    else:
        noise_mode = "procedural"
        st.info("DRAM uses the existing procedural/random workflow; it has no FinFET noise presets.")
    one, all_modes = st.columns(2)
    if one.button("Generate Pair", type="primary"):
        if number_of_samples < 1:
            st.error("Please select at least 1 sample before generating.")
        else:
            with st.spinner("Generating pair(s)..."):
                pairs = []
                progress = st.progress(0, text=f"Generating 0 / {number_of_samples} pair(s)")
                for offset in range(number_of_samples):
                    pairs.append(generate_one(architecture, generation_seed + offset, noise_mode, pair_id + offset))
                    completed = offset + 1
                    progress.progress(completed / number_of_samples,
                                      text=f"Generating {completed} / {number_of_samples} pair(s) · {number_of_samples - completed} remaining")
            st.session_state.generated_pairs = pairs
            st.session_state.generated_validation_results = []
            st.session_state.postprocess_preview = None
            for pair in pairs:
                saved_to = save_pair(pair)
                if pair["noise_mode"] == "high":
                    metrics, warning = high_noise_quality(pair["search"])
                    if warning:
                        st.warning(warning + f" Metrics: {metrics}")
            st.success(f"Saved {len(pairs)} generated input pair(s) to {saved_to}")
    if all_modes.button("Generate All Noise Levels", disabled=architecture != "FinFET"):
        if number_of_samples < 1:
            st.error("Please select at least 1 sample before generating.")
        else:
            with st.spinner("Generating Clean / Low / Medium / High..."):
                pairs = []
                total_variants = number_of_samples * len(NOISE_MODES)
                progress = st.progress(0, text=f"Generating 0 / {total_variants} noise variants")
                for offset in range(number_of_samples):
                    def update_variant(completed, _total, mode):
                        overall = offset * len(NOISE_MODES) + completed
                        progress.progress(overall / total_variants,
                                          text=f"Generating {overall} / {total_variants}: {mode.title()} · {total_variants - overall} remaining")
                    variants = generate_all_noise(generation_seed + offset, pair_id + offset, update_variant)
                    pairs.extend(variants)
            st.session_state.generated_pairs = pairs
            st.session_state.generated_validation_results = []
            st.session_state.postprocess_preview = None
            for pair in pairs:
                save_pair(pair)
                if pair["noise_mode"] == "high":
                    metrics, warning = high_noise_quality(pair["search"])
                    if warning:
                        st.warning(warning + f" Metrics: {metrics}")
            st.success(f"Saved {len(pairs)} generated variants to {output_path()}")

    if st.session_state.generated_pairs:
        st.divider()
        st.subheader("Generated Pair Preview")
        for pair in st.session_state.generated_pairs:
            st.markdown(f"#### {pair['noise_mode'].upper()}")
            render_pair(pair)
        postprocess_section(st.session_state.generated_pairs)
        preview_record = st.session_state.postprocess_preview
        if preview_record:
            with st.expander("Processed Preview", expanded=True):
                render_filter_preview(st.session_state.generated_pairs[preview_record["index"]], preview_record["pair"])
        if st.button("Validate Generated Pairs"):
            rows = []
            progress = st.progress(0)
            for index, pair in enumerate(st.session_state.generated_pairs, start=1):
                result = localize(pair["search"], pair["reference"])
                gt = ground_truth(pair["metadata"])
                _, _, error_px = errors(result, gt)
                save_generated_prediction(pair, result, error_px)
                rows.append({"Sample": pair["sample_id"], "Architecture": pair["architecture"],
                             "Noise": pair["noise_mode"], "GT X": gt[0], "GT Y": gt[1],
                             "Pred X": result["center_x"], "Pred Y": result["center_y"],
                             "error_px": error_px, "confidence": result.get("confidence"),
                             "Result": result_status(error_px), "pair": pair, "result": result})
                progress.progress(index / len(st.session_state.generated_pairs), text=f"Validating {index} / {len(st.session_state.generated_pairs)}")
            st.session_state.generated_validation_results = rows
    if st.session_state.generated_validation_results:
        st.divider()
        st.subheader("Generated Validation Results")
        rows = st.session_state.generated_validation_results
        validation_summary(rows)
        st.dataframe([{key: value for key, value in row.items() if key not in {"pair", "result"}}
                      for row in rows], use_container_width=True)
        for row in rows:
            with st.expander(f"{row['Noise'].upper()} — {row['Result']} ({row['error_px']:.2f} px)"):
                render_pair(row["pair"], row["result"], row["error_px"])
                show_result(row["result"], ground_truth(row["pair"]["metadata"]))


def prediction_tab() -> None:
    st.subheader("Prediction Test")
    st.caption("Uploaded arrays are passed directly to ClassicalSEMLocalizer().localize(search, reference).")
    left, right = st.columns(2)
    with left:
        search_upload = st.file_uploader("Upload Search Image", type=IMAGE_TYPES, key="search_upload")
    with right:
        reference_upload = st.file_uploader("Upload Reference Image", type=IMAGE_TYPES, key="reference_upload")
    use_gt = st.checkbox("I know the ground-truth center")
    gt = None
    if use_gt:
        gx, gy = st.columns(2)
        with gx:
            ground_x = float(st.number_input("Ground Truth X", value=0.0, format="%.3f"))
        with gy:
            ground_y = float(st.number_input("Ground Truth Y", value=0.0, format="%.3f"))
        gt = (ground_x, ground_y)
    st.session_state.manual_ground_truth = gt
    if st.button("Run Prediction", type="primary"):
        try:
            progress = st.progress(10, text="Loading images")
            search = decode_upload(search_upload)
            reference = decode_upload(reference_upload)
            if search is None or reference is None:
                raise ValueError("Upload both a Search image and a Reference image.")
            progress.progress(30, text="Preparing Search/Reference")
            progress.progress(60, text="Running localization")
            with st.status("Running localization...", expanded=False) as status:
                result = localize(search, reference)
                status.update(label="Creating overlay...", state="running")
            progress.progress(85, text="Creating overlay")
            error_px = errors(result, gt)[2] if gt is not None else None
            overlay = render_localization_result(search, gt, result, result.get("confidence"), error_px,
                                                  reference_size=FOOTPRINT_PX)
            st.session_state.uploaded_search = search
            st.session_state.uploaded_reference = reference
            st.session_state.manual_prediction_result = {"search": search, "reference": reference,
                                                         "ground_truth": gt, "result": result,
                                                         "error_px": error_px, "overlay": overlay}
            progress.progress(100, text="Complete")
            status.update(label="Prediction complete", state="complete")
        except Exception as exc:
            st.error(str(exc))
    record = st.session_state.manual_prediction_result
    if record:
        left, right = st.columns(2)
        with left:
            st.caption("Reference")
            st.image(record["reference"], clamp=True, use_container_width=True)
        with right:
            st.caption("Search — green GT when supplied, red prediction")
            st.image(cv2.cvtColor(record["overlay"], cv2.COLOR_BGR2RGB), use_container_width=True)
        show_result(record["result"], record["ground_truth"])
        if st.button("Save Prediction Result"):
            folder = save_manual_prediction()
            st.success(f"Saved manual prediction inputs, overlay, and JSON to {folder}")


def evaluation_directory() -> Path:
    directory = output_path() / "evaluation"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_figure(figure, filename: str) -> Path:
    path = evaluation_directory() / filename
    figure.savefig(path, dpi=160, bbox_inches="tight")
    return path


def pr_curve_values(rows: list[dict], tolerance: float) -> tuple[np.ndarray, np.ndarray, float | None, str | None]:
    rows = scored_validation_rows(rows)
    confidence = np.asarray([float(row["confidence"]) for row in rows if row.get("confidence") is not None], dtype=float)
    correct = np.asarray([float(row["error_px"]) <= tolerance for row in rows if row.get("confidence") is not None], dtype=bool)
    if confidence.size == 0 or not np.isfinite(confidence).all() or np.ptp(confidence) < 1e-9:
        return np.array([]), np.array([]), None, "Prediction confidence has insufficient variation for a meaningful precision-recall curve."
    thresholds = np.r_[confidence.max() + np.finfo(float).eps, np.unique(confidence)[::-1], confidence.min() - np.finfo(float).eps]
    positives = int(correct.sum())
    precision, recall = [], []
    for threshold in thresholds:
        predicted_positive = confidence >= threshold
        tp = int(np.sum(predicted_positive & correct))
        fp = int(np.sum(predicted_positive & ~correct))
        fn = int(np.sum(~predicted_positive & correct))
        precision.append(tp / (tp + fp) if tp + fp else 1.0)
        recall.append(tp / (tp + fn) if positives else 0.0)
    recall_array, precision_array = np.asarray(recall), np.asarray(precision)
    order = np.argsort(recall_array)
    auc = float(np.trapezoid(precision_array[order], recall_array[order])) if positives else None
    return recall_array, precision_array, auc, None


def metadata_rows(rows: list[dict]) -> list[dict]:
    columns = ("Sample", "Architecture", "Noise Mode", "Search Noise", "Reference Noise", "GT X", "GT Y",
               "Pred X", "Pred Y", "error_px", "confidence", "Result")
    return [{key: row.get(key) for key in columns if key in row} for row in rows]


def csv_bytes(rows: list[dict]) -> str:
    if not rows:
        return ""
    fieldnames = list(rows[0])
    from io import StringIO
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def evaluation_analysis(rows: list[dict]) -> None:
    """Analysis-only outputs derived from already stored validation rows."""
    rows = scored_validation_rows(rows)
    if not rows:
        return
    with st.expander("Evaluation Analysis", expanded=False):
        st.caption("All plots below use the stored validation results; they do not rerun localization.")
        tolerance = st.select_slider("Analysis pixel tolerance", options=[1, 2, 3, 4, 5], value=5,
                                    key="analysis_tolerance")
        action = st.radio("Analysis output", ("Precision-Recall Curve", "Pixel Tolerance", "Confusion Matrix",
                                                "Stress Study", "Failure Case Analysis", "Metadata Comparison",
                                                "Official Scoring Utility"), horizontal=True, key="analysis_action")
        confidence_threshold = None
        if action == "Confusion Matrix":
            available_confidences = [float(row["confidence"]) for row in rows if row.get("confidence") is not None]
            if available_confidences:
                confidence_threshold = st.number_input("Confidence threshold", value=float(np.median(available_confidences)),
                                                       key="analysis_confidence_threshold")
        if action == "Precision-Recall Curve" and st.button("Generate PR Curve"):
            recall, precision, auc, warning = pr_curve_values(rows, tolerance)
            if warning:
                st.warning(warning)
            else:
                figure, axis = plt.subplots(figsize=(6, 4))
                axis.plot(recall, precision, marker="o")
                axis.set(xlabel="Recall", ylabel="Precision", title=f"PR curve: error ≤ {tolerance} px (n={len(rows)})", xlim=(0, 1), ylim=(0, 1.05))
                path = save_figure(figure, "pr_curve.png")
                st.pyplot(figure)
                plt.close(figure)
                st.caption(f"PR-AUC: {auc:.3f}. Positive = error ≤ {tolerance} px; predicted positive = confidence ≥ threshold. Saved: {path}")
        elif action == "Pixel Tolerance" and st.button("Generate Pixel-Tolerance Evaluation"):
            table = []
            for value in range(1, 6):
                success = sum(float(row["error_px"]) <= value for row in rows)
                table.append({"Tolerance (px)": value, "Success": success, "Failure": len(rows) - success,
                              "Accuracy": success / len(rows)})
            st.dataframe(table, use_container_width=True)
            figure, axis = plt.subplots(figsize=(6, 4))
            axis.plot([row["Tolerance (px)"] for row in table], [row["Accuracy"] for row in table], marker="o")
            axis.set(xlabel="Pixel tolerance", ylabel="Localization accuracy", title=f"Pixel-tolerance performance (n={len(rows)})", xticks=range(1, 6), ylim=(0, 1.05))
            path = save_figure(figure, "pixel_tolerance_accuracy.png")
            st.pyplot(figure)
            plt.close(figure)
            st.caption(f"Saved: {path}")
        elif action == "Confusion Matrix" and st.button("Generate Confusion Matrix"):
            confidences = [float(row["confidence"]) for row in rows if row.get("confidence") is not None]
            if not confidences:
                st.warning("Confidence is unavailable for these stored results.")
            else:
                correct = np.asarray([float(row["error_px"]) <= tolerance for row in rows])
                predicted = np.asarray([float(row.get("confidence") or 0) >= confidence_threshold for row in rows])
                tp, fp = int(np.sum(predicted & correct)), int(np.sum(predicted & ~correct))
                fn, tn = int(np.sum(~predicted & correct)), int(np.sum(~predicted & ~correct))
                matrix = np.array([[tp, fp], [fn, tn]])
                figure, axis = plt.subplots(figsize=(5, 4))
                image = axis.imshow(matrix, cmap="Blues")
                for (y, x), value in np.ndenumerate(matrix): axis.text(x, y, str(value), ha="center", va="center")
                axis.set(xticks=[0, 1], xticklabels=["Correct", "Incorrect"], yticks=[0, 1],
                         yticklabels=["Predicted positive", "Predicted negative"], xlabel="Actual localization", ylabel="Confidence decision",
                         title=f"Confidence confusion matrix (error ≤ {tolerance} px)")
                path = save_figure(figure, "confusion_matrix.png")
                st.pyplot(figure)
                plt.close(figure)
                st.caption(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}. Saved: {path}")
        elif action == "Stress Study" and st.button("Generate Stress Study"):
            groups = {}
            for row in rows:
                level = row.get("Noise Mode") or row.get("Search Noise") or "unknown"
                groups.setdefault(str(level), []).append(row)
            table = [{"Stress level": level, "Samples": len(group),
                      "Accuracy@2px": float(np.mean([float(item["error_px"]) <= 2 for item in group])),
                      "Accuracy@5px": float(np.mean([float(item["error_px"]) <= 5 for item in group])),
                      "Mean Error": float(np.mean([float(item["error_px"]) for item in group])),
                      "Median Error": float(np.median([float(item["error_px"]) for item in group])),
                      "Mean Confidence": float(np.mean([float(item["confidence"]) for item in group if item.get("confidence") is not None])) if any(item.get("confidence") is not None for item in group) else None}
                     for level, group in sorted(groups.items())]
            st.dataframe(table, use_container_width=True)
            figure, axis = plt.subplots(figsize=(6, 4))
            axis.plot([row["Stress level"] for row in table], [row["Accuracy@5px"] for row in table], marker="o")
            axis.set(xlabel="Metadata noise level", ylabel="Accuracy@5px", title="Noise stress study", ylim=(0, 1.05))
            path = save_figure(figure, "stress_study.png")
            st.pyplot(figure)
            plt.close(figure)
            st.caption(f"Uses stored `noise_mode` metadata only. Saved: {path}")
        elif action == "Failure Case Analysis":
            failures = sorted(rows, key=lambda row: float(row["error_px"]), reverse=True)
            low_confidence = min(rows, key=lambda row: float(row.get("confidence") or float("inf")))
            near_miss = min(rows, key=lambda row: abs(float(row["error_px"]) - tolerance))
            if st.button("Show Failure Case Analysis"):
                for label, row in (("Worst localization error", failures[0]), ("Lowest-confidence result", low_confidence), ("Representative near-miss", near_miss)):
                    with st.container(border=True):
                        st.markdown(f"**{label}: sample {row['Sample']}**")
                        render_pair(row["pair"], row["result"], row["error_px"])
                        show_result(row["result"], ground_truth(row["pair"]["metadata"]))
            if st.button("Save Failure Overlays"):
                folder = evaluation_directory() / "failures"
                folder.mkdir(parents=True, exist_ok=True)
                for row in failures[:30]:
                    overlay = render_localization_result(row["pair"]["search"], ground_truth(row["pair"]["metadata"]),
                                                          row["result"], row["result"].get("confidence"), row["error_px"], FOOTPRINT_PX)
                    cv2.imwrite(str(folder / f"failure_{row['Sample']}.png"), overlay)
                st.success(f"Saved up to 30 failure overlays to {folder}")
        elif action == "Metadata Comparison" and st.button("Generate Metadata Comparison"):
            table = metadata_rows(rows)
            text = csv_bytes(table)
            path = evaluation_directory() / "metadata_comparison.csv"
            path.write_text(text, encoding="utf-8")
            st.dataframe(table, use_container_width=True)
            st.download_button("Download metadata CSV", text, file_name="metadata_comparison.csv", mime="text/csv")
            st.caption(f"Saved: {path}")
        elif action == "Official Scoring Utility" and st.button("Run Official Scoring Utility"):
            st.info("Official scoring utility not found. Add the organizer-provided utility to enable standardized scoring output.")


def validation_history_summary(runs: list[dict]) -> list[dict]:
    summary = []
    for run in runs:
        rows = scored_validation_rows(run["rows"])
        if not rows:
            continue
        errors = np.asarray([float(row["error_px"]) for row in rows])
        summary.append({"Search Noise": run["search_noise"], "Reference Noise": run["reference_noise"],
                        "Combination": f"{run['search_noise']} / {run['reference_noise']}", "Samples": len(rows),
                        "Mean Error": float(errors.mean()), "Median Error": float(np.median(errors)),
                        **{f"Acc@{level}": float(np.mean(errors <= level)) for level in range(1, 6)}})
    return summary


def display_and_store_graph(figure, filename: str) -> None:
    path = save_figure(figure, filename)
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    st.session_state.current_evaluation_graph = {"name": filename, "image": buffer.getvalue(), "path": str(path)}
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)
    st.caption(f"Saved: {path}")


def evaluation_graphs(current_rows: list[dict], has_noise_metadata: bool = True) -> None:
    """Fast plots consuming persisted validation history; never invokes localization."""
    current_rows = scored_validation_rows(current_rows)
    if not current_rows:
        st.info("Graphs requiring GT are unavailable until at least one selected pair has matched ground truth.")
        return
    runs = st.session_state.validation_runs
    summary = validation_history_summary(runs)
    if not summary:
        return
    st.divider()
    st.subheader("Evaluation Graphs")
    tolerance, confidence_threshold = st.columns(2)
    with tolerance:
        pixel_tolerance = st.select_slider("Pixel tolerance", options=[1, 2, 3, 4, 5], value=5, key="graph_tolerance")
    with confidence_threshold:
        available = [float(row["confidence"]) for row in current_rows if row.get("confidence") is not None]
        threshold = st.number_input("Confidence threshold", value=float(np.median(available)) if available else 0.0,
                                    key="graph_confidence_threshold")
    st.dataframe(summary, use_container_width=True)
    (evaluation_directory() / "validation_summary.csv").write_text(csv_bytes(summary), encoding="utf-8")
    pr, error, accuracy, confusion, stress = st.columns(5)
    if not has_noise_metadata:
        st.info("Noise-specific plots are unavailable because the supplied ground-truth file has no noise_mode metadata.")
    if pr.button("Generate PR Curve"):
        with st.spinner("Generating graph..."):
            recall, precision, auc, warning = pr_curve_values(current_rows, pixel_tolerance)
            if warning:
                st.warning("Confidence scores have insufficient variation for a meaningful PR curve.")
            else:
                figure, axis = plt.subplots(figsize=(6, 4))
                axis.plot(recall, precision, marker="o")
                axis.set(xlabel="Recall", ylabel="Precision", title=f"PR curve: error ≤ {pixel_tolerance} px", xlim=(0, 1), ylim=(0, 1.05))
                display_and_store_graph(figure, "pr_curve.png")
                st.caption(f"PR-AUC: {auc:.3f}. Accepted = confidence ≥ threshold; correct = error ≤ {pixel_tolerance} px.")
    if error.button("Generate Pixel Error Graph", disabled=not has_noise_metadata):
        with st.spinner("Generating graph..."):
            labels = [row["Combination"] for row in summary]
            figure, axes = plt.subplots(1, 2, figsize=(12, 4))
            axes[0].plot(labels, [row["Mean Error"] for row in summary], marker="o", label="Mean")
            axes[0].plot(labels, [row["Median Error"] for row in summary], marker="s", label="Median")
            axes[0].set(xlabel="Search / Reference noise", ylabel="Pixel error", title="Pixel Error vs Noise")
            axes[0].tick_params(axis="x", rotation=35); axes[0].legend()
            accuracy_current = [float(np.mean([float(row["error_px"]) <= level for row in current_rows])) for level in range(1, 6)]
            axes[1].plot(range(1, 6), accuracy_current, marker="o")
            axes[1].set(xlabel="Allowed pixel error", ylabel="Accuracy", title="Accuracy vs Pixel Tolerance", xticks=range(1, 6), ylim=(0, 1.05))
            display_and_store_graph(figure, "pixel_error_vs_noise.png")
            figure2, axis2 = plt.subplots(figsize=(6, 4)); axis2.plot(range(1, 6), accuracy_current, marker="o")
            axis2.set(xlabel="Allowed pixel error", ylabel="Accuracy", title="Accuracy vs Pixel Tolerance", xticks=range(1, 6), ylim=(0, 1.05))
            save_figure(figure2, "accuracy_vs_pixel_tolerance.png"); plt.close(figure2)
    if accuracy.button("Generate Accuracy vs Noise Graph", disabled=not has_noise_metadata):
        with st.spinner("Generating graph..."):
            figure, axis = plt.subplots(figsize=(8, 4))
            labels = [row["Combination"] for row in summary]
            for level in range(1, 6): axis.plot(labels, [row[f"Acc@{level}"] for row in summary], marker="o", label=f"@{level}px")
            axis.set(xlabel="Search / Reference noise", ylabel="Accuracy", title="Accuracy vs Noise", ylim=(0, 1.05))
            axis.tick_params(axis="x", rotation=35); axis.legend(ncol=5)
            display_and_store_graph(figure, "accuracy_vs_noise.png")
    if confusion.button("Generate Confusion Matrix"):
        with st.spinner("Generating graph..."):
            correct = np.asarray([float(row["error_px"]) <= pixel_tolerance for row in current_rows])
            accepted = np.asarray([float(row.get("confidence") or 0) >= threshold for row in current_rows])
            matrix = np.array([[np.sum(accepted & correct), np.sum(accepted & ~correct)],
                               [np.sum(~accepted & correct), np.sum(~accepted & ~correct)]])
            figure, axis = plt.subplots(figsize=(5, 4)); axis.imshow(matrix, cmap="Blues")
            for (y, x), value in np.ndenumerate(matrix): axis.text(x, y, str(int(value)), ha="center", va="center")
            axis.set(xticks=[0, 1], xticklabels=["Correct", "Incorrect"], yticks=[0, 1],
                     yticklabels=["Accepted", "Rejected"], xlabel="Localization", ylabel="Confidence decision",
                     title=f"Confusion Matrix (≤ {pixel_tolerance}px)")
            display_and_store_graph(figure, "confusion_matrix.png")
    if stress.button("Generate Stress Study", disabled=not has_noise_metadata):
        with st.spinner("Generating graph..."):
            order = {"clean": 0, "low": 1, "medium": 2, "high": 3}
            ordered = sorted(summary, key=lambda row: (order.get(str(row["Search Noise"]).lower(), 99),))
            figure, axis = plt.subplots(figsize=(8, 4)); labels = [row["Combination"] for row in ordered]
            axis.plot(labels, [row["Acc@5"] for row in ordered], marker="o")
            axis.set(xlabel="Search / Reference noise", ylabel="Accuracy@5px", title="Baseline-to-Stress Study", ylim=(0, 1.05))
            axis.tick_params(axis="x", rotation=35)
            display_and_store_graph(figure, "stress_study.png")
    graph = st.session_state.current_evaluation_graph
    if graph:
        st.markdown(f"##### Current Graph — {graph['name']}")
        st.image(graph["image"], use_container_width=True)


def variant_editor(kind: str, title: str) -> list[dict]:
    key = f"{kind}_variants"
    values = st.session_state[key]
    st.markdown(f"##### {title}")
    for index, item in enumerate(values):
        label_column, directory_column, remove_column = st.columns((2, 5, 1))
        with label_column:
            st.text_input("Label", value=item["label"], key=f"multi_{kind}_label_{index}")
        with directory_column:
            st.text_input("Directory", value=item["directory"], key=f"multi_{kind}_directory_{index}")
        with remove_column:
            if st.button("Remove", key=f"remove_{kind}_{index}", disabled=len(values) == 1):
                st.session_state[key] = [entry for position, entry in enumerate(values) if position != index]
                st.rerun()
    if st.button(f"+ Add {kind.title()} Variant", key=f"add_{kind}"):
        st.session_state[key] = values + [{"label": f"{kind.title()} {len(values) + 1}", "directory": ""}]
        st.rerun()
    return [{"label": st.session_state[f"multi_{kind}_label_{index}"].strip(),
             "directory": st.session_state[f"multi_{kind}_directory_{index}"].strip()}
            for index in range(len(values))]


def multi_dataset_section(ground_truth_upload=None) -> None:
    st.subheader("Multi-Dataset Comparison")
    left, right = st.columns(2)
    with left:
        search_variants = variant_editor("search", "Search Variants")
    with right:
        reference_variants = variant_editor("reference", "Reference Variants")
    mode = st.radio("Comparison Mode", ("All Combinations", "Matched Labels Only"), horizontal=True,
                    key="multi_comparison_mode")
    force = st.checkbox("Force Re-evaluate completed combinations", key="multi_force")
    if st.button("Run Multi-Dataset Comparison", type="primary"):
        try:
            external_gt = None
            if ground_truth_upload is not None:
                external_gt = parse_external_ground_truth(ground_truth_upload.name, ground_truth_upload.getvalue())
            if any(not item["label"] or not item["directory"] for item in search_variants + reference_variants):
                raise ValueError("Every variant needs both a label and a directory.")
            st.session_state.search_variants, st.session_state.reference_variants = search_variants, reference_variants
            pairs, skipped_labels = build_combinations(search_variants, reference_variants,
                                                       "matched" if mode == "Matched Labels Only" else "all")
            if not pairs:
                raise ValueError("No variant combinations matched the selected comparison mode.")
            progress = st.progress(0, text=f"Combination 0 / {len(pairs)}")
            summaries, combination_results = [], []
            for combination_index, (search_variant, reference_variant) in enumerate(pairs, start=1):
                def update(sample_index, total_samples, _search, _reference):
                    fraction = ((combination_index - 1) + sample_index / max(total_samples, 1)) / len(pairs)
                    progress.progress(fraction, text=(f"Combination {combination_index} / {len(pairs)} — "
                                                       f"{search_variant['label']} × {reference_variant['label']} · "
                                                       f"Sample {sample_index} / {total_samples}"))
                rows, details = evaluate_variant_pair(search_variant, reference_variant, evaluation_directory(), force=force,
                                                      progress=update, external_ground_truth=external_gt)
                summary = {"Search": search_variant["label"], "Reference": reference_variant["label"],
                           **multi_metrics(rows), "Missing Search": len(details["missing_search"]),
                           "Missing Reference": len(details["missing_reference"]),
                           "Matched GT": details.get("matched_gt", len([row for row in rows if row.get("error_px") is not None])),
                           "Missing GT": details.get("missing_gt", 0), "Unused GT": details.get("unused_gt", 0),
                           "Result CSV": details["path"]}
                summaries.append(summary)
                combination_results.append({"search": search_variant, "reference": reference_variant,
                                            "rows": rows, "details": details})
            progress.progress(100, text=f"Multi-dataset comparison complete — {len(pairs)} / {len(pairs)} combinations")
            write_experiment(evaluation_directory(), search_variants, reference_variants, mode, summaries)
            st.session_state.multi_evaluation_results = {"summaries": summaries, "combinations": combination_results,
                                                         "skipped_labels": skipped_labels, "mode": mode,
                                                         "ground_truth_source": ground_truth_upload.name if ground_truth_upload else "generated dataset metadata"}
        except Exception as exc:
            st.error(str(exc))
    result = st.session_state.multi_evaluation_results
    if result:
        st.caption("Ground truth source: " + result.get("ground_truth_source", "generated dataset metadata"))
        if result["skipped_labels"]:
            st.warning("Skipped unmatched Search labels: " + ", ".join(result["skipped_labels"]))
        st.dataframe(result["summaries"], use_container_width=True)
        st.caption("Completed combinations are saved in evaluation/multi_dataset and are reused on the next run unless Force Re-evaluate is selected.")
        available = [item for item in result["combinations"] if item["rows"]]
        if available:
            selection = st.selectbox("Browse combination", range(len(available)),
                                     format_func=lambda index: f"{available[index]['search']['label']} × {available[index]['reference']['label']}",
                                     key="multi_combination_viewer")
            combination = available[selection]
            failures_only = st.checkbox("Show failures only", key="multi_failures_only")
            rows = sorted(combination["rows"], key=lambda row: float(row["error_px"]) if row.get("error_px") is not None else -1,
                          reverse=failures_only)
            sample = st.selectbox("Browse sample", range(len(rows)),
                                  format_func=lambda index: (f"{rows[index]['sample_id']} — {float(rows[index]['error_px']):.2f} px"
                                                              if rows[index].get("error_px") is not None
                                                              else f"{rows[index]['sample_id']} — Missing Ground Truth"),
                                  key="multi_sample_viewer")
            row = rows[sample]
            search = cv2.imread(str(Path(combination["search"]["directory"]) / row["search_path"]), cv2.IMREAD_GRAYSCALE)
            reference = cv2.imread(str(Path(combination["reference"]["directory"]) / row["reference_path"]), cv2.IMREAD_GRAYSCALE)
            if search is not None and reference is not None:
                prediction = {"center_x": float(row["pred_x"]), "center_y": float(row["pred_y"]),
                              "confidence": float(row["confidence"]) if row.get("confidence") not in (None, "") else None}
                metadata = ({"ground_truth_center": {"x": float(row["gt_x"]), "y": float(row["gt_y"])}}
                            if row.get("gt_x") is not None and row.get("gt_y") is not None else {})
                pair = {"search": search, "reference": reference, "metadata": metadata}
                st.caption(f"Search: {combination['search']['label']} · Reference: {combination['reference']['label']} · Sample: {row['sample_id']}")
                render_pair(pair, prediction, float(row["error_px"]) if row.get("error_px") is not None else None)
                gt = ground_truth(metadata) if metadata else None
                show_result(prediction, gt)


def ground_truth_file_controls():
    """Render the optional Dataset Validation GT input and in-memory templates."""
    upload = st.file_uploader("Ground Truth / Metadata File (optional)", type=["csv", "json"],
                              key="dataset_validation_ground_truth")
    with st.expander("Ground Truth File Format"):
        st.caption("center_x / center_y are the reference center in the Search image. Origin is top-left; +X is right and +Y is down.")
        csv_template = "sample_id,center_x,center_y\n000001,512.4,487.2\n000002,436.0,521.5\n"
        json_template = json.dumps({"samples": [{"sample_id": "000001", "center_x": 512.4, "center_y": 487.2}]}, indent=2)
        st.markdown("CSV example")
        st.code(csv_template, language="csv")
        st.markdown("JSON example")
        st.code(json_template, language="json")
        st.caption("Optional fields: reference_filename, search_filename, noise_mode, and rotation_deg. Exact normalized filenames are preferred over sample IDs.")
        csv_download, json_download = st.columns(2)
        with csv_download:
            st.download_button("Download CSV Template", csv_template, file_name="ground_truth_template.csv", mime="text/csv")
        with json_download:
            st.download_button("Download JSON Template", json_template, file_name="ground_truth_template.json", mime="application/json")
    return upload


def dataset_tab() -> None:
    st.subheader("Dataset Validation")
    validation_mode = st.radio("Validation Mode", ("Single Comparison", "Multi-Dataset Comparison"), horizontal=True,
                               key="dataset_validation_mode")
    ground_truth_upload = ground_truth_file_controls()
    if validation_mode == "Multi-Dataset Comparison":
        multi_dataset_section(ground_truth_upload)
        return
    st.caption("Generated datasets use their existing metadata automatically. With an uploaded GT file, plain image directories are also supported. Matching is deterministic by filename or normalized sample ID.")
    search_noise, reference_noise = st.columns(2)
    with search_noise:
        noise_level_directory = st.text_input("Noise Level Directory", value="data/finfet",
                                              key="noise_level_directory")
    with reference_noise:
        reference_directory = st.text_input("Reference Directory", value="data/finfet",
                                            key="reference_directory")
    selected_sample_id = st.text_input("Sample ID (optional)", key="noise_comparison_sample_id")
    if st.button("Validate Dataset", type="primary"):
        try:
            external_gt = None
            gt_source = "generated dataset metadata"
            if ground_truth_upload is not None:
                external_gt = parse_external_ground_truth(ground_truth_upload.name, ground_truth_upload.getvalue())
                gt_source = ground_truth_upload.name
            progress = st.progress(0, text="Preparing validation")
            def update_progress(completed, total):
                progress.progress(completed / total if total else 1.0,
                                  text=f"Validating {completed} / {total}")
            with st.spinner("Validating matching sample IDs from the selected directories..."):
                rows, skipped, gt_summary = noise_comparison_rows(
                    noise_level_directory, reference_directory, selected_sample_id, on_progress=update_progress,
                    external_gt=external_gt, return_gt_summary=True)
            progress.progress(100, text=f"Validation complete — {len(rows)} / {len(rows) + len(skipped)}")
            result = {
                "rows": rows, "skipped": skipped,
                "search_noise": rows[0].get("Search Noise Level", Path(noise_level_directory).name) if rows else Path(noise_level_directory).name,
                "reference_noise": rows[0].get("Reference Noise Level", Path(reference_directory).name) if rows else Path(reference_directory).name,
                "search_directory": str(Path(noise_level_directory).expanduser()),
                "reference_directory": str(Path(reference_directory).expanduser()),
                "ground_truth_source": gt_source,
                "gt_summary": gt_summary,
            }
            st.session_state.dataset_validation_results = result
            upload_key = (hashlib.sha256(ground_truth_upload.getvalue()).hexdigest()
                          if ground_truth_upload is not None else "automatic")
            identity = (result["search_directory"], result["reference_directory"], upload_key)
            previous = [run for run in st.session_state.validation_runs
                        if (run["search_directory"], run["reference_directory"], run.get("ground_truth_key", "automatic")) != identity]
            result["ground_truth_key"] = upload_key
            previous.append(result)
            st.session_state.validation_runs = previous
        except Exception as exc:
            st.error(str(exc))
    record = st.session_state.dataset_validation_results
    if record and "skipped" in record:
        st.divider()
        st.subheader(f"Validation: {record['search_noise']} Search / {record['reference_noise']} Reference")
        gt_summary = record.get("gt_summary", {})
        st.caption(f"Ground truth source: {record.get('ground_truth_source', 'generated dataset metadata')} · "
                   f"Matched GT: {gt_summary.get('matched_gt', len(scored_validation_rows(record['rows'])))} · "
                   f"Missing GT: {gt_summary.get('missing_gt', 0)} · "
                   f"Unused GT records: {gt_summary.get('unused_gt', 0)}")
        display_noise_comparison(record["rows"], record["skipped"])
        evaluation_graphs(record["rows"], gt_summary.get("has_noise_metadata", True))
        return

    # The active UI intentionally has one validation action. Legacy code below
    # remains only for compatibility with old saved Streamlit session objects.
    return

    common_parent = Path(noise_level_directory).expanduser().parent
    found_noise_roots = noise_dataset_roots(str(common_parent)) if common_parent == Path(reference_directory).expanduser().parent else {}
    if found_noise_roots:
        st.caption("Available noise dataset roots: " + ", ".join(
            f"{mode.title()} ({path})" for mode, path in found_noise_roots.items()))
    else:
        st.caption("The comparison matrix is available when both directories are sibling noise folders under one parent.")
    validate, compare, inspect, matrix = st.columns(4)
    if validate.button("Validate Dataset", type="primary"):
        try:
            problem = dataset_manifest_problem(noise_level_directory)
            if problem:
                raise ValueError(problem)
            with st.spinner("Running existing dataset evaluator..."):
                st.session_state.dataset_validation_results = run_evaluator(noise_level_directory)
        except Exception as exc:
            st.error(str(exc))
    if compare.button("Validate Noise Pair"):
        try:
            with st.spinner("Comparing the selected Noise Level Directory against the Reference Directory..."):
                rows, skipped = noise_comparison_rows(noise_level_directory, reference_directory)
            st.session_state.noise_comparison_results = {"rows": rows, "skipped": skipped,
                                                          "search_noise": Path(noise_level_directory).name,
                                                          "reference_noise": Path(reference_directory).name}
        except Exception as exc:
            st.error(str(exc))
    if inspect.button("Inspect Noise Pair"):
        try:
            if not selected_sample_id:
                raise ValueError("Enter a Sample ID to inspect one exact noise pair.")
            with st.spinner("Localizing the selected matching pair..."):
                rows, skipped = noise_comparison_rows(noise_level_directory, reference_directory, selected_sample_id)
            st.session_state.noise_comparison_results = {"rows": rows, "skipped": skipped,
                                                          "search_noise": Path(noise_level_directory).name,
                                                          "reference_noise": Path(reference_directory).name}
        except Exception as exc:
            st.error(str(exc))
    if matrix.button("Run Noise Comparison Matrix"):
        try:
            if not found_noise_roots:
                raise ValueError("No per-noise dataset folders with existing manifests were found.")
            combinations = [(search_mode, reference_mode) for search_mode in found_noise_roots for reference_mode in found_noise_roots]
            progress = st.progress(0)
            matrix_rows = []
            for index, (search_mode, reference_mode) in enumerate(combinations, start=1):
                rows, skipped = noise_comparison_rows(str(found_noise_roots[search_mode]),
                                                      str(found_noise_roots[reference_mode]))
                matrix_rows.append({"Search Noise": search_mode, "Reference Noise": reference_mode,
                                    **comparison_metrics(rows), "Missing/Skipped": len(skipped)})
                progress.progress(index / len(combinations), text=f"Comparing {search_mode}/{reference_mode} ({index}/{len(combinations)})")
            st.session_state.noise_comparison_matrix = matrix_rows
        except Exception as exc:
            st.error(str(exc))
    comparison = st.session_state.noise_comparison_results
    if comparison:
        st.divider()
        st.subheader(f"Noise Comparison: {comparison['search_noise'].title()} Search / {comparison['reference_noise'].title()} Reference")
        display_noise_comparison(comparison["rows"], comparison["skipped"])
    if st.session_state.noise_comparison_matrix:
        st.divider()
        st.subheader("Noise Comparison Matrix")
        st.dataframe(st.session_state.noise_comparison_matrix, use_container_width=True)
    record = st.session_state.dataset_validation_results
    if not record:
        return
    combined = record["metrics"].get("combined", {})
    if combined:
        display_names = {"samples": "Samples", "accuracy_at_2px": "Accuracy@2px",
                         "accuracy_at_5px": "Accuracy@5px", "accuracy_at_10px": "Accuracy@10px",
                         "accuracy_at_20px": "Accuracy@20px", "mean_error_px": "Mean Error",
                         "median_error_px": "Median Error", "p90_error_px": "P90 Error",
                         "p95_error_px": "P95 Error", "max_error_px": "Max Error",
                         "false_localization_rate": "False Localization Rate"}
        metric_columns = st.columns(3)
        for index, (key, label) in enumerate(display_names.items()):
            value = combined.get(key)
            if isinstance(value, float):
                shown = f"{value:.2%}" if "accuracy" in key or "rate" in key else f"{value:.2f} px"
            else:
                shown = "N/A" if value is None else str(value)
            metric_columns[index % 3].metric(label, shown)
    st.dataframe(record["rows"], use_container_width=True)
    if record["rows"]:
        selected = st.selectbox("Inspect sample", range(len(record["rows"])),
                                format_func=lambda i: str(record["rows"][i].get("sample_id", i)))
        display_dataset_sample(record, record["rows"][selected])
        failures = sorted((row for row in record["rows"] if float(row["error_px"]) > 20),
                          key=lambda row: float(row["error_px"]), reverse=True)
        st.subheader("Failure Viewer")
        if not failures:
            st.success("No samples exceeded 20 px error.")
        for row in failures[:10]:
            with st.expander(f"{row.get('sample_id')} — {float(row['error_px']):.2f} px"):
                display_dataset_sample(record, row)
    with st.expander("Evaluator console output"):
        st.code(record["console"] or "No console output")


def main() -> None:
    st.set_page_config(page_title="SEM Localization", layout="wide")
    initialize_state()
    st.title("SEM Localization Workbench")
    generate, prediction, dataset = st.tabs(("Generate Pairs", "Prediction Test", "Dataset Validation"))
    with generate:
        generation_tab()
    with prediction:
        prediction_tab()
    with dataset:
        dataset_tab()


if __name__ == "__main__":
    main()
