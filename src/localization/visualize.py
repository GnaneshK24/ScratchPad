"""Shared, geometry-correct localization visualizations."""
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .coordinates import center_to_bbox

GT_COLOR = (0, 255, 0)          # BGR green
PREDICTION_COLOR = (0, 0, 255)  # BGR red


def _load_bgr(image):
    if isinstance(image, (str, Path)):
        value = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if value is None:
            raise ValueError(f"Could not load image: {image}")
        return value
    value = np.asarray(image)
    if value.ndim == 2:
        return cv2.cvtColor(value.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if value.ndim == 3 and value.shape[2] == 3:
        return value.astype(np.uint8).copy()
    raise ValueError("Image must be a path, grayscale array, or BGR array")


def _clamped_bbox(bbox, width, height):
    x0, y0, x1, y1 = (int(round(float(value))) for value in bbox)
    return (max(0, min(width - 1, x0)), max(0, min(height - 1, y0)),
            max(0, min(width - 1, x1)), max(0, min(height - 1, y1)))


def render_localization_result(search_image, ground_truth=None, prediction=None,
                               confidence=None, error=None, reference_size=100):
    """Return the shared BGR Search-image overlay.

    Ground truth and prediction are ``(x, y)`` pairs or mappings with
    ``center_x``, ``center_y``, and an optional ``bbox``.  The default box is
    the actual geometric 100x100 reference region, never an intensity centroid.
    """
    canvas = _load_bgr(search_image)
    height, width = canvas.shape[:2]

    def unpack(value):
        if isinstance(value, dict):
            x, y = float(value['center_x']), float(value['center_y'])
            bbox = value.get('bbox') or center_to_bbox(x, y, reference_size)
        else:
            x, y = float(value[0]), float(value[1])
            bbox = center_to_bbox(x, y, reference_size)
        return x, y, _clamped_bbox(bbox, width, height)

    if prediction is None:
        raise ValueError('prediction is required')
    px, py, pred_box = unpack(prediction)
    if ground_truth is not None:
        gx, gy, gt_box = unpack(ground_truth)
    else:
        gx = gy = None
        gt_box = None
    if error is None and ground_truth is not None:
        error = float(np.hypot(px - gx, py - gy))

    if gt_box is not None:
        cv2.rectangle(canvas, gt_box[:2], gt_box[2:], GT_COLOR, 2)
    cv2.rectangle(canvas, pred_box[:2], pred_box[2:], PREDICTION_COLOR, 2)
    if gt_box is not None:
        cv2.drawMarker(canvas, (int(round(gx)), int(round(gy))), GT_COLOR,
                       markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2, line_type=cv2.LINE_AA)
    cv2.drawMarker(canvas, (int(round(px)), int(round(py))), PREDICTION_COLOR,
                   markerType=cv2.MARKER_CROSS, markerSize=14, thickness=2, line_type=cv2.LINE_AA)
    if gt_box is not None:
        cv2.putText(canvas, "Ground Truth", (gt_box[0], max(24, gt_box[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, GT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(canvas, "Prediction", (pred_box[0], max(50, pred_box[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, PREDICTION_COLOR, 2, cv2.LINE_AA)

    details = [f"Pred: ({px:.0f}, {py:.0f})"]
    if gt_box is not None:
        details.insert(0, f"GT: ({gx:.0f}, {gy:.0f})")
    if error is not None:
        details.append(f"Error: {error:.1f} px")
    if confidence is not None:
        details.append(f"Confidence: {float(confidence):.2f}")
    cv2.rectangle(canvas, (8, 8), (300, 28 * len(details) + 12), (0, 0, 0), -1)
    for row, text in enumerate(details):
        cv2.putText(canvas, text, (16, 31 + 28 * row), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def save_localization_result(search_image, output_path, ground_truth, prediction,
                             confidence=None, error=None, reference_size=100):
    """Save the shared green-GT/red-prediction Search-image visualization."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image = render_localization_result(search_image, ground_truth, prediction,
                                       confidence, error, reference_size)
    if not cv2.imwrite(str(output_path), image):
        raise IOError(f"Could not save visualization: {output_path}")
    return image


def visualize_localization_result(search_image, reference_image, ground_truth, prediction,
                                  confidence=None, error=None, reference_size=100):
    """Display the exact same reusable Search overlay beside its reference."""
    overlay = render_localization_result(search_image, ground_truth, prediction,
                                         confidence, error, reference_size)
    reference = _load_bgr(reference_image)
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    axes[0].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Search image: green = GT, red = Prediction")
    axes[1].imshow(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Reference image")
    for axis in axes:
        axis.axis("off")
    fig.tight_layout()
    plt.show()
    return overlay


def save_prediction(search, result, output_path, ground_truth=None, heatmap=None):
    """Backward-compatible evaluation entry point; delegates to the one renderer."""
    if ground_truth is None:
        ground_truth = (result['center_x'], result['center_y'])
    return save_localization_result(search, output_path, ground_truth, result,
                                    result.get('confidence'))
