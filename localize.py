"""Public command-line entry point for the production SEM localizer."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from src.localization.inference import localize as production_localize


def _load_grayscale(path: str, kind: str):
    image_path = Path(path).expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"{kind} image does not exist: {image_path}")
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"{kind} image could not be decoded: {image_path}")
    return image


def predict(reference_path: str, search_path: str) -> dict:
    """Load inputs and call the one production matcher in its required order."""
    reference = _load_grayscale(reference_path, "Reference")
    search = _load_grayscale(search_path, "Search")
    # The matcher API is (search, reference).  Keeping this explicit prevents
    # the public --reference/--search interface from being accidentally swapped.
    return production_localize(search, reference)


def main() -> None:
    parser = argparse.ArgumentParser(description="Locate a FinFET reference pattern in a SEM search image.")
    parser.add_argument("--reference", required=True, help="Path to the reference image.")
    parser.add_argument("--search", required=True, help="Path to the search image.")
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of the default x,y.")
    args = parser.parse_args()
    try:
        result = predict(args.reference, args.search)
    except Exception as exc:
        print(f"localize.py: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc

    if args.json:
        print(json.dumps({"x": float(result["center_x"]), "y": float(result["center_y"]),
                          "confidence": float(result["confidence"])}, separators=(",", ":")))
    else:
        # Keep evaluator-facing stdout intentionally machine-readable.
        print(f"{float(result['center_x']):.6f},{float(result['center_y']):.6f}")


if __name__ == "__main__":
    main()
