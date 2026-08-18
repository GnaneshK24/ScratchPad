"""Generate center-rule-compliant FinFET localization pairs.

This is deliberately a thin public entry point.  It delegates all image
creation and ground-truth construction to the established FinFET generator.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC_ROOT = ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from dataset_generator import FinFETSEMDatasetGenerator  # noqa: E402


def _configure_console_encoding() -> None:
    """Allow the established generator's progress messages on Windows consoles."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
        except (AttributeError, OSError):
            # Some redirected streams do not support reconfiguration; generation
            # itself remains independent of presentation encoding.
            pass


def main() -> None:
    _configure_console_encoding()
    parser = argparse.ArgumentParser(
        description="Generate reproducible FinFET reference/search localization pairs."
    )
    parser.add_argument("--architecture", type=str.lower, required=True, choices=("finfet",),
                        help="Architecture style for this release (finfet only).")
    parser.add_argument("--num-pairs", type=int, required=True,
                        help="Number of reference/search pairs to generate (at least 1).")
    parser.add_argument("--output-dir", required=True,
                        help="Directory that will receive reference/, search/, and ground_truth.csv.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional NumPy seed for reproducible generation.")
    parser.add_argument("--noise-mode", default="clean",
                        choices=("clean", "low", "medium", "high", "random", "standard"),
                        help="Existing FinFET acquisition preset (default: clean).")
    parser.add_argument("--base-images-dir", default=str(ROOT / "finfet_base_images"),
                        help="Directory containing the included FinFET base-layout images.")
    args = parser.parse_args()

    if args.num_pairs < 1:
        parser.error("Number of pairs must be at least 1.")
    # generate_dataset() invokes generate_image_pair(), whose final-image
    # centre-rule resolution is the single source of truth for FinFET GT.
    generator = FinFETSEMDatasetGenerator(
        input_dir=args.base_images_dir,
        seed=args.seed,
        noise_mode=args.noise_mode,
    )
    generator.generate_dataset(output_count=args.num_pairs, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
