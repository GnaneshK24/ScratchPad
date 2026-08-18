"""Evaluate the classical SEM localizer on every dataset found under a root."""
import argparse
import json
from pathlib import Path

from tqdm import tqdm

from .config import LocalizationConfig
from .dataset import SEMLocalizationDataset
from .evaluate import main as evaluate_main


def _is_dataset_root(path: Path) -> bool:
    return (
        path.is_dir()
        and (
            (path / "ground_truth.csv").exists()
            or (path / "annotations.json").exists()
        )
        and (path / "reference").is_dir()
        and (path / "search").is_dir()
    )


def _discover_datasets(root: Path):
    datasets = []
    if _is_dataset_root(root):
        datasets.append(root)
    for child in sorted(root.iterdir()):
        if _is_dataset_root(child):
            datasets.append(child)
    return datasets


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all recognized SEM datasets under a folder."
    )
    parser.add_argument(
        "--data-root",
        default="data",
        help="Folder to scan for datasets (default: data)",
    )
    parser.add_argument(
        "--output-root",
        default="evaluation_all",
        help="Folder where per-dataset outputs will be written",
    )
    parser.add_argument(
        "--visualize-count",
        type=int,
        default=20,
        help="How many worst samples to visualize per dataset",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    datasets = _discover_datasets(data_root)
    if not datasets:
        raise FileNotFoundError(
            f"No datasets found under {data_root}. "
            "Expected a folder with ground_truth.csv or annotations.json."
        )

    print(f"Found {len(datasets)} dataset(s) under {data_root}")
    summary = []

    for dataset_dir in tqdm(datasets, desc="Evaluating datasets", unit="dataset", dynamic_ncols=True):
        ds = SEMLocalizationDataset(dataset_dir, LocalizationConfig())
        out_dir = output_root / dataset_dir.name
        print(f"Evaluating {dataset_dir} ({len(ds)} sample(s))")

        # Reuse the existing evaluator so behavior stays consistent.
        import sys

        old_argv = sys.argv
        try:
            sys.argv = [
                "evaluate.py",
                "--dataset-dir",
                str(dataset_dir),
                "--output-dir",
                str(out_dir),
                "--visualize-count",
                str(args.visualize_count),
            ]
            evaluate_main()
        finally:
            sys.argv = old_argv

        metrics_path = out_dir / "metrics.json"
        combined = json.loads(metrics_path.read_text())["combined"]
        summary.append(
            {
                "dataset": dataset_dir.name,
                "path": str(dataset_dir),
                "samples": combined["samples"],
                "accuracy_at_5px": combined["accuracy_at_5px"],
                "accuracy_at_10px": combined["accuracy_at_10px"],
                "mean_error_px": combined["mean_error_px"],
                "median_error_px": combined["median_error_px"],
                "p90_error_px": combined["p90_error_px"],
                "p95_error_px": combined["p95_error_px"],
                "max_error_px": combined["max_error_px"],
                "false_localization_rate": combined["false_localization_rate"],
            }
        )

    (output_root / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
