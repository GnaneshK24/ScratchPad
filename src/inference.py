"""Command-line entry point for the classical SEM localizer."""
import argparse
import json
from pathlib import Path
import cv2
from localization.classical_matcher import ClassicalSEMLocalizer

def localize_reference_pattern(reference_path, search_path):
    reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(str(search_path), cv2.IMREAD_GRAYSCALE)
    result = ClassicalSEMLocalizer().localize(search, reference)
    return {**result, 'predicted_x': result['center_x'], 'predicted_y': result['center_y'],
            'reference_path': str(reference_path), 'search_path': str(search_path)}

def main():
    parser = argparse.ArgumentParser(description='Locate a 100x100 SEM reference in a 1000x1000 search image.')
    parser.add_argument('--reference_path', required=True); parser.add_argument('--search_path', required=True)
    parser.add_argument('--output', help='Optional JSON result path')
    args = parser.parse_args(); result = localize_reference_pattern(args.reference_path, args.search_path)
    print(json.dumps(result, indent=2))
    if args.output: Path(args.output).write_text(json.dumps(result, indent=2))
if __name__ == '__main__': main()
