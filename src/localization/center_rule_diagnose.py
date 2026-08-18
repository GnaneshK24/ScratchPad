"""Audit whether FinFET labels follow the challenge centre tie rule.

This module is intentionally diagnostic-only.  It never supplies a ground
truth coordinate to the matcher and it never mutates annotations.  Candidate
equivalence is defined by the same fused-response peak tolerance used by the
classical localizer's centre tie-break.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .classical_matcher import ClassicalSEMLocalizer, _representations
from .center_rule import resolve_equivalent_peak
from .config import LocalizationConfig
from .coordinates import search_center_to_top_left
from .dataset import SEMLocalizationDataset


def _template(reference):
    if reference.shape == (1000, 1000):
        return cv2.resize(reference, (100, 100), interpolation=cv2.INTER_AREA)
    return reference


def equivalent_centers(peaks, search_center, tolerance):
    """Return response-map peaks equivalent to the best match and its centre choice."""
    return resolve_equivalent_peak(peaks, search_center, tolerance)


def main():
    parser = argparse.ArgumentParser(description='Audit FinFET ground truth against the official centre tie rule.')
    parser.add_argument('--dataset-dir', required=True)
    parser.add_argument('--output-dir', default='evaluation')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--candidate-limit', type=int, default=100)
    parser.add_argument('--tolerance', type=float,
                        help='Equivalent-score tolerance; defaults to the matcher configuration.')
    parser.add_argument('--position-tolerance', type=float, default=5.,
                        help='Maximum GT-to-centre-choice distance for a consistent label.')
    args = parser.parse_args()

    dataset = SEMLocalizationDataset(args.dataset_dir, LocalizationConfig())
    records = dataset.records[:args.limit] if args.limit else dataset.records
    if not records:
        raise SystemExit('No FinFET records found.')
    non_finfet = [record for record in records if str(record.get('process_type', 'FinFET')).lower() != 'finfet']
    if non_finfet:
        raise SystemExit('This diagnostic is FinFET-only; the selected dataset contains non-FinFET records.')

    matcher = ClassicalSEMLocalizer()
    tolerance = matcher.config.equivalent_score_tolerance if args.tolerance is None else args.tolerance
    rows = []
    for record in tqdm(records, desc='Auditing centre rule', unit='sample', dynamic_ncols=True):
        search = cv2.imread(str(dataset.root / record['search']), cv2.IMREAD_GRAYSCALE)
        reference = _template(cv2.imread(str(dataset.root / record['reference']), cv2.IMREAD_GRAYSCALE))
        if search is None or reference is None:
            raise FileNotFoundError(f"Could not read pair {record['sample_id']}")
        search_repr = _representations(search, matcher.config)
        reference_repr = _representations(reference, matcher.config)
        fused = matcher._fused(search_repr, reference_repr)
        peaks = matcher._peaks(fused, args.candidate_limit, matcher.config.nms_radius)
        search_center = (search.shape[1] / 2., search.shape[0] / 2.)
        equivalents, closest = equivalent_centers(peaks, search_center, tolerance)
        gt_x, gt_y = float(record['center_x']), float(record['center_y'])
        gt_top_left = search_center_to_top_left(gt_x, gt_y)
        tx, ty = int(round(gt_top_left[0])), int(round(gt_top_left[1]))
        if not (0 <= tx < fused.shape[1] and 0 <= ty < fused.shape[0]):
            raise ValueError(f"Ground truth for sample {record['sample_id']} is outside the valid template domain.")
        distance = float(np.hypot(gt_x - closest['x'], gt_y - closest['y']))
        rows.append({
            'sample_id': record['sample_id'],
            'stored_gt_x': gt_x,
            'stored_gt_y': gt_y,
            'search_center_x': search_center[0],
            'search_center_y': search_center[1],
            'number_equivalent_matches': len(equivalents),
            'equivalent_candidate_centers': json.dumps(equivalents),
            'closest_equivalent_x': closest['x'],
            'closest_equivalent_y': closest['y'],
            'distance_gt_to_closest_equivalent': distance,
            'gt_follows_center_rule': distance <= args.position_tolerance,
            'best_match_score': float(peaks[0][2]),
            'gt_match_score': float(fused[ty, tx]),
        })

    ambiguous = [row for row in rows if row['number_equivalent_matches'] > 1]
    consistent = [row for row in rows if row['gt_follows_center_rule']]
    summary = {
        'total_samples': len(rows),
        'ambiguous_samples': len(ambiguous),
        'center_rule_consistent_samples': len(consistent),
        'center_rule_inconsistent_samples': len(rows) - len(consistent),
        'center_rule_consistency_percentage': 100. * len(consistent) / len(rows),
        'equivalence_score_tolerance': tolerance,
        'position_tolerance_px': args.position_tolerance,
        'contract_verdict': ('CONSISTENT' if len(consistent) == len(rows)
                             else 'INCONSISTENT_STORED_GT_DO_NOT_MEASURE_OFFICIAL_CENTRE_RULE'),
    }
    output = Path(args.output_dir) / 'diagnostics'
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'center_rule_consistency.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / 'center_rule_consistency_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
