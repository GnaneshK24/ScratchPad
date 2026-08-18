"""Explain each previously unclassified candidate-lifecycle failure.

This is diagnostic-only.  It replays the frozen pre-Top-30 lifecycle and
records nearest GT-basin candidates before and after every destructive stage.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .classical_matcher import ClassicalMatcherConfig, ClassicalSEMLocalizer
from .config import LocalizationConfig
from .dataset import SEMLocalizationDataset
from .trace_lifecycle import _center, _nearest, _trace_pair


def _stage(label, candidates, gt, *, source=None):
    nearest = _nearest(candidates, gt)
    candidate = nearest['candidate']
    result = {
        f'{label}_rank': nearest['rank'],
        f'{label}_distance_px': nearest['distance_px'],
        f'{label}_available': bool(nearest['distance_px'] is not None and nearest['distance_px'] <= 5.),
        f'{label}_source': None if candidate is None else candidate.get('candidate_source', source),
        f'{label}_x': None if candidate is None else _center(candidate)[0],
        f'{label}_y': None if candidate is None else _center(candidate)[1],
    }
    return result


def _classify(trace, gt):
    fused = _nearest(trace['raw_fused'], gt)
    coarse = _nearest(trace['candidate_union_raw'], gt)
    nms = _nearest(trace['rank_after_nms'], gt)
    cheap = _nearest(trace['cheap_all'], gt)
    shortlist = _nearest(trace['shortlist'], gt)
    attempts = trace['verification_attempts']
    input_candidates = [attempt['input'] for attempt in attempts]
    output_candidates = [attempt['output'] for attempt in attempts if attempt['output'] is not None]
    if shortlist['distance_px'] is not None and shortlist['distance_px'] <= 5.:
        verified = _nearest(output_candidates, gt)
        if verified['distance_px'] is None:
            return 'G7. GT shortlist candidate produced no verification output'
        if verified['distance_px'] > 5.:
            return 'G2. rotation/scale verification drifts GT candidate from its basin'
    if nms['distance_px'] is not None and nms['distance_px'] <= 10. and (cheap['distance_px'] is None or cheap['distance_px'] > 5.):
        return 'G3. cheap full-resolution rerank loses the GT candidate basin'
    if fused['distance_px'] is not None and fused['distance_px'] <= 5. and (coarse['distance_px'] is None or coarse['distance_px'] > 10.):
        return 'G4. GT peak absent from coarse representation candidate union'
    if input_candidates and output_candidates:
        return 'G7. candidate identity cannot be preserved through verification'
    return 'G7. no GT-basin candidate reaches local verification'


def main():
    parser = argparse.ArgumentParser(description='Explain the old G/other failure category without changing inference.')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--lifecycle', required=True)
    parser.add_argument('--output-dir', default='evaluation/diagnostics')
    args = parser.parse_args()
    with Path(args.lifecycle).open(newline='', encoding='utf-8') as file:
        sample_ids = {
            row['sample_id'] for row in csv.DictReader(file)
            if row['role'] == 'gt_nearest' and row['failure_category'] == 'G. other'
        }
    dataset = SEMLocalizationDataset(args.dataset, LocalizationConfig())
    records = [record for record in dataset.records if record['sample_id'] in sample_ids]
    # The original 26-failure taxonomy used the Top-25 matcher.  Reproduce it
    # exactly; this script must not be affected by the later Top-30 selection.
    matcher = ClassicalSEMLocalizer(ClassicalMatcherConfig(cheap_rerank_top_k=25, verify_top_k=25))
    rows = []
    categories = Counter()
    for record in tqdm(records, desc='Explaining other failures', unit='sample', dynamic_ncols=True):
        search = cv2.imread(str(dataset.root / record['search']), cv2.IMREAD_GRAYSCALE)
        reference = cv2.imread(str(dataset.root / record['reference']), cv2.IMREAD_GRAYSCALE)
        gt = (float(record['center_x']), float(record['center_y']))
        trace = _trace_pair(matcher, search, reference, gt)
        category = _classify(trace, gt)
        categories[category] += 1
        inputs = [attempt['input'] for attempt in trace['verification_attempts']]
        outputs = [attempt['output'] for attempt in trace['verification_attempts'] if attempt['output'] is not None]
        row = {
            'sample_id': record['sample_id'],
            'source_layout': record.get('source_layout'),
            'old_category': 'G. other',
            'new_category': category,
            'gt_x': gt[0], 'gt_y': gt[1],
            **_stage('raw_fused', trace['raw_fused'], gt, source='full_resolution_weighted_fusion'),
            **_stage('coarse_union', trace['candidate_union_raw'], gt),
            **_stage('after_nms', trace['rank_after_nms'], gt),
            **_stage('after_cheap_rerank', trace['cheap_all'], gt),
            **_stage('shortlist', trace['shortlist'], gt),
            **_stage('verification_input', inputs, gt),
            **_stage('rotation_scale_output', outputs, gt),
            **_stage('final_distinct', trace['distinct'], gt),
            **_stage('equivalent_group', trace['equivalent'], gt),
            'phase_response': trace['phase_response'],
            'ecc_score': trace['ecc_score'],
            'center_rule_candidate_count': len(trace['equivalent']),
            'center_rule_selected_x': _center(trace['selected'])[0],
            'center_rule_selected_y': _center(trace['selected'])[1],
            'center_rule_selected_distance_px': float(np.hypot(trace['final_center'][0] - gt[0], trace['final_center'][1] - gt[1])),
            'final_selected_x': trace['final_center'][0],
            'final_selected_y': trace['final_center'][1],
        }
        rows.append(row)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'other_failure_breakdown.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        'samples': len(rows),
        'baseline_configuration': {'cheap_rerank_top_k': 25, 'verify_top_k': 25},
        'categories': dict(categories),
    }
    (output / 'other_failure_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
