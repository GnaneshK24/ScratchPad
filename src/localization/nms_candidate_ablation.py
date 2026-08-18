"""Fast, cacheable NMS candidate-recall ablation on tuning layouts only.

It deliberately stops before cheap re-ranking and local verification.  This
isolates whether a GT basin was destroyed by NMS rather than by later stages.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .classical_matcher import ClassicalMatcherConfig, ClassicalSEMLocalizer
from .config import LocalizationConfig
from .dataset import SEMLocalizationDataset


CONFIGURATIONS = (
    ('spatial_r12', 'spatial', 12),
    ('spatial_r16', 'spatial', 16),
    ('spatial_r20_baseline', 'spatial', 20),
    ('spatial_r24', 'spatial', 24),
    ('score_aware_r20', 'score_aware', 20),
    ('clustered_r20', 'clustered', 20),
    ('representation_aware_r20', 'representation_aware', 20),
)


def _distance(candidate, gt):
    return float(np.hypot(candidate['x'] + 50 - gt[0], candidate['y'] + 50 - gt[1]))


def _select(raw, mode, radius, limit=100, score_margin=0.20, cluster_limit=2):
    """Apply a deterministic non-destructive NMS policy to pre-NMS evidence."""
    selected = []
    for item in raw:
        close = [other for other in selected if np.hypot(item['x'] - other['x'], item['y'] - other['y']) < radius]
        if not close:
            selected.append(dict(item))
        elif mode == 'spatial':
            continue
        elif mode == 'score_aware':
            # Retain a close peak only when its normalized evidence is not
            # clearly weaker than the existing local representative.
            if max(other['candidate_priority'] - item['candidate_priority'] for other in close) <= score_margin:
                selected.append(dict(item))
        elif mode == 'representation_aware':
            if not any(other['candidate_source'] == item['candidate_source'] for other in close):
                selected.append(dict(item))
        elif mode == 'clustered':
            # A cluster can retain two independently-generated high-evidence
            # representatives.  This is bounded, source-aware diversity—not
            # a blanket NMS disable.
            cluster_best = max(other['candidate_priority'] for other in close)
            distinct_sources = {other['candidate_source'] for other in close}
            if (len(close) < cluster_limit and item['candidate_source'] not in distinct_sources
                    and item['candidate_priority'] >= cluster_best - score_margin):
                selected.append(dict(item))
        else:
            raise ValueError(f'Unknown NMS mode: {mode}')
        if len(selected) >= limit:
            break
    return selected


def _row(name, mode, radius, raw, selected, gt):
    raw_has_gt = any(_distance(candidate, gt) <= 10. for candidate in raw)
    selected_has_gt = any(_distance(candidate, gt) <= 10. for candidate in selected)
    values = {
        'configuration': name,
        'mode': mode,
        'nms_radius_px': radius,
        'raw_candidate_count': len(raw),
        'selected_candidate_count': len(selected),
        'gt_available_before_nms_within_10px': raw_has_gt,
        'gt_lost_directly_by_nms_within_10px': raw_has_gt and not selected_has_gt,
    }
    for k in (5, 10, 20, 30, 50):
        values[f'candidate_recall_at_{k}_within_5px'] = any(_distance(candidate, gt) <= 5. for candidate in selected[:k])
    return values


def _summary(rows):
    result = {'samples': len(rows)}
    for key in ('raw_candidate_count', 'selected_candidate_count'):
        result[f'mean_{key}'] = float(np.mean([row[key] for row in rows]))
    result['nms_induced_gt_losses'] = int(sum(row['gt_lost_directly_by_nms_within_10px'] for row in rows))
    result['nms_induced_gt_loss_rate'] = float(np.mean([row['gt_lost_directly_by_nms_within_10px'] for row in rows]))
    for k in (5, 10, 20, 30, 50):
        key = f'candidate_recall_at_{k}_within_5px'
        result[key] = float(np.mean([row[key] for row in rows]))
    return result


def main():
    parser = argparse.ArgumentParser(description='NMS-only candidate preservation ablation on tuning layouts.')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output-dir', default='evaluation/tuning/nms_ablation')
    parser.add_argument('--tuning-layout-count', type=int, default=12)
    parser.add_argument('--rebuild-cache', action='store_true')
    args = parser.parse_args()
    dataset = SEMLocalizationDataset(args.dataset, LocalizationConfig())
    layouts = sorted({record['source_layout'] for record in dataset.records})
    tuning_layouts = layouts[:args.tuning_layout_count]
    records = [record for record in dataset.records if record['source_layout'] in tuning_layouts]
    output = Path(args.output_dir)
    evidence_dir = output / 'candidate_evidence_cache'
    evidence_dir.mkdir(parents=True, exist_ok=True)
    by_config = {name: [] for name, _, _ in CONFIGURATIONS}
    for record in tqdm(records, desc='Caching pre-NMS candidate evidence', unit='sample', dynamic_ncols=True):
        cache_path = evidence_dir / f"{record['sample_id']}.json"
        if cache_path.exists() and not args.rebuild_cache:
            payload = json.loads(cache_path.read_text(encoding='utf-8'))
        else:
            search = cv2.imread(str(dataset.root / record['search']), cv2.IMREAD_GRAYSCALE)
            reference = cv2.imread(str(dataset.root / record['reference']), cv2.IMREAD_GRAYSCALE)
            if reference.shape == (1000, 1000):
                reference = cv2.resize(reference, (100, 100), interpolation=cv2.INTER_AREA)
            payload = {'record': record, 'raw_by_configuration': {}}
            for name, _, radius in CONFIGURATIONS:
                trace = {}
                config = ClassicalMatcherConfig(nms_radius=radius)
                ClassicalSEMLocalizer(config)._coarse(search, reference, limit=config.top_k, trace=trace)
                payload['raw_by_configuration'][name] = trace['candidate_union_raw']
            cache_path.write_text(json.dumps(payload), encoding='utf-8')
        gt = (float(record['center_x']), float(record['center_y']))
        for name, mode, radius in CONFIGURATIONS:
            raw = payload['raw_by_configuration'][name]
            selected = _select(raw, mode, radius)
            by_config[name].append(_row(name, mode, radius, raw, selected, gt))
    rows = [_summary(by_config[name]) | {'configuration': name, 'mode': mode, 'nms_radius_px': radius}
            for name, mode, radius in CONFIGURATIONS]
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'nms_candidate_ablation.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    details = [row for name, _, _ in CONFIGURATIONS for row in by_config[name]]
    with (output / 'nms_candidate_details.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(details[0]))
        writer.writeheader(); writer.writerows(details)
    report = {'tuning_layouts': tuning_layouts, 'samples': len(records), 'configurations': rows,
              'note': 'Candidate-only diagnostic; no final verification or matcher behavior was changed.'}
    (output / 'nms_candidate_ablation_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
