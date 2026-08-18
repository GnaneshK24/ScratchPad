"""Deterministic, cacheable Top-30 classical reranker ablation on tuning data."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .classical_matcher import ClassicalMatcherConfig, ClassicalSEMLocalizer, _gray, _representations, _variant
from .coordinates import top_left_to_center
from .dataset import SEMLocalizationDataset
from .config import LocalizationConfig


COMPONENTS = ('intensity', 'local', 'gradient', 'gx', 'gy', 'bandpass', 'edge', 'structure')


def _json(value):
    if isinstance(value, dict): return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)): return [_json(item) for item in value]
    if isinstance(value, np.generic): return value.item()
    return value


def _distinct(candidates):
    result = []
    for candidate in sorted(candidates, key=lambda item: item['score'], reverse=True):
        if all(np.hypot(candidate['x'] - other['x'], candidate['y'] - other['y']) >= 100 for other in result): result.append(candidate)
    return result or candidates[:1]


def _features(dataset, record, candidates):
    config = ClassicalMatcherConfig(cheap_rerank_top_k=30, verify_top_k=30)
    matcher = ClassicalSEMLocalizer(config)
    search = _gray(cv2.imread(str(dataset.root / record['search']), cv2.IMREAD_GRAYSCALE))
    reference = _gray(cv2.imread(str(dataset.root / record['reference']), cv2.IMREAD_GRAYSCALE))
    if reference.shape == (1000, 1000): reference = cv2.resize(reference, (100, 100), interpolation=cv2.INTER_AREA)
    search_repr = _representations(search, matcher.config)
    # Compute representation support from duplicate local maxima before the
    # 100 px periodic-basin collapse.
    for candidate in candidates:
        candidate['support_count'] = len({
            other.get('candidate_source') or 'unknown'
            for other in candidates
            if np.hypot(candidate['x'] - other['x'], candidate['y'] - other['y']) <= 5.
        })
    features = []
    for candidate in _distinct(candidates):
        x, y = round(candidate['x']), round(candidate['y'])
        ref_local = _representations(_variant(reference, candidate['rotation'], candidate['scale']), matcher.config)['local']
        dx, dy, ecc, phase = matcher._align(search_repr['local'][y:y + 100, x:x + 100], ref_local)
        features.append({**candidate, 'phase_response': phase, 'ecc_response': ecc,
                         'rotation_consistency': -abs(float(candidate['rotation'])),
                         'scale_consistency': -abs(float(candidate['scale']) - 1.),
                         'aligned_center_x': candidate['x'] + dx + 50,
                         'aligned_center_y': candidate['y'] + dy + 50})
    return features


def _robust_z(values):
    values = np.asarray(values, dtype=float); median = float(np.median(values)); mad = float(np.median(np.abs(values - median)))
    return np.clip((values - median) / max(1.4826 * mad, .002), -4., 4.)


def _rank_values(values):
    order = np.argsort(-np.asarray(values, dtype=float)); ranks = np.empty(len(values), dtype=float); ranks[order] = np.arange(1, len(values) + 1)
    return 1. / ranks


def _score(candidates, policy):
    fields = ('score', *COMPONENTS, 'phase_response', 'ecc_response', 'rotation_consistency', 'scale_consistency', 'support_count')
    z = {field: _robust_z([candidate[field] if field not in COMPONENTS else candidate['components'][field] for candidate in candidates]) for field in fields}
    ranks = {field: _rank_values([candidate[field] if field not in COMPONENTS else candidate['components'][field] for candidate in candidates]) for field in fields}
    weights = {'score': .35, 'intensity': .08, 'local': .08, 'gradient': .10, 'gx': .05, 'gy': .05,
               'bandpass': .06, 'edge': .06, 'structure': .05, 'phase_response': .05,
               'ecc_response': .04, 'rotation_consistency': .01, 'scale_consistency': .01, 'support_count': .01}
    values = []
    for index, candidate in enumerate(candidates):
        if policy == 'fused_baseline':
            value = candidate['score']
        elif policy == 'weighted_normalized':
            value = sum(weights[field] * z[field][index] for field in fields)
        elif policy == 'rank_aggregation':
            value = sum(weights[field] * ranks[field][index] for field in fields)
        elif policy == 'robust_voting':
            votes = sum(z[field][index] > 0. for field in fields)
            value = .75 * z['score'][index] + .25 * votes
        else:
            raise ValueError(policy)
        values.append(float(value))
    return values


def _select(candidates, policy):
    candidates = [dict(candidate) for candidate in candidates]
    values = _score(candidates, policy)
    for candidate, value in zip(candidates, values): candidate['rerank_score'] = value
    candidates.sort(key=lambda item: item['rerank_score'], reverse=True)
    # Visual equivalence remains the unchanged official centre-rule gate.  A
    # reranker may choose a different leading basin but may not add a generic
    # centre preference outside a visual-equivalence group.
    visual_best = candidates[0]['score']
    equivalent = [candidate for candidate in candidates if visual_best - candidate['score'] <= .012]
    return min(equivalent, key=lambda item: np.hypot(item['x'] + 50 - 500, item['y'] + 50 - 500))


def _metrics(errors):
    errors = np.asarray(errors, dtype=float)
    return {'samples': len(errors), **{f'accuracy_at_{value}px': float(np.mean(errors <= value)) for value in (1, 2, 5, 10)},
            'median_error_px': float(np.median(errors)), 'mean_error_px': float(np.mean(errors)), 'p95_error_px': float(np.percentile(errors, 95)),
            **{f'error_over_{value}px': float(np.mean(errors > value)) for value in (20, 50, 100)}}


def main():
    parser = argparse.ArgumentParser(description='Classical reranker ablation using cached Top-30 verification evidence.')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--candidate-cache-dir', default='evaluation/tuning/candidate_cache')
    parser.add_argument('--output-dir', default='evaluation/tuning/reranker')
    parser.add_argument('--tuning-layout-count', type=int, default=12)
    parser.add_argument('--rebuild-features', action='store_true')
    args = parser.parse_args()
    dataset = SEMLocalizationDataset(args.dataset, LocalizationConfig())
    layouts = sorted({record['source_layout'] for record in dataset.records})
    records = [record for record in dataset.records if record['source_layout'] in layouts[:args.tuning_layout_count]]
    output = Path(args.output_dir); feature_dir = output / 'feature_cache'; feature_dir.mkdir(parents=True, exist_ok=True)
    policies = ('fused_baseline', 'weighted_normalized', 'rank_aggregation', 'robust_voting')
    errors = {policy: [] for policy in policies}
    for record in tqdm(records, desc='Caching reranker features', unit='sample', dynamic_ncols=True):
        feature_path = feature_dir / f"{record['sample_id']}.json"
        if feature_path.exists() and not args.rebuild_features:
            candidates = json.loads(feature_path.read_text(encoding='utf-8'))['candidates']
        else:
            cached = json.loads((Path(args.candidate_cache_dir) / f"{record['sample_id']}.json").read_text(encoding='utf-8'))
            candidates = _features(dataset, record, [item for item in cached['verified'][:30] if item is not None])
            feature_path.write_text(json.dumps(_json({'record': record, 'candidates': candidates}), indent=2), encoding='utf-8')
        gt = (float(record['center_x']), float(record['center_y']))
        for policy in policies:
            selected = _select(candidates, policy)
            errors[policy].append(float(np.hypot(selected['aligned_center_x'] - gt[0], selected['aligned_center_y'] - gt[1])))
    rows = [{'policy': policy, **_metrics(errors[policy])} for policy in policies]
    with (output / 'reranker_ablation.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output / 'reranker_ablation_summary.json').write_text(json.dumps({'samples': len(records), 'policies': rows,
        'note': 'Phase/ECC are diagnostic features for every candidate here; this experiment is not promoted unless it beats the baseline.'}, indent=2), encoding='utf-8')
    print(json.dumps({'samples': len(records), 'policies': rows}, indent=2))


if __name__ == '__main__':
    main()
