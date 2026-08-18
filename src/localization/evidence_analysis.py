"""Cache-only FinFET diversity, equivalence, and centre-rule diagnostics.

All policy choices in this module are tuned on the existing 72-sample source
layout split.  It never invokes candidate generation or verification.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .classical_matcher import ClassicalMatcherConfig, ClassicalSEMLocalizer, _gray, _representations, _variant
from .coordinates import top_left_to_center
from .dataset import SEMLocalizationDataset
from .config import LocalizationConfig


def _distance(candidate, point):
    return float(np.hypot(candidate['x'] + 50 - point[0], candidate['y'] + 50 - point[1]))


def _distinct(candidates):
    ordered = sorted((candidate for candidate in candidates if candidate is not None), key=lambda value: value['score'], reverse=True)
    result = []
    for candidate in ordered:
        if all(np.hypot(candidate['x'] - other['x'], candidate['y'] - other['y']) >= 100 for other in result):
            result.append(candidate)
    return result or ordered[:1]


def _center_select(candidates):
    return min(candidates, key=lambda value: np.hypot(value['x'] + 50 - 500, value['y'] + 50 - 500))


def _mad(values):
    values = np.asarray(values, dtype=float)
    return float(np.median(np.abs(values - np.median(values))))


def _equivalent(candidates, family, value):
    """Return the top visual-score cluster under one fixed global rule."""
    candidates = sorted(candidates, key=lambda item: item['score'], reverse=True)
    best = candidates[0]['score']
    scores = [item['score'] for item in candidates]
    if family == 'absolute':
        return [item for item in candidates if best - item['score'] <= value]
    if family == 'relative':
        return [item for item in candidates if item['score'] >= best * value]
    robust_scale = max(1.4826 * _mad(scores), .002)
    if family == 'robust_gap':
        return [item for item in candidates if (best - item['score']) / robust_scale <= value]
    if family == 'score_cluster':
        group = [candidates[0]]
        for previous, candidate in zip(candidates, candidates[1:]):
            if (previous['score'] - candidate['score']) > value * robust_scale:
                break
            group.append(candidate)
        return group
    raise ValueError(f'Unknown equivalence family: {family}')


def _group_support(candidates, radius=5.):
    """Merge repeated local maxima and retain representation-level support."""
    groups = []
    for candidate in sorted((item for item in candidates if item is not None), key=lambda item: item['score'], reverse=True):
        group = next((group for group in groups if np.hypot(candidate['x'] - group['x'], candidate['y'] - group['y']) <= radius), None)
        if group is None:
            groups.append({'x': candidate['x'], 'y': candidate['y'], 'score': candidate['score'], 'members': [candidate]})
        else:
            group['members'].append(candidate)
            if candidate['score'] > group['score']:
                group.update({'x': candidate['x'], 'y': candidate['y'], 'score': candidate['score']})
    for group in groups:
        members = group['members']
        by_source = {}
        for member in members:
            source = member.get('candidate_source') or 'unknown'
            by_source[source] = max(by_source.get(source, -np.inf), member['score'])
        group['representations_supporting_candidate'] = sorted(by_source)
        group['number_of_supporting_representations'] = len(by_source)
        group['per_representation_scores'] = by_source
        group['best_representation_score'] = max(by_source.values())
        group['mean_normalized_score'] = float(np.mean(list(by_source.values())))
        group['median_normalized_score'] = float(np.median(list(by_source.values())))
    return groups


def _diversity_choice(candidates, policy):
    groups = _group_support(candidates)
    if policy == 'weighted_fusion':
        for group in groups: group['policy_score'] = group['score']
    else:
        source_scores = defaultdict(list)
        for group in groups:
            for source, score in group['per_representation_scores'].items(): source_scores[source].append(score)
        source_rank = {}
        source_z = {}
        for source, scores in source_scores.items():
            ordered = sorted(scores, reverse=True)
            source_rank[source] = {score: index + 1 for index, score in enumerate(ordered)}
            median = float(np.median(scores)); scale = max(1.4826 * _mad(scores), .002)
            source_z[source] = (median, scale)
        for group in groups:
            ranks = [1. / source_rank[source][score] for source, score in group['per_representation_scores'].items()]
            zscores = [(score - source_z[source][0]) / source_z[source][1]
                       for source, score in group['per_representation_scores'].items()]
            if policy == 'rank_aggregation':
                group['policy_score'] = float(np.mean(ranks))
            elif policy == 'representation_voting':
                group['policy_score'] = float(group['number_of_supporting_representations'] + .01 * max(zscores))
            elif policy == 'hybrid_voting_normalized_score':
                group['policy_score'] = float(group['number_of_supporting_representations'] + .25 * np.mean(zscores))
            else:
                raise ValueError(policy)
    groups.sort(key=lambda item: item['policy_score'], reverse=True)
    return groups[0]


def _align(record, root, matcher, candidate, cache):
    key = (record['sample_id'], round(candidate['x'], 4), round(candidate['y'], 4), candidate.get('rotation'), candidate.get('scale'))
    if key in cache: return cache[key]
    search = _gray(cv2.imread(str(root / record['search']), cv2.IMREAD_GRAYSCALE))
    reference = _gray(cv2.imread(str(root / record['reference']), cv2.IMREAD_GRAYSCALE))
    if reference.shape == (1000, 1000): reference = cv2.resize(reference, (100, 100), interpolation=cv2.INTER_AREA)
    search_repr = _representations(search, matcher.config)
    x, y = round(candidate['x']), round(candidate['y'])
    local_ref = _representations(_variant(reference, candidate.get('rotation', 0.), candidate.get('scale', 1.)), matcher.config)['local']
    dx, dy, _, _ = matcher._align(search_repr['local'][y:y + 100, x:x + 100], local_ref)
    cache[key] = top_left_to_center(candidate['x'] + dx, candidate['y'] + dy)
    return cache[key]


def _metrics(errors):
    errors = np.asarray(errors, dtype=float)
    return {
        'samples': int(len(errors)),
        **{f'accuracy_at_{value}px': float(np.mean(errors <= value)) for value in (1, 2, 5, 10)},
        'median_error_px': float(np.median(errors)), 'p95_error_px': float(np.percentile(errors, 95)),
        **{f'error_over_{value}px': float(np.mean(errors > value)) for value in (20, 50, 100)},
    }


def main():
    parser = argparse.ArgumentParser(description='Analyse cached Top-50 FinFET candidate evidence.')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--cache-dir', default='evaluation/tuning/candidate_cache')
    parser.add_argument('--output-dir', default='evaluation/tuning/evidence_analysis')
    parser.add_argument('--tuning-layout-count', type=int, default=12)
    args = parser.parse_args()
    dataset = SEMLocalizationDataset(args.dataset, LocalizationConfig())
    layouts = sorted({record['source_layout'] for record in dataset.records})
    records = [record for record in dataset.records if record['source_layout'] in layouts[:args.tuning_layout_count]]
    caches = [json.loads((Path(args.cache_dir) / f"{record['sample_id']}.json").read_text(encoding='utf-8')) for record in records]
    matcher = ClassicalSEMLocalizer(ClassicalMatcherConfig(cheap_rerank_top_k=30, verify_top_k=30))
    alignment_cache = {}
    sample_rows, strategy_errors = [], defaultdict(list)
    methods = [('absolute', .012), ('relative', .970), ('relative', .975), ('relative', .980), ('relative', .985),
               ('robust_gap', .5), ('robust_gap', 1.), ('robust_gap', 1.5), ('robust_gap', 2.),
               ('score_cluster', .5), ('score_cluster', 1.), ('score_cluster', 1.5), ('score_cluster', 2.)]
    diversity_policies = ('weighted_fusion', 'rank_aggregation', 'representation_voting', 'hybrid_voting_normalized_score')
    candidate_oracle, center_oracle, baseline_errors = [], [], []
    for record, payload in zip(records, caches):
        gt = (float(record['center_x']), float(record['center_y']))
        verified = [item for item in payload['verified'][:30] if item is not None]
        distinct = _distinct(verified)
        baseline_equivalent = _equivalent(distinct, 'absolute', .012)
        baseline = _center_select(baseline_equivalent)
        baseline_center = _align(record, dataset.root, matcher, baseline, alignment_cache)
        baseline_error = float(np.hypot(baseline_center[0] - gt[0], baseline_center[1] - gt[1]))
        baseline_errors.append(baseline_error)
        gt_candidate = min(distinct, key=lambda item: _distance(item, gt)) if distinct else None
        gt_candidate_score = None if gt_candidate is None or _distance(gt_candidate, gt) > 5. else gt_candidate['score']
        score_values = [item['score'] for item in distinct]
        row = {
            'sample_id': record['sample_id'], 'source_layout': record['source_layout'],
            'sample_type': ('failed_equivalence' if baseline_error > 5. and gt_candidate_score is not None and gt_candidate not in baseline_equivalent
                            else 'successful_center_tie' if baseline_error <= 5. and len(baseline_equivalent) > 1
                            else 'ambiguous_periodic' if len(baseline_equivalent) > 1 else 'unique_match'),
            'best_score': distinct[0]['score'], 'gt_candidate_score': gt_candidate_score,
            'score_difference': None if gt_candidate_score is None else distinct[0]['score'] - gt_candidate_score,
            'score_ratio': None if gt_candidate_score is None else gt_candidate_score / max(abs(distinct[0]['score']), 1e-6),
            'median_candidate_score': float(np.median(score_values)), 'mad_candidate_score': _mad(score_values),
            'std_candidate_score': float(np.std(score_values)), 'number_equivalent_existing': len(baseline_equivalent),
            'distance_gt_to_center': float(np.hypot(gt[0] - 500, gt[1] - 500)),
            'distance_selected_to_center': float(np.hypot(baseline['x'] + 50 - 500, baseline['y'] + 50 - 500)),
            'baseline_error_px': baseline_error,
        }
        sample_rows.append(row)
        raw_top50 = payload['raw_fused'][:50]
        candidate_oracle.append(any(_distance(candidate, gt) <= 5. for candidate in raw_top50))
        center_oracle.append(any(_distance(candidate, gt) <= 5. for candidate in verified))
        for family, value in methods:
            equivalent = _equivalent(distinct, family, value)
            selected = _center_select(equivalent)
            center = _align(record, dataset.root, matcher, selected, alignment_cache)
            strategy_errors[f'{family}:{value:g}'].append(float(np.hypot(center[0] - gt[0], center[1] - gt[1])))
        for policy in diversity_policies:
            selected = _diversity_choice(verified, policy)
            # This policy experiment tests the winner only; it deliberately
            # does not redefine the official visual-equivalence contract.
            proxy = max(selected['members'], key=lambda item: item['score'])
            center = _align(record, dataset.root, matcher, proxy, alignment_cache)
            strategy_errors[f'diversity:{policy}'].append(float(np.hypot(center[0] - gt[0], center[1] - gt[1])))
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    with (output / 'equivalence_score_analysis.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(sample_rows[0])); writer.writeheader(); writer.writerows(sample_rows)
    analyses = defaultdict(list)
    for row in sample_rows: analyses[row['sample_type']].append(row)
    summary = {kind: {'samples': len(rows), 'mean_best_score': float(np.mean([r['best_score'] for r in rows])),
                      'mean_score_difference': float(np.mean([r['score_difference'] for r in rows if r['score_difference'] is not None])) if any(r['score_difference'] is not None for r in rows) else None,
                      'mean_equivalent_count': float(np.mean([r['number_equivalent_existing'] for r in rows]))}
               for kind, rows in analyses.items()}
    (output / 'equivalence_score_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    threshold_rows = []
    for family, value in methods:
        threshold_rows.append({'family': family, 'value': value, **_metrics(strategy_errors[f'{family}:{value:g}'])})
    with (output / 'equivalence_threshold_ablation.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(threshold_rows[0])); writer.writeheader(); writer.writerows(threshold_rows)
    diversity_rows = [{'policy': policy, **_metrics(strategy_errors[f'diversity:{policy}'])} for policy in diversity_policies]
    with (output / 'candidate_diversity_ablation.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(diversity_rows[0])); writer.writeheader(); writer.writerows(diversity_rows)
    oracle = {'samples': len(records), 'current_accuracy_at_5px': _metrics(baseline_errors)['accuracy_at_5px'],
              'candidate_oracle_accuracy_at_5px_raw_top50': float(np.mean(candidate_oracle)),
              'center_rule_oracle_accuracy_at_5px_verified_top30': float(np.mean(center_oracle)),
              'definition': 'The centre-rule oracle assumes the official GT is selected whenever its verified Top-30 candidate is supplied to a perfect legitimate-equivalence group.'}
    (output / 'center_rule_oracle.json').write_text(json.dumps(oracle, indent=2), encoding='utf-8')
    print(json.dumps({'equivalence_summary': summary, 'thresholds': threshold_rows, 'diversity': diversity_rows, 'oracle': oracle}, indent=2))


if __name__ == '__main__':
    main()
