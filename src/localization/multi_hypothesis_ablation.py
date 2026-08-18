"""Non-destructive Top-50 hypothesis-retention experiment for FinFET tuning.

This module is intentionally separate from production inference.  It preserves
parents when an existing refinement crosses a configurable spatial basin
boundary, then applies the unchanged score and official conditional centre tie.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .classical_matcher import ClassicalMatcherConfig, ClassicalSEMLocalizer, _gray, _representations, _variant
from .config import LocalizationConfig
from .dataset import SEMLocalizationDataset


ROOT_LIMIT = 50
LEVEL_A_KEEP = 20
LEVEL_B_KEEP = 15
DEDUP_RADIUS = 3.0
EQUIVALENCE_MARGIN = .012


def _json(value):
    if isinstance(value, dict): return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)): return [_json(item) for item in value]
    if isinstance(value, np.generic): return value.item()
    return value


def _distance(a, b): return float(np.hypot(a['x'] - b['x'], a['y'] - b['y']))
def _gt_distance(hypothesis, gt): return float(np.hypot(hypothesis['x'] + 50 - gt[0], hypothesis['y'] + 50 - gt[1]))


def _rank(hypotheses, limit=None):
    result = sorted(hypotheses, key=lambda item: (-item['score'], item['hypothesis_id']))
    return result if limit is None else result[:limit]


def _deduplicate(hypotheses):
    """Only collapse numerically duplicate hypotheses, never periodic cells."""
    kept = []
    for hypothesis in _rank(hypotheses):
        if all(_distance(hypothesis, other) > DEDUP_RADIUS for other in kept): kept.append(hypothesis)
    return kept


def _expand(parent, x, y, score, stage, suffix, threshold, extra=None):
    displacement = float(np.hypot(x - parent['x'], y - parent['y']))
    extra = extra or {}
    if displacement < threshold:
        return [{**parent, 'x': float(x), 'y': float(y), 'score': float(score), 'stage': stage,
                 'displacement_from_parent': displacement,
                 'displacement_from_root': float(np.hypot(x - parent['root_x'], y - parent['root_y'])), **extra}], None
    child = {**parent, 'hypothesis_id': f"{parent['hypothesis_id']}-{suffix}", 'parent_id': parent['hypothesis_id'],
             'x': float(x), 'y': float(y), 'score': float(score), 'stage': stage,
             'displacement_from_parent': displacement,
             'displacement_from_root': float(np.hypot(x - parent['root_x'], y - parent['root_y'])), **extra}
    # The parent remains exactly as it was before the basin crossing.
    return [dict(parent), child], child


def _phase(search_local, reference_local):
    shift, response = cv2.phaseCorrelate(reference_local.astype(np.float32), search_local.astype(np.float32))
    return float(shift[0]), float(shift[1]), float(np.clip(response, 0., 1.))


def _ecc(search_local, reference_local, config):
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        score, warp = cv2.findTransformECC(reference_local.astype(np.float32), search_local.astype(np.float32), warp,
                                            cv2.MOTION_TRANSLATION,
                                            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                             config.ecc_iterations, config.ecc_epsilon))
        dx, dy = -float(warp[0, 2]), -float(warp[1, 2])
        if abs(dx) <= 4 and abs(dy) <= 4: return dx, dy, float(np.clip(score, -1., 1.))
    except cv2.error:
        pass
    return 0., 0., 0.


def _local_images(search_repr, reference, hypothesis, config):
    x, y = round(hypothesis['x']), round(hypothesis['y'])
    if x < 0 or y < 0 or x + 100 > search_repr['local'].shape[1] or y + 100 > search_repr['local'].shape[0]: return None
    patch = search_repr['local'][y:y + 100, x:x + 100]
    ref = _representations(_variant(reference, hypothesis.get('rotation', 0.), hypothesis.get('scale', 1.)), config)['local']
    return patch, ref


def _select(hypotheses):
    ranked = _rank(hypotheses)
    best_score = ranked[0]['score']
    equivalent = [item for item in ranked if item['score'] >= best_score - EQUIVALENCE_MARGIN]
    best = min(equivalent, key=lambda item: (item['x'] + 50 - 500) ** 2 + (item['y'] + 50 - 500) ** 2)
    return best, equivalent


def _oracle(hypotheses, gt): return bool(any(_gt_distance(item, gt) <= 5. for item in hypotheses))


def _metrics(rows):
    errors = np.asarray([row['error_px'] for row in rows], dtype=float)
    return {'samples': len(rows), **{f'accuracy_at_{value}px': float(np.mean(errors <= value)) for value in (1, 2, 5, 10)},
            'median_error_px': float(np.median(errors)), 'mean_error_px': float(np.mean(errors)),
            'p90_error_px': float(np.percentile(errors, 90)), 'p95_error_px': float(np.percentile(errors, 95)),
            'max_error_px': float(errors.max()), **{f'error_over_{value}px': float(np.mean(errors > value)) for value in (20, 50, 100)},
            'runtime_ms_per_sample': float(np.mean([row['runtime_ms'] for row in rows]))}


def _baseline_metrics(dataset, records, raw_cache_dir, matcher, runtime_ms):
    """Recover the complete existing Top-30 metric set from its cached candidates."""
    rows = []
    for record in records:
        verified = json.loads((Path(raw_cache_dir) / f"{record['sample_id']}.json").read_text(encoding='utf-8'))['verified'][:30]
        hypotheses = [{**candidate, 'hypothesis_id': f'B{index:02d}'} for index, candidate in enumerate(verified) if candidate is not None]
        distinct = []
        for candidate in _rank(hypotheses):
            if all(_distance(candidate, other) >= 100 for other in distinct): distinct.append(candidate)
        selected, _ = _select(distinct)
        local = _local_images(_representations(_gray(cv2.imread(str(dataset.root / record['search']), cv2.IMREAD_GRAYSCALE)), matcher.config),
                              cv2.resize(_gray(cv2.imread(str(dataset.root / record['reference']), cv2.IMREAD_GRAYSCALE)), (100, 100), interpolation=cv2.INTER_AREA),
                              selected, matcher.config)
        dx, dy, _ = (0., 0., 0.) if local is None else _ecc(*local, matcher.config)
        gt = (float(record['center_x']), float(record['center_y']))
        rows.append({'error_px': float(np.hypot(selected['x'] + dx + 50 - gt[0], selected['y'] + dy + 50 - gt[1])),
                     'runtime_ms': runtime_ms})
    return _metrics(rows)


def _prepare(record, raw, matcher, search_repr, reference):
    roots = []
    for index, candidate in enumerate(raw[:ROOT_LIMIT]):
        roots.append({'hypothesis_id': f'H{index:02d}', 'root_id': f'H{index:02d}', 'parent_id': None, 'stage': 'raw',
                      'x': float(candidate['x']), 'y': float(candidate['y']), 'score': float(candidate['score']),
                      'root_x': float(candidate['x']), 'root_y': float(candidate['y']),
                      'rotation': 0., 'scale': 1., 'coarse_score': float(candidate['score']),
                      'coarse_rotation': 0., 'candidate_source': 'fused_full',
                      'displacement_from_parent': 0., 'displacement_from_root': 0.})
    cheap = matcher._cheap_rerank(search_repr, reference, roots)
    cheap_by_id = {candidate['hypothesis_id']: candidate for candidate in cheap}
    # A parent retained after a cheap-local split keeps its existing raw
    # fused evidence.  This keeps the experiment self-contained and leaves
    # the production localizer's candidate payload and ranking untouched.
    for root in roots:
        cheap_by_id[root['hypothesis_id']]['cheap_parent_score'] = root['score']
    return roots, cheap_by_id


def _run_threshold(record, roots, cheap_by_id, matcher, search_repr, reference, threshold, verify_cache, phase_cache, ecc_cache):
    trace = []
    started = time.perf_counter()
    # Level A: existing cheap structural local verification for every raw root.
    level_a = []
    for root in roots:
        candidate = cheap_by_id[root['hypothesis_id']]
        expanded, _ = _expand(root, candidate['x'], candidate['y'], candidate['cheap_score'], 'cheap_local', 'A', threshold,
                              {'rotation': 0., 'scale': 1., 'candidate_source': candidate.get('candidate_source', 'fused_full')})
        if len(expanded) == 2:
            expanded[0]['score'] = float(candidate['cheap_parent_score'])
        level_a.extend(expanded)
    level_a = _deduplicate(level_a)
    level_a_top = _rank(level_a, LEVEL_A_KEEP)
    trace.extend({**item, 'trace_stage': 'level_a'} for item in level_a)
    # Level B: existing rotation/scale local verification, followed by phase correlation.
    rotation_expanded = []
    rotation_events = []
    for hypothesis in level_a_top:
        key = (round(hypothesis['x'], 4), round(hypothesis['y'], 4), round(hypothesis.get('coarse_rotation', 0.), 4))
        if key not in verify_cache:
            verification = matcher._verify(search_repr, reference, hypothesis)
            verify_cache[key] = verification
        verification = verify_cache[key]
        if verification is None:
            rotation_expanded.append(dict(hypothesis)); continue
        expanded, child = _expand(hypothesis, verification['x'], verification['y'], verification['score'], 'rotation_scale', 'R', threshold,
                                  {'rotation': verification['rotation'], 'scale': verification['scale'],
                                   'candidate_source': verification.get('candidate_source', hypothesis.get('candidate_source'))})
        rotation_expanded.extend(expanded)
        rotation_events.append({'parent': dict(hypothesis), 'child': None if child is None else dict(child),
                                'verified': dict(verification), 'parent_preserved': child is not None})
    rotation_expanded = _deduplicate(rotation_expanded)
    phase_expanded = []
    for hypothesis in rotation_expanded:
        key = (round(hypothesis['x'], 4), round(hypothesis['y'], 4), round(hypothesis.get('rotation', 0.), 4), round(hypothesis.get('scale', 1.), 4))
        if key not in phase_cache:
            local = _local_images(search_repr, reference, hypothesis, matcher.config)
            phase_cache[key] = None if local is None else _phase(*local)
        phase = phase_cache[key]
        if phase is None:
            phase_expanded.append(dict(hypothesis)); continue
        dx, dy, response = phase
        expanded, _ = _expand(hypothesis, hypothesis['x'] + dx, hypothesis['y'] + dy, hypothesis['score'], 'phase', 'P', threshold,
                              {'phase_response': response})
        phase_expanded.extend(expanded)
    level_b = _deduplicate(phase_expanded)
    level_b_top = _rank(level_b, LEVEL_B_KEEP)
    trace.extend({**item, 'trace_stage': 'level_b'} for item in level_b)
    # Level C: ECC final refinement on the fixed Level-B Top-15.
    level_c = []
    for hypothesis in level_b_top:
        key = (round(hypothesis['x'], 4), round(hypothesis['y'], 4), round(hypothesis.get('rotation', 0.), 4), round(hypothesis.get('scale', 1.), 4))
        if key not in ecc_cache:
            local = _local_images(search_repr, reference, hypothesis, matcher.config)
            ecc_cache[key] = None if local is None else _ecc(*local, matcher.config)
        ecc = ecc_cache[key]
        if ecc is None:
            level_c.append(dict(hypothesis)); continue
        dx, dy, response = ecc
        expanded, _ = _expand(hypothesis, hypothesis['x'] + dx, hypothesis['y'] + dy, hypothesis['score'], 'ecc', 'E', threshold,
                              {'ecc_response': response})
        level_c.extend(expanded)
    level_c = _deduplicate(level_c)
    trace.extend({**item, 'trace_stage': 'level_c'} for item in level_c)
    selected, equivalent = _select(level_c)
    gt = (float(record['center_x']), float(record['center_y']))
    result = {'sample_id': record['sample_id'], 'source_layout': record['source_layout'], 'threshold_px': threshold,
              'error_px': _gt_distance(selected, gt), 'selected_hypothesis_id': selected['hypothesis_id'],
              'selected_x': selected['x'] + 50, 'selected_y': selected['y'] + 50, 'selected_score': selected['score'],
              'equivalent_hypotheses': len(equivalent), 'runtime_ms': (time.perf_counter() - started) * 1000,
              'raw_oracle_at_5px': _oracle(roots, gt), 'level_a_oracle_at_5px': _oracle(level_a_top, gt),
              'level_b_oracle_at_5px': _oracle(level_b_top, gt), 'level_c_oracle_at_5px': _oracle(level_c, gt)}
    return result, trace, rotation_events


def _g2_rows(record, result, rotation_events, old):
    gt = (float(record['center_x']), float(record['center_y']))
    event = min(rotation_events, key=lambda item: _gt_distance(item['parent'], gt), default=None)
    child = None if event is None else event['child']
    return {'sample_id': record['sample_id'], 'threshold_px': result['threshold_px'], 'gt_x': gt[0], 'gt_y': gt[1],
            'original_root_hypothesis': None if event is None else event['parent']['hypothesis_id'],
            'original_root_x': None if event is None else event['parent']['x'] + 50,
            'original_root_y': None if event is None else event['parent']['y'] + 50,
            'old_refined_x': old.get('rotation_scale_output_x'), 'old_refined_y': old.get('rotation_scale_output_y'),
            'old_error_px': old.get('center_rule_selected_distance_px'),
            # A root which does not survive the fixed Top-20 Level-A cascade
            # never reaches rotation/scale; it was not destructively replaced.
            # Keep that distinct from a reached parent that was not retained.
            'reached_rotation_scale': event is not None,
            'parent_preserved': None if event is None else bool(event['parent_preserved']),
            'child_x': None if child is None else child['x'] + 50, 'child_y': None if child is None else child['y'] + 50,
            'new_selected_x': result['selected_x'], 'new_selected_y': result['selected_y'],
            'new_error_px': result['error_px'], 'recovered': result['error_px'] <= 5.}


def main():
    parser = argparse.ArgumentParser(description='Non-destructive multi-hypothesis FinFET tuning ablation.')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--raw-cache-dir', default='evaluation/tuning/candidate_cache')
    parser.add_argument('--output-dir', default='evaluation/tuning/final_multi_hypothesis')
    parser.add_argument('--g2-breakdown', default='evaluation/diagnostics/other_failure_breakdown.csv')
    parser.add_argument('--tuning-layout-count', type=int, default=12)
    parser.add_argument('--thresholds', nargs='+', type=float, default=[5., 10., 15.])
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int)
    args = parser.parse_args()
    dataset = SEMLocalizationDataset(args.dataset, LocalizationConfig())
    layouts = sorted({record['source_layout'] for record in dataset.records})
    records = [record for record in dataset.records if record['source_layout'] in layouts[:args.tuning_layout_count]]
    output = Path(args.output_dir); sample_dir = output / 'sample_cache'; sample_dir.mkdir(parents=True, exist_ok=True)
    g2_old = {}
    if Path(args.g2_breakdown).exists():
        with Path(args.g2_breakdown).open(newline='', encoding='utf-8') as file:
            g2_old = {row['sample_id']: row for row in csv.DictReader(file) if row['new_category'].startswith('G2.')}
    matcher = ClassicalSEMLocalizer(ClassicalMatcherConfig(cheap_rerank_top_k=30, verify_top_k=30))
    selected_records = records[args.start:args.stop]
    for record in tqdm(selected_records, desc='Expanding multi-hypotheses', unit='sample', dynamic_ncols=True):
        sample_path = sample_dir / f"{record['sample_id']}.json"
        existing = json.loads(sample_path.read_text(encoding='utf-8')) if sample_path.exists() else {'record': record, 'runs': {}}
        missing = [threshold for threshold in args.thresholds if str(threshold) not in existing['runs']]
        if not missing: continue
        raw = json.loads((Path(args.raw_cache_dir) / f"{record['sample_id']}.json").read_text(encoding='utf-8'))['raw_fused'][:ROOT_LIMIT]
        search = _gray(cv2.imread(str(dataset.root / record['search']), cv2.IMREAD_GRAYSCALE))
        reference = _gray(cv2.imread(str(dataset.root / record['reference']), cv2.IMREAD_GRAYSCALE))
        if reference.shape == (1000, 1000): reference = cv2.resize(reference, (100, 100), interpolation=cv2.INTER_AREA)
        search_repr = _representations(search, matcher.config)
        roots, cheap_by_id = _prepare(record, raw, matcher, search_repr, reference)
        verify_cache, phase_cache, ecc_cache = {}, {}, {}
        for threshold in missing:
            result, trace, events = _run_threshold(record, roots, cheap_by_id, matcher, search_repr, reference, threshold,
                                                   verify_cache, phase_cache, ecc_cache)
            existing['runs'][str(threshold)] = {'result': result, 'trace': trace, 'rotation_events': events}
        sample_path.write_text(json.dumps(_json(existing)), encoding='utf-8')
    pending = [record['sample_id'] for record in records if not (sample_dir / f"{record['sample_id']}.json").exists()
               or any(str(threshold) not in json.loads((sample_dir / f"{record['sample_id']}.json").read_text(encoding='utf-8'))['runs'] for threshold in args.thresholds)]
    if pending:
        print(json.dumps({'cached_samples': len(records) - len(pending), 'pending_samples': len(pending), 'pending_ids': pending}, indent=2)); return
    all_results, all_traces, all_g2 = defaultdict(list), [], []
    for record in records:
        payload = json.loads((sample_dir / f"{record['sample_id']}.json").read_text(encoding='utf-8'))
        for threshold in args.thresholds:
            run = payload['runs'][str(threshold)]; result = run['result']; all_results[threshold].append(result)
            all_traces.extend([{**item, 'sample_id': record['sample_id'], 'source_layout': record['source_layout'], 'threshold_px': threshold,
                                'center_x': item['x'] + 50, 'center_y': item['y'] + 50} for item in run['trace']])
            if record['sample_id'] in g2_old: all_g2.append(_g2_rows(record, result, run['rotation_events'], g2_old[record['sample_id']]))
    baseline_old = next(row for row in csv.DictReader((Path('evaluation/tuning') / 'shortlist_ablation.csv').open(newline='', encoding='utf-8')) if int(row['shortlist_size']) == 30)
    baseline_runtime = float(baseline_old['runtime_ms_per_sample'])
    baseline = _baseline_metrics(dataset, records, args.raw_cache_dir, matcher, baseline_runtime)
    ablation = [{'configuration': 'BASELINE_TOP30', 'split_threshold_px': None, **baseline}]
    for threshold in args.thresholds: ablation.append({'configuration': f'MH-{int(threshold)}', 'split_threshold_px': threshold, **_metrics(all_results[threshold])})
    candidates = ablation[1:]
    best = max(candidates, key=lambda row: (row['accuracy_at_5px'], -row['error_over_50px'], -row['error_over_100px'], row['accuracy_at_2px'], -row['p95_error_px'], -row['runtime_ms_per_sample']))
    accepted = bool(best['accuracy_at_5px'] > baseline['accuracy_at_5px'] and best['error_over_50px'] <= baseline['error_over_50px'])
    reason = ('Best MH configuration exceeded Accuracy@5 without worsening >50 px errors.' if accepted
              else 'No MH configuration exceeded the 80.6% tuning Accuracy@5 baseline without worsening catastrophic errors.')
    oracle_rows = []
    for threshold in args.thresholds:
        rows = all_results[threshold]
        for stage, field in (('raw_top50', 'raw_oracle_at_5px'), ('level_a_top20', 'level_a_oracle_at_5px'),
                             ('level_b_top15', 'level_b_oracle_at_5px'), ('level_c_final', 'level_c_oracle_at_5px')):
            oracle_rows.append({'configuration': f'MH-{int(threshold)}', 'split_threshold_px': threshold, 'stage': stage,
                                'oracle_accuracy_at_5px': float(np.mean([row[field] for row in rows]))})
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'ablation.csv').open('w', newline='', encoding='utf-8') as file:
        fieldnames = list(dict.fromkeys(key for row in ablation for key in row))
        writer = csv.DictWriter(file, fieldnames=fieldnames); writer.writeheader(); writer.writerows(ablation)
    with (output / 'oracle_by_stage.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(oracle_rows[0])); writer.writeheader(); writer.writerows(oracle_rows)
    with (output / 'hypothesis_trace.csv').open('w', newline='', encoding='utf-8') as file:
        fieldnames = list(dict.fromkeys(key for row in all_traces for key in row))
        writer = csv.DictWriter(file, fieldnames=fieldnames); writer.writeheader(); writer.writerows(all_traces)
    with (output / 'g2_recovery.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(all_g2[0])); writer.writeheader(); writer.writerows(all_g2)
    best_config = {'accepted': accepted, 'split_threshold': best['split_threshold_px'], 'dedup_radius': DEDUP_RADIUS,
                   'level_a_keep': LEVEL_A_KEEP, 'level_b_keep': LEVEL_B_KEEP, 'equivalence_margin': EQUIVALENCE_MARGIN,
                   'production_change': False}
    (output / 'best_config.json').write_text(json.dumps(best_config, indent=2), encoding='utf-8')
    summary = {'accepted': accepted, 'reason': reason, 'baseline_accuracy_at_5': baseline['accuracy_at_5px'],
               'best_tuning_accuracy_at_5': best['accuracy_at_5px'], 'best_threshold': best['split_threshold_px'],
               'baseline': baseline, 'best': best, 'g2_samples': len(g2_old), 'production_change_accepted': False}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__': main()
