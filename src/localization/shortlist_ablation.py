"""Cached, source-layout-safe shortlist ablation for the classical matcher."""
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


def _json(value):
    if isinstance(value, dict): return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_json(v) for v in value]
    if isinstance(value, np.generic): return value.item()
    return value


def _center(candidate):
    return top_left_to_center(float(candidate['x']), float(candidate['y']))


def _select(matcher, candidates):
    verified = [candidate for candidate in candidates if candidate is not None]
    if not verified:
        return None, [], []
    verified.sort(key=lambda candidate: candidate['score'], reverse=True)
    distinct = []
    for candidate in verified:
        if all(np.hypot(candidate['x'] - other['x'], candidate['y'] - other['y']) >= 100 for other in distinct):
            distinct.append(candidate)
    distinct = distinct or verified[:1]
    best_score = distinct[0]['score']
    equivalent = [candidate for candidate in distinct if best_score - candidate['score'] <= matcher.config.equivalent_score_tolerance]
    best = min(equivalent, key=lambda candidate: np.hypot(candidate['x'] + 50 - 500, candidate['y'] + 50 - 500))
    return best, distinct, equivalent


def _align(matcher, record, root, candidate):
    search = _gray(cv2.imread(str(root / record['search']), cv2.IMREAD_GRAYSCALE))
    reference = _gray(cv2.imread(str(root / record['reference']), cv2.IMREAD_GRAYSCALE))
    if reference.shape == (1000, 1000): reference = cv2.resize(reference, (100, 100), interpolation=cv2.INTER_AREA)
    search_repr = _representations(search, matcher.config)
    x, y = round(candidate['x']), round(candidate['y'])
    ref = _representations(_variant(reference, candidate['rotation'], candidate['scale']), matcher.config)['local']
    dx, dy, _, _ = matcher._align(search_repr['local'][y:y + 100, x:x + 100], ref)
    return top_left_to_center(candidate['x'] + dx, candidate['y'] + dy)


def _metrics(rows):
    errors = np.asarray([row['error_px'] for row in rows], dtype=float)
    return {
        'samples': len(rows),
        **{f'accuracy_at_{value}px': float(np.mean(errors <= value)) for value in (1, 2, 5, 10)},
        'median_error_px': float(np.median(errors)), 'mean_error_px': float(np.mean(errors)),
        'p95_error_px': float(np.percentile(errors, 95)),
        **{f'error_over_{value}px': float(np.mean(errors > value)) for value in (20, 50, 100)},
        'runtime_ms_per_sample': float(np.mean([row['runtime_ms'] for row in rows])),
    }


def _build_cache(dataset, records, cache_dir, max_shortlist):
    matcher = ClassicalSEMLocalizer(ClassicalMatcherConfig(cheap_rerank_top_k=max_shortlist, verify_top_k=max_shortlist))
    cache_dir.mkdir(parents=True, exist_ok=True)
    for record in tqdm(records, desc='Caching verification evidence', unit='sample', dynamic_ncols=True):
        path = cache_dir / f"{record['sample_id']}.json"
        if path.exists():
            continue
        started = time.perf_counter()
        search = _gray(cv2.imread(str(dataset.root / record['search']), cv2.IMREAD_GRAYSCALE))
        reference = _gray(cv2.imread(str(dataset.root / record['reference']), cv2.IMREAD_GRAYSCALE))
        if reference.shape == (1000, 1000): reference = cv2.resize(reference, (100, 100), interpolation=cv2.INTER_AREA)
        search_repr = _representations(search, matcher.config)
        reference_repr = _representations(reference, matcher.config)
        fused = matcher._fused(search_repr, reference_repr)
        raw_fused = [{'x': float(x), 'y': float(y), 'score': float(score)}
                     for x, y, score in matcher._peaks(fused, max_shortlist, matcher.config.nms_radius)]
        coarse = matcher._coarse(search, reference, limit=matcher.config.top_k)
        cheap = matcher._cheap_rerank(search_repr, reference, coarse)[:max_shortlist]
        base_runtime_ms = (time.perf_counter() - started) * 1000
        verified, verify_runtime_ms = [], []
        for candidate in cheap:
            verify_started = time.perf_counter()
            verified.append(matcher._verify(search_repr, reference, candidate))
            verify_runtime_ms.append((time.perf_counter() - verify_started) * 1000)
        payload = {'record': record, 'raw_fused': raw_fused, 'verified': verified,
                   'base_runtime_ms': base_runtime_ms, 'verify_runtime_ms': verify_runtime_ms}
        path.write_text(json.dumps(_json(payload)), encoding='utf-8')


def _oracle(cache, gt, count):
    candidates = cache['raw_fused'][:count]
    return min(np.hypot(candidate['x'] + 50 - gt[0], candidate['y'] + 50 - gt[1]) for candidate in candidates)


def main():
    parser = argparse.ArgumentParser(description='Cache expensive FinFET verification, then measure shortlist prefixes.')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output-dir', default='evaluation/tuning')
    parser.add_argument('--tuning-layout-count', type=int, default=12)
    parser.add_argument('--shortlists', nargs='+', type=int, default=[5, 10, 20, 30, 40, 50])
    parser.add_argument('--rebuild-cache', action='store_true')
    args = parser.parse_args()
    max_shortlist = max(args.shortlists)
    dataset = SEMLocalizationDataset(args.dataset, LocalizationConfig())
    layouts = sorted({record['source_layout'] for record in dataset.records})
    if len(layouts) <= args.tuning_layout_count: raise SystemExit('Need at least one held-out source layout.')
    tuning_layouts, heldout_layouts = layouts[:args.tuning_layout_count], layouts[args.tuning_layout_count:]
    records = [record for record in dataset.records if record['source_layout'] in tuning_layouts]
    output = Path(args.output_dir); cache_dir = output / 'candidate_cache'
    if args.rebuild_cache and cache_dir.exists():
        for path in cache_dir.glob('*.json'): path.unlink()
    _build_cache(dataset, records, cache_dir, max_shortlist)
    matcher = ClassicalSEMLocalizer(ClassicalMatcherConfig(cheap_rerank_top_k=max_shortlist, verify_top_k=max_shortlist))
    caches = [json.loads((cache_dir / f"{record['sample_id']}.json").read_text()) for record in records]
    ablation_rows = []
    for shortlist in args.shortlists:
        results = []
        for record, cache in tqdm(zip(records, caches), total=len(records), desc=f'Ranking prefix {shortlist}', unit='sample', dynamic_ncols=True):
            started = time.perf_counter(); best, _, _ = _select(matcher, cache['verified'][:shortlist])
            if best is None: continue
            center = _align(matcher, record, dataset.root, best)
            gt = (float(record['center_x']), float(record['center_y']))
            alignment_ms = (time.perf_counter() - started) * 1000
            results.append({'error_px': float(np.hypot(center[0] - gt[0], center[1] - gt[1])),
                            'runtime_ms': float(cache['base_runtime_ms']) + sum(cache['verify_runtime_ms'][:shortlist]) + alignment_ms})
        ablation_rows.append({'shortlist_size': shortlist, **_metrics(results)})
    with (output / 'shortlist_ablation.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(ablation_rows[0]))
        writer.writeheader(); writer.writerows(ablation_rows)
    best = max(ablation_rows, key=lambda row: (row['accuracy_at_5px'], -row['error_over_50px'],
                                                -row['error_over_100px'], row['accuracy_at_2px'],
                                                -row['median_error_px'], -row['runtime_ms_per_sample']))
    best_config = dict(matcher.config.__dict__)
    best_config.update({'cheap_rerank_top_k': int(best['shortlist_size']),
                        'verify_top_k': int(best['shortlist_size']),
                        'selection_metrics': best,
                        'tuning_metadata': {'tuning_layouts': tuning_layouts, 'heldout_layouts': heldout_layouts,
                                            'objective': 'accuracy@5, error>50, error>100, accuracy@2, median, runtime'}})
    (output / 'best_config.json').write_text(json.dumps(best_config, indent=2), encoding='utf-8')
    oracle = {}
    for count in (5, 10, 20, 50):
        errors = [_oracle(cache, (float(record['center_x']), float(record['center_y'])), count)
                  for record, cache in zip(records, caches)]
        oracle[f'oracle_at_5px_using_top_{count}'] = float(np.mean(np.asarray(errors) <= 5.))
    result = {'tuning_layouts': tuning_layouts, 'heldout_layouts': heldout_layouts,
              'samples_tuning': len(records), 'samples_heldout': len(dataset.records) - len(records),
              'oracle': oracle, 'ablation': ablation_rows, 'best_config': best_config}
    (output / 'shortlist_ablation_summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
