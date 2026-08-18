"""Tune the local verification window from cached pre-verification candidates.

The coarse candidates and cheap Top-30 ranking are cached once per tuning
sample.  Only rotation/scale verification is repeated for each requested
window, which directly tests the observed G2 periodic-basin failure.
"""
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
    if isinstance(value, dict): return {str(key): _json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [_json(item) for item in value]
    if isinstance(value, np.generic): return value.item()
    return value


def _select(matcher, verified):
    candidates = sorted((item for item in verified if item is not None), key=lambda item: item['score'], reverse=True)
    distinct = []
    for candidate in candidates:
        if all(np.hypot(candidate['x'] - other['x'], candidate['y'] - other['y']) >= 100 for other in distinct):
            distinct.append(candidate)
    distinct = distinct or candidates[:1]
    best_score = distinct[0]['score']
    equivalent = [item for item in distinct if best_score - item['score'] <= matcher.config.equivalent_score_tolerance]
    return min(equivalent, key=lambda item: np.hypot(item['x'] + 50 - 500, item['y'] + 50 - 500))


def _align(matcher, root, record, candidate):
    search = _gray(cv2.imread(str(root / record['search']), cv2.IMREAD_GRAYSCALE))
    reference = _gray(cv2.imread(str(root / record['reference']), cv2.IMREAD_GRAYSCALE))
    if reference.shape == (1000, 1000): reference = cv2.resize(reference, (100, 100), interpolation=cv2.INTER_AREA)
    search_repr = _representations(search, matcher.config)
    x, y = round(candidate['x']), round(candidate['y'])
    local_ref = _representations(_variant(reference, candidate['rotation'], candidate['scale']), matcher.config)['local']
    dx, dy, _, _ = matcher._align(search_repr['local'][y:y + 100, x:x + 100], local_ref)
    return top_left_to_center(candidate['x'] + dx, candidate['y'] + dy)


def _metrics(rows):
    errors = np.asarray([row['error_px'] for row in rows], dtype=float)
    return {'samples': len(rows), **{f'accuracy_at_{value}px': float(np.mean(errors <= value)) for value in (1, 2, 5, 10)},
            'median_error_px': float(np.median(errors)), 'mean_error_px': float(np.mean(errors)),
            'p95_error_px': float(np.percentile(errors, 95)), **{f'error_over_{value}px': float(np.mean(errors > value)) for value in (20, 50, 100)},
            'verified_gt_candidate_recall_at_5px': float(np.mean([row['verified_gt_candidate_at_5px'] for row in rows])),
            'runtime_ms_per_sample': float(np.mean([row['runtime_ms'] for row in rows]))}


def _build_payload(dataset, record, cache_path, windows):
    payload = json.loads(cache_path.read_text(encoding='utf-8')) if cache_path.exists() else {'record': record, 'verified_by_window': {}, 'verify_runtime_ms_by_window': {}}
    missing = [window for window in windows if str(window) not in payload['verified_by_window']]
    if not missing: return payload
    base_config = ClassicalMatcherConfig(cheap_rerank_top_k=30, verify_top_k=30)
    matcher = ClassicalSEMLocalizer(base_config)
    search = _gray(cv2.imread(str(dataset.root / record['search']), cv2.IMREAD_GRAYSCALE))
    reference = _gray(cv2.imread(str(dataset.root / record['reference']), cv2.IMREAD_GRAYSCALE))
    if reference.shape == (1000, 1000): reference = cv2.resize(reference, (100, 100), interpolation=cv2.INTER_AREA)
    started = time.perf_counter()
    search_repr = _representations(search, matcher.config)
    if 'cheap' not in payload:
        coarse = matcher._coarse(search, reference, limit=matcher.config.top_k)
        payload['cheap'] = matcher._cheap_rerank(search_repr, reference, coarse)[:30]
        payload['base_runtime_ms'] = (time.perf_counter() - started) * 1000
    for window in missing:
        window_matcher = ClassicalSEMLocalizer(ClassicalMatcherConfig(refinement_window=window, cheap_rerank_top_k=30, verify_top_k=30))
        verify_started = time.perf_counter()
        verified = [window_matcher._verify(search_repr, reference, candidate) for candidate in payload['cheap']]
        payload['verified_by_window'][str(window)] = verified
        payload['verify_runtime_ms_by_window'][str(window)] = (time.perf_counter() - verify_started) * 1000
    cache_path.write_text(json.dumps(_json(payload)), encoding='utf-8')
    return payload


def main():
    parser = argparse.ArgumentParser(description='Ablate only local verification-window width on tuning layouts.')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--output-dir', default='evaluation/tuning/verification_window')
    parser.add_argument('--tuning-layout-count', type=int, default=12)
    parser.add_argument('--windows', nargs='+', type=int, default=[20, 32])
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--stop', type=int)
    args = parser.parse_args()
    dataset = SEMLocalizationDataset(args.dataset, LocalizationConfig())
    layouts = sorted({record['source_layout'] for record in dataset.records})
    records = [record for record in dataset.records if record['source_layout'] in layouts[:args.tuning_layout_count]]
    output = Path(args.output_dir); cache_dir = output / 'candidate_evidence_cache'; cache_dir.mkdir(parents=True, exist_ok=True)
    selected = records[args.start:args.stop]
    for record in tqdm(selected, desc='Caching verification windows', unit='sample', dynamic_ncols=True):
        _build_payload(dataset, record, cache_dir / f"{record['sample_id']}.json", args.windows)
    missing = [record['sample_id'] for record in records if not (cache_dir / f"{record['sample_id']}.json").exists()
               or any(str(window) not in json.loads((cache_dir / f"{record['sample_id']}.json").read_text(encoding='utf-8'))['verified_by_window'] for window in args.windows)]
    if missing:
        print(json.dumps({'cached_samples': len(records) - len(missing), 'pending_samples': len(missing), 'pending_ids': missing}, indent=2))
        return
    all_rows = []
    for window in args.windows:
        matcher = ClassicalSEMLocalizer(ClassicalMatcherConfig(refinement_window=window, cheap_rerank_top_k=30, verify_top_k=30))
        rows = []
        for record in records:
            payload = json.loads((cache_dir / f"{record['sample_id']}.json").read_text(encoding='utf-8'))
            verified = payload['verified_by_window'][str(window)]
            best = _select(matcher, verified)
            align_started = time.perf_counter(); center = _align(matcher, dataset.root, record, best); align_ms = (time.perf_counter() - align_started) * 1000
            gt = (float(record['center_x']), float(record['center_y']))
            rows.append({'window_px': window, 'sample_id': record['sample_id'],
                         'error_px': float(np.hypot(center[0] - gt[0], center[1] - gt[1])),
                         'verified_gt_candidate_at_5px': any(np.hypot(candidate['x'] + 50 - gt[0], candidate['y'] + 50 - gt[1]) <= 5. for candidate in verified if candidate is not None),
                         'runtime_ms': float(payload['base_runtime_ms']) + float(payload['verify_runtime_ms_by_window'][str(window)]) + align_ms})
        all_rows.append({'window_px': window, **_metrics(rows)})
    with (output / 'verification_window_ablation.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    (output / 'verification_window_ablation_summary.json').write_text(json.dumps({'samples': len(records), 'windows': args.windows, 'results': all_rows}, indent=2), encoding='utf-8')
    print(json.dumps({'samples': len(records), 'results': all_rows}, indent=2))


if __name__ == '__main__':
    main()
