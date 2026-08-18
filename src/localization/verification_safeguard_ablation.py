"""Cache-only ablation for a conditional periodic-basin verification safeguard."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .classical_matcher import ClassicalMatcherConfig, ClassicalSEMLocalizer, _gray, _representations, _variant
from .coordinates import top_left_to_center
from .dataset import SEMLocalizationDataset
from .config import LocalizationConfig


def _select(matcher, candidates):
    candidates = sorted((item for item in candidates if item is not None), key=lambda item: item['score'], reverse=True)
    distinct = []
    for candidate in candidates:
        if all(np.hypot(candidate['x'] - other['x'], candidate['y'] - other['y']) >= 100 for other in distinct): distinct.append(candidate)
    distinct = distinct or candidates[:1]
    equivalent = [item for item in distinct if distinct[0]['score'] - item['score'] <= matcher.config.equivalent_score_tolerance]
    return min(equivalent, key=lambda item: np.hypot(item['x'] + 50 - 500, item['y'] + 50 - 500))


def _align(matcher, root, record, candidate, cache):
    key = (record['sample_id'], round(candidate['x'], 4), round(candidate['y'], 4), candidate['rotation'], candidate['scale'])
    if key in cache: return cache[key]
    search = _gray(cv2.imread(str(root / record['search']), cv2.IMREAD_GRAYSCALE))
    reference = _gray(cv2.imread(str(root / record['reference']), cv2.IMREAD_GRAYSCALE))
    if reference.shape == (1000, 1000): reference = cv2.resize(reference, (100, 100), interpolation=cv2.INTER_AREA)
    search_repr = _representations(search, matcher.config)
    x, y = round(candidate['x']), round(candidate['y'])
    ref_local = _representations(_variant(reference, candidate['rotation'], candidate['scale']), matcher.config)['local']
    dx, dy, _, _ = matcher._align(search_repr['local'][y:y + 100, x:x + 100], ref_local)
    cache[key] = top_left_to_center(candidate['x'] + dx, candidate['y'] + dy)
    return cache[key]


def _metrics(errors, interventions):
    errors = np.asarray(errors, dtype=float)
    return {'samples': len(errors), **{f'accuracy_at_{value}px': float(np.mean(errors <= value)) for value in (1, 2, 5, 10)},
            'median_error_px': float(np.median(errors)), 'mean_error_px': float(np.mean(errors)), 'p95_error_px': float(np.percentile(errors, 95)),
            **{f'error_over_{value}px': float(np.mean(errors > value)) for value in (20, 50, 100)},
            'mean_intervened_candidates_per_sample': float(np.mean(interventions))}


def main():
    parser = argparse.ArgumentParser(description='Test only a verified-candidate basin-jump safeguard from cache.')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--wide-cache-dir', default='evaluation/tuning/candidate_cache')
    parser.add_argument('--local-cache-dir', default='evaluation/tuning/verification_window/candidate_evidence_cache')
    parser.add_argument('--output-dir', default='evaluation/tuning/verification_safeguard')
    parser.add_argument('--tuning-layout-count', type=int, default=12)
    parser.add_argument('--max-shifts', nargs='+', type=float, default=[12., 16., 20., 24., 32., 40.])
    args = parser.parse_args()
    dataset = SEMLocalizationDataset(args.dataset, LocalizationConfig())
    layouts = sorted({record['source_layout'] for record in dataset.records})
    records = [record for record in dataset.records if record['source_layout'] in layouts[:args.tuning_layout_count]]
    matcher = ClassicalSEMLocalizer(ClassicalMatcherConfig(cheap_rerank_top_k=30, verify_top_k=30))
    alignment_cache = {}; rows = []
    for shift in [None, *args.max_shifts]:
        errors, interventions = [], []
        for record in records:
            wide = json.loads((Path(args.wide_cache_dir) / f"{record['sample_id']}.json").read_text(encoding='utf-8'))['verified'][:30]
            local = json.loads((Path(args.local_cache_dir) / f"{record['sample_id']}.json").read_text(encoding='utf-8'))
            cheap = local['cheap']; narrow = local['verified_by_window']['20']
            if not (len(wide) == len(cheap) == len(narrow)):
                raise RuntimeError(f'Cache candidate identity mismatch for {record["sample_id"]}')
            candidates, count = [], 0
            for input_candidate, wide_candidate, narrow_candidate in zip(cheap, wide, narrow):
                candidate = wide_candidate
                if shift is not None and wide_candidate is not None and narrow_candidate is not None:
                    jump = float(np.hypot(wide_candidate['x'] - input_candidate['x'], wide_candidate['y'] - input_candidate['y']))
                    if jump > shift:
                        candidate = narrow_candidate; count += 1
                candidates.append(candidate)
            best = _select(matcher, candidates)
            center = _align(matcher, dataset.root, record, best, alignment_cache)
            gt = (float(record['center_x']), float(record['center_y']))
            errors.append(float(np.hypot(center[0] - gt[0], center[1] - gt[1])))
            interventions.append(count)
        rows.append({'max_verification_shift_px': 'baseline' if shift is None else shift, **_metrics(errors, interventions)})
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    with (output / 'verification_safeguard_ablation.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (output / 'verification_safeguard_ablation_summary.json').write_text(json.dumps({'samples': len(records), 'results': rows}, indent=2), encoding='utf-8')
    print(json.dumps({'samples': len(records), 'results': rows}, indent=2))


if __name__ == '__main__':
    main()
