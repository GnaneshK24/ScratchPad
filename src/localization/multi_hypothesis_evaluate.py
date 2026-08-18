"""One-shot held-out evaluator for a frozen multi-hypothesis configuration."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .classical_matcher import ClassicalMatcherConfig, ClassicalSEMLocalizer, _gray, _representations
from .config import LocalizationConfig
from .dataset import SEMLocalizationDataset
from .multi_hypothesis_ablation import (_json, _metrics, _prepare, _run_threshold)


def main():
    parser = argparse.ArgumentParser(description='Evaluate one frozen non-destructive MH configuration.')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--source-layouts', nargs='+', required=True)
    parser.add_argument('--split-threshold', type=float, required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    dataset = SEMLocalizationDataset(args.dataset, LocalizationConfig())
    allowed = set(args.source_layouts)
    records = [record for record in dataset.records if record['source_layout'] in allowed]
    if not records: raise SystemExit('No records match --source-layouts.')
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    matcher = ClassicalSEMLocalizer(ClassicalMatcherConfig(cheap_rerank_top_k=30, verify_top_k=30))
    rows, traces = [], []
    for record in tqdm(records, desc='Evaluating frozen MH', unit='sample', dynamic_ncols=True):
        started = time.perf_counter()
        search = _gray(cv2.imread(str(dataset.root / record['search']), cv2.IMREAD_GRAYSCALE))
        reference = _gray(cv2.imread(str(dataset.root / record['reference']), cv2.IMREAD_GRAYSCALE))
        if reference.shape == (1000, 1000): reference = cv2.resize(reference, (100, 100), interpolation=cv2.INTER_AREA)
        search_repr = _representations(search, matcher.config)
        reference_repr = _representations(reference, matcher.config)
        fused = matcher._fused(search_repr, reference_repr)
        raw = [{'x': float(x), 'y': float(y), 'score': float(score)}
               for x, y, score in matcher._peaks(fused, 50, matcher.config.nms_radius)]
        roots, cheap_by_id = _prepare(record, raw, matcher, search_repr, reference)
        base_ms = (time.perf_counter() - started) * 1000
        result, trace, _ = _run_threshold(record, roots, cheap_by_id, matcher, search_repr, reference,
                                          args.split_threshold, {}, {}, {})
        result['runtime_ms'] += base_ms
        rows.append(result)
        traces.extend([{**item, 'sample_id': record['sample_id'], 'source_layout': record['source_layout'],
                        'center_x': item['x'] + 50, 'center_y': item['y'] + 50} for item in trace])
    metrics = _metrics(rows)
    metrics['split_threshold'] = args.split_threshold
    with (output / 'predictions.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    with (output / 'hypothesis_trace.csv').open('w', newline='', encoding='utf-8') as file:
        fieldnames = list(dict.fromkeys(key for row in traces for key in row))
        writer = csv.DictWriter(file, fieldnames=fieldnames); writer.writeheader(); writer.writerows(traces)
    (output / 'metrics.json').write_text(json.dumps(_json(metrics), indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__': main()
