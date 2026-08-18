"""GT-only lifecycle trace for failed classical FinFET localizations."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .classical_matcher import ClassicalSEMLocalizer, _gray, _representations, _variant
from .coordinates import top_left_to_center
from .dataset import SEMLocalizationDataset
from .config import LocalizationConfig


def _center(candidate):
    if 'center_x' in candidate:
        return float(candidate['center_x']), float(candidate['center_y'])
    return top_left_to_center(float(candidate['x']), float(candidate['y']))


def _nearest(candidates, target):
    if not candidates:
        return {'rank': None, 'distance_px': None, 'candidate': None}
    distances = [float(np.hypot(_center(candidate)[0] - target[0], _center(candidate)[1] - target[1]))
                 for candidate in candidates]
    index = int(np.argmin(distances))
    return {'rank': index + 1, 'distance_px': distances[index], 'candidate': candidates[index]}


def _rank_or_none(candidates, target, tolerance=5.):
    nearest = _nearest(candidates, target)
    return nearest['rank'] if nearest['distance_px'] is not None and nearest['distance_px'] <= tolerance else None


def _trace_pair(matcher, search_image, reference_image, gt):
    search, reference = _gray(search_image), _gray(reference_image)
    if reference.shape == (1000, 1000):
        reference = cv2.resize(reference, (100, 100), interpolation=cv2.INTER_AREA)
    search_repr = _representations(search, matcher.config)
    reference_repr = _representations(reference, matcher.config)
    fused = matcher._fused(search_repr, reference_repr)
    raw_fused = [{'x': float(x), 'y': float(y), 'score': float(score)}
                 for x, y, score in matcher._peaks(fused, 100, matcher.config.nms_radius)]

    coarse_trace = {}
    nms = matcher._coarse(search, reference, limit=matcher.config.top_k, trace=coarse_trace)
    cheap_all = matcher._cheap_rerank(search_repr, reference, nms)
    shortlist = cheap_all[:matcher.config.cheap_rerank_top_k]
    verification_attempts = []
    verified = []
    for candidate in shortlist:
        verified_candidate = matcher._verify(search_repr, reference, candidate)
        verification_attempts.append({
            'input': dict(candidate),
            'output': None if verified_candidate is None else dict(verified_candidate),
        })
        if verified_candidate is not None:
            verified.append(verified_candidate)
    verified.sort(key=lambda item: item['score'], reverse=True)
    distinct = []
    for candidate in verified:
        if all(np.hypot(candidate['x'] - other['x'], candidate['y'] - other['y']) >= 100 for other in distinct):
            distinct.append(candidate)
    distinct = distinct or verified[:1]
    appearance_best = distinct[0]['score']
    equivalent = [candidate for candidate in distinct
                  if appearance_best - candidate['score'] <= matcher.config.equivalent_score_tolerance]
    selected = (min(equivalent, key=lambda candidate: np.hypot(candidate['x'] + 50 - 500, candidate['y'] + 50 - 500))
                if matcher.config.use_center_tie_break else distinct[0])
    pre_align_center = _center(selected)
    x, y = round(selected['x']), round(selected['y'])
    ref_local = _representations(_variant(reference, selected['rotation'], selected['scale']), matcher.config)['local']
    dx, dy, ecc, phase = matcher._align(search_repr['local'][y:y + 100, x:x + 100], ref_local)
    final_center = top_left_to_center(selected['x'] + dx, selected['y'] + dy)
    return {
        'raw_fused': raw_fused,
        'candidate_union_raw': coarse_trace['candidate_union_raw'],
        'rank_after_nms': nms,
        'cheap_all': cheap_all,
        'shortlist': shortlist,
        'verification_attempts': verification_attempts,
        'verified': verified,
        'distinct': distinct,
        'equivalent': equivalent,
        'selected': selected,
        'pre_align_center': pre_align_center,
        'final_center': final_center,
        'phase_response': phase,
        'ecc_score': ecc,
    }


def _record(role, target, trace, selected_by_center_rule):
    nearest_equivalent = _nearest(trace['equivalent'], target)
    return {
        'role': role,
        'raw_fused_rank': _rank_or_none(trace['raw_fused'], target),
        'candidate_union_rank': _rank_or_none(trace['candidate_union_raw'], target, 10.),
        'rank_after_nms': _rank_or_none(trace['rank_after_nms'], target, 10.),
        'rank_after_coarse_verification': _rank_or_none(trace['cheap_all'], target),
        'shortlist_rank': _rank_or_none(trace['shortlist'], target),
        'rank_after_rotation_scale': _rank_or_none(trace['verified'], target),
        'rank_after_phase': 1 if selected_by_center_rule else None,
        'rank_after_ecc': 1 if selected_by_center_rule else None,
        'final_score_rank': _rank_or_none(trace['distinct'], target),
        'equivalent_group_membership': nearest_equivalent['rank'] is not None and nearest_equivalent['distance_px'] <= 5.,
        'selected_by_center_rule': selected_by_center_rule,
        'distance_to_search_center': float(np.hypot(target[0] - 500., target[1] - 500.)),
    }


def _classify(gt, trace):
    raw = _rank_or_none(trace['raw_fused'], gt)
    union = _rank_or_none(trace['candidate_union_raw'], gt, 10.)
    nms = _rank_or_none(trace['rank_after_nms'], gt, 10.)
    shortlist = _rank_or_none(trace['shortlist'], gt)
    verified = _rank_or_none(trace['verified'], gt)
    equivalent = _rank_or_none(trace['equivalent'], gt)
    pre_align_error = float(np.hypot(trace['pre_align_center'][0] - gt[0], trace['pre_align_center'][1] - gt[1]))
    final_error = float(np.hypot(trace['final_center'][0] - gt[0], trace['final_center'][1] - gt[1]))
    if raw and union and not nms:
        return 'A. GT candidate removed by NMS'
    if nms and not shortlist:
        return 'B. GT candidate removed by shortlist limit'
    if verified:
        if pre_align_error <= 5. and final_error > 5.:
            return 'F. refinement moves initially-correct candidate away'
        if not equivalent:
            return 'D. GT candidate incorrectly excluded from equivalent-match group'
        if pre_align_error > 5.:
            return 'E. centre tie-break selects wrong candidate because equivalence grouping is wrong'
        return 'C. GT candidate survives but verification ranks wrong candidate higher'
    return 'G. other'


def main():
    parser = argparse.ArgumentParser(description='Trace failed FinFET localization candidates without altering inference.')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--predictions', required=True, help='Baseline predictions.csv used only to select failures.')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--failure-threshold', type=float, default=5.)
    args = parser.parse_args()
    with Path(args.predictions).open(newline='') as file:
        failure_ids = {row['sample_id'] for row in csv.DictReader(file) if float(row['error_px']) > args.failure_threshold}
    dataset = SEMLocalizationDataset(args.dataset, LocalizationConfig())
    records = [record for record in dataset.records if record['sample_id'] in failure_ids]
    matcher = ClassicalSEMLocalizer()
    rows = []
    categories = Counter()
    for record in tqdm(records, desc='Tracing failed candidates', unit='sample', dynamic_ncols=True):
        search = cv2.imread(str(dataset.root / record['search']), cv2.IMREAD_GRAYSCALE)
        reference = cv2.imread(str(dataset.root / record['reference']), cv2.IMREAD_GRAYSCALE)
        gt = (float(record['center_x']), float(record['center_y']))
        trace = _trace_pair(matcher, search, reference, gt)
        category = _classify(gt, trace)
        categories[category] += 1
        final_error = float(np.hypot(trace['final_center'][0] - gt[0], trace['final_center'][1] - gt[1]))
        gt_row = _record('gt_nearest', gt, trace, np.hypot(trace['pre_align_center'][0] - gt[0], trace['pre_align_center'][1] - gt[1]) <= 5.)
        winner_row = _record('final_wrong_winner', trace['pre_align_center'], trace, True)
        for row in (gt_row, winner_row):
            rows.append({'sample_id': record['sample_id'], 'source_layout': record.get('source_layout'),
                         'failure_category': category, 'final_error_px': final_error,
                         'phase_response': trace['phase_response'], 'ecc_score': trace['ecc_score'], **row})
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'candidate_lifecycle_trace.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = {'failed_samples': len(records), 'failure_categories': dict(categories)}
    (output / 'failure_category_summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
