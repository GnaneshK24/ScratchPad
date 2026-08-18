"""Geometry and candidate-recall diagnostics for the classical localizer.

Run this before tuning.  It measures whether the recorded centre is a local
structural match and whether candidate generation retains it at all.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .classical_matcher import ClassicalSEMLocalizer, _representations
from .config import LocalizationConfig
from .coordinates import search_center_to_top_left
from .dataset import SEMLocalizationDataset


def _recall(candidates, center, k, tolerance):
    return any(np.hypot(c['center_x'] - center[0], c['center_y'] - center[1]) <= tolerance
               for c in candidates[:k])


def _template(reference):
    if reference.shape == (1000, 1000):
        return cv2.resize(reference, (100, 100), interpolation=cv2.INTER_AREA)
    return reference


def _dominant_period(score_map, axis):
    """Estimate a response-map repeat distance for ambiguity diagnostics only."""
    profile = np.mean(score_map, axis=axis).astype(np.float64)
    profile -= np.mean(profile)
    if np.allclose(profile, 0.):
        return None
    autocorrelation = np.correlate(profile, profile, mode='full')[len(profile) - 1:]
    minimum_lag = max(8, len(profile) // 100)
    maximum_lag = max(minimum_lag + 1, len(profile) // 2)
    if maximum_lag <= minimum_lag:
        return None
    lag = minimum_lag + int(np.argmax(autocorrelation[minimum_lag:maximum_lag]))
    return int(lag)


def main():
    parser = argparse.ArgumentParser(description='Validate geometry and candidate recall before matcher tuning.')
    parser.add_argument('--dataset-dir', required=True)
    parser.add_argument('--limit', type=int, default=50)
    parser.add_argument('--noise-modes', nargs='+')
    parser.add_argument('--neighborhood', type=int, default=8)
    parser.add_argument('--candidate-limit', type=int, default=200,
                        help='Maximum independent candidates for the recall-only experiment.')
    parser.add_argument('--output')
    parser.add_argument('--output-dir', help='Directory for candidate_rank.csv and candidate_recall_summary.json')
    args = parser.parse_args()

    dataset = SEMLocalizationDataset(args.dataset_dir, LocalizationConfig())
    records = dataset.records
    if args.noise_modes:
        allowed = set(args.noise_modes)
        records = [r for r in records if r.get('noise_mode') in allowed]
    records = records[:args.limit]
    if not records:
        raise SystemExit('No records match the requested diagnostic subset.')

    matcher = ClassicalSEMLocalizer()
    representations = ('intensity', 'local', 'gradient', 'gx', 'gy', 'bandpass', 'edge', 'structure', 'distance')
    local_offsets = {name: [] for name in representations}
    local_scores = {name: [] for name in representations}
    gt_ranks = {name: [] for name in (*representations, 'fused')}
    periodic_samples = 0
    gt_equivalent_to_best = 0
    gt_is_centre_preferred = 0
    phase_responses = []
    candidate_rows = []
    rank_rows = []
    ambiguity_rows = []
    rank_candidates = {name: [] for name in (*representations, 'fused', 'independent_coarse')}

    for record in tqdm(records, desc='Diagnosing', unit='sample', dynamic_ncols=True):
        search = cv2.imread(str(dataset.root / record['search']), cv2.IMREAD_GRAYSCALE)
        reference = _template(cv2.imread(str(dataset.root / record['reference']), cv2.IMREAD_GRAYSCALE))
        gx, gy = float(record['center_x']), float(record['center_y'])
        tx, ty = search_center_to_top_left(gx, gy)
        tx, ty = int(round(tx)), int(round(ty))
        search_repr, reference_repr = _representations(search, matcher.config), _representations(reference, matcher.config)
        all_scores = {}
        for name in representations:
            scores = cv2.matchTemplate(search_repr[name], reference_repr[name], cv2.TM_CCOEFF_NORMED)
            all_scores[name] = scores
            x0, x1 = max(0, tx - args.neighborhood), min(scores.shape[1], tx + args.neighborhood + 1)
            y0, y1 = max(0, ty - args.neighborhood), min(scores.shape[0], ty + args.neighborhood + 1)
            _, score, _, (x, y) = cv2.minMaxLoc(scores[y0:y1, x0:x1])
            x, y = x + x0, y + y0
            local_offsets[name].append(float(np.hypot(x - tx, y - ty)))
            local_scores[name].append(float(score))
            gt_score = float(scores[ty, tx])
            gt_ranks[name].append(int(np.count_nonzero(scores > gt_score)) + 1)
            peaks = matcher._peaks(scores, args.candidate_limit, matcher.config.nms_radius)
            candidates = [{'center_x': x + 50., 'center_y': y + 50., 'score': score} for x, y, score in peaks]
            rank_candidates[name].append(candidates)
            rank_rows.append({'sample_id': record['sample_id'], 'representation': name, 'gt_x': gx, 'gt_y': gy,
                              'gt_score': gt_score, 'best_score': float(peaks[0][2]), 'gt_rank': gt_ranks[name][-1],
                              'top1_error_px': float(np.hypot(candidates[0]['center_x'] - gx, candidates[0]['center_y'] - gy))})
        fused = (matcher.config.intensity_weight * all_scores['intensity']
                 + matcher.config.contrast_weight * all_scores['local']
                 + matcher.config.gradient_weight * all_scores['gradient']
                 + matcher.config.gx_weight * all_scores['gx']
                 + matcher.config.gy_weight * all_scores['gy']
                 + matcher.config.bandpass_weight * all_scores['bandpass']
                 + matcher.config.edge_weight * all_scores['edge']
                 + matcher.config.local_structure_weight * all_scores['structure']
                 + matcher.config.shape_weight * all_scores['distance'])
        gt_ranks['fused'].append(int(np.count_nonzero(fused > fused[ty, tx])) + 1)
        fused_peaks = matcher._peaks(fused, args.candidate_limit, matcher.config.nms_radius)
        fused_candidates = [{'center_x': x + 50., 'center_y': y + 50., 'score': score} for x, y, score in fused_peaks]
        rank_candidates['fused'].append(fused_candidates)
        rank_rows.append({'sample_id': record['sample_id'], 'representation': 'fused', 'gt_x': gx, 'gt_y': gy,
                          'gt_score': float(fused[ty, tx]), 'best_score': float(fused_peaks[0][2]),
                          'gt_rank': gt_ranks['fused'][-1],
                          'top1_error_px': float(np.hypot(fused_candidates[0]['center_x'] - gx, fused_candidates[0]['center_y'] - gy))})
        # Evaluation-side analysis of the official centre tie rule.  It never
        # feeds GT into inference: it only reports whether this generator's
        # random labelled occurrence is compatible with an equivalent-match
        # interpretation of the challenge contract.
        peaks = matcher._peaks(fused, 100, matcher.config.nms_radius)
        best_score = peaks[0][2]
        equivalent = [(x + 50., y + 50.) for x, y, score in peaks
                      if best_score - score <= matcher.config.equivalent_score_tolerance]
        if len(equivalent) > 1:
            periodic_samples += 1
        if best_score - float(fused[ty, tx]) <= matcher.config.equivalent_score_tolerance:
            gt_equivalent_to_best += 1
        if equivalent:
            official = min(equivalent, key=lambda point: np.hypot(point[0] - 500., point[1] - 500.))
            if np.hypot(official[0] - (tx + 50.), official[1] - (ty + 50.)) <= 5.:
                gt_is_centre_preferred += 1
        gt_fused_score = float(fused[ty, tx])
        near_equivalent = [(x + 50., y + 50., float(score)) for x, y, score in peaks
                           if score >= gt_fused_score - matcher.config.equivalent_score_tolerance]
        distant_equivalents = [item for item in near_equivalent
                               if np.hypot(item[0] - gx, item[1] - gy) > 50.]
        ambiguity_rows.append({
            'sample_id': record['sample_id'],
            'gt_score': gt_fused_score,
            'best_score': float(best_score),
            'second_best_score': float(peaks[1][2]) if len(peaks) > 1 else float('nan'),
            'best_minus_gt_score': float(best_score - gt_fused_score),
            'near_equivalent_match_count': len(near_equivalent),
            'distant_near_equivalent_match_count': len(distant_equivalents),
            'dominant_horizontal_period_px': _dominant_period(fused, axis=0),
            'dominant_vertical_period_px': _dominant_period(fused, axis=1),
            'ambiguity_label': 'AMBIGUOUS_PERIODIC_PATTERN' if distant_equivalents else 'UNIQUE_OR_UNRESOLVED',
        })
        patch = search_repr['local'][ty:ty + 100, tx:tx + 100]
        if patch.shape == (100, 100):
            _, response = cv2.phaseCorrelate(reference_repr['local'], patch)
            phase_responses.append(float(response))
        candidates = matcher._coarse(search, reference, limit=args.candidate_limit)
        union_candidates = [{'center_x': c['x'] + 50., 'center_y': c['y'] + 50., 'score': c['coarse_score']} for c in candidates]
        candidate_rows.append(union_candidates)
        rank_candidates['independent_coarse'].append(union_candidates)
        gt_union_score = max((item['score'] for item in union_candidates
                              if np.hypot(item['center_x'] - gx, item['center_y'] - gy) <= 5), default=float('-inf'))
        rank_rows.append({'sample_id': record['sample_id'], 'representation': 'independent_coarse', 'gt_x': gx, 'gt_y': gy,
                          'gt_score': gt_union_score, 'best_score': union_candidates[0]['score'] if union_candidates else float('nan'),
                          'gt_rank': next((index + 1 for index, item in enumerate(union_candidates)
                                           if np.hypot(item['center_x'] - gx, item['center_y'] - gy) <= 5), args.candidate_limit + 1),
                          'top1_error_px': float(np.hypot(union_candidates[0]['center_x'] - gx, union_candidates[0]['center_y'] - gy))})

    geometry = {
        name: {'mean_local_offset_px': float(np.mean(local_offsets[name])),
               'p95_local_offset_px': float(np.percentile(local_offsets[name], 95)),
               'mean_local_score': float(np.mean(local_scores[name]))}
        for name in representations
    }
    # A correct centre/top-left/scale adapter puts the best structural peak in
    # the small GT neighbourhood; this does not claim that noisy images should
    # have a high absolute correlation.
    geometry['pass'] = bool(np.median(local_offsets['intensity']) <= 2.0 and
                            np.median(local_offsets['gradient']) <= 3.0)
    geometry['mean_phase_response'] = float(np.mean(phase_responses)) if phase_responses else None
    k_values = tuple(k for k in (50, 75, 100, 150, 200) if k <= args.candidate_limit)
    recalls = {f'top_{k}_within_{tolerance}px': float(np.mean([
        _recall(candidates, (float(record['center_x']), float(record['center_y'])), k, tolerance)
        for record, candidates in zip(records, candidate_rows)
    ])) for k in k_values for tolerance in (2, 5, 10, 20)}
    ranks = {name: {'median': float(np.median(values)), 'p90': float(np.percentile(values, 90))}
             for name, values in gt_ranks.items()}
    result = {'dataset': str(args.dataset_dir), 'samples': len(records), 'geometry': geometry,
              'candidate_recall': recalls, 'gt_rank': ranks,
              'official_semantics_audit': {
                  'equivalence_tolerance': matcher.config.equivalent_score_tolerance,
                  'periodic_ambiguous_samples': periodic_samples,
                  'periodic_ambiguous_rate': periodic_samples / len(records),
                  'generator_gt_equivalent_to_best_rate': gt_equivalent_to_best / len(records),
                  'generator_gt_is_official_centre_choice_rate': gt_is_centre_preferred / len(records),
              },
              'candidate_method': 'independent_representation_union_nms'}
    candidate_recall_summary = {}
    for name, all_candidates in rank_candidates.items():
        candidate_recall_summary[name] = {f'candidate_recall_at_{k}': float(np.mean([
            _recall(candidates, (float(record['center_x']), float(record['center_y'])), k, 5)
            for record, candidates in zip(records, all_candidates)]))
            for k in (1, 5, 10, 20, 50, 100) if k <= args.candidate_limit}
    result['candidate_rank_recall'] = candidate_recall_summary
    print(json.dumps(result, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
    if args.output_dir:
        output_dir = Path(args.output_dir) / 'diagnostics'
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / 'candidate_rank.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list(rank_rows[0]))
            writer.writeheader(); writer.writerows(rank_rows)
        (output_dir / 'candidate_recall_summary.json').write_text(json.dumps(candidate_recall_summary, indent=2), encoding='utf-8')
        with (output_dir / 'ambiguity_report.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list(ambiguity_rows[0]))
            writer.writeheader(); writer.writerows(ambiguity_rows)


if __name__ == '__main__':
    main()
