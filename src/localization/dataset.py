"""Validated loader for the existing DRAM and FinFET generator formats."""
import csv
import json
import warnings
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from .coordinates import validate_center


def _read_gray(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None: raise ValueError(f"Cannot read image: {path}")
    return image


class SEMLocalizationDataset:
    """Reads existing ``annotations.json`` or ``ground_truth.csv`` datasets.

    FinFET's existing generator writes 1000px references; these are downsampled
    to the documented 100px localization contract, just as the benchmark does.
    """
    def __init__(self, dataset_dir, config, records=None):
        self.root, self.config = Path(dataset_dir), config
        self.records = records if records is not None else self._discover()
        if not self.records: raise ValueError(f"No recognized samples in {self.root}")
        self._validate_records()

    def _discover(self):
        ann = self.root / 'annotations.json'; csv_path = self.root / 'ground_truth.csv'
        if ann.exists():
            pairs = json.loads(ann.read_text()).get('pairs', [])
            return [dict(sample_id=str(p.get('pair_id', i)), reference=p['reference_path'], search=p['search_path'],
                         center_x=p['ground_truth_center']['x'], center_y=p['ground_truth_center']['y'],
                         source_layout=p.get('source_layout'), noise_mode=p.get('noise_mode', 'unknown'),
                         process_type=p.get('architecture', p.get('dataset_type', 'unknown')))
                    for i, p in enumerate(pairs)]
        if csv_path.exists():
            with csv_path.open(newline='') as handle:
                return [dict(sample_id=str(r.get('pair_id', i)), reference='reference/' + r['reference_file'],
                             search='search/' + r['search_file'], center_x=float(r['center_x']), center_y=float(r['center_y']),
                             source_layout=r.get('source_layout') or None, noise_mode=r.get('noise_mode', 'unknown'),
                             process_type=r.get('process_type', 'FinFET')) for i, r in enumerate(csv.DictReader(handle))]
        raise FileNotFoundError('Expected annotations.json or ground_truth.csv at dataset root')

    def _validate_records(self):
        missing_groups = 0
        for r in self.records:
            search, reference = _read_gray(self.root / r['search']), _read_gray(self.root / r['reference'])
            if search.shape != (self.config.search_size, self.config.search_size):
                raise ValueError(f"{r['sample_id']}: search is {search.shape}, expected 1000x1000")
            if reference.shape not in ((self.config.reference_size, self.config.reference_size), (1000, 1000)):
                raise ValueError(f"{r['sample_id']}: reference is {reference.shape}, expected 100x100 or existing 1000x1000")
            x, y = float(r['center_x']), float(r['center_y'])
            try: validate_center(x, y, self.config.search_size, self.config.reference_size)
            except ValueError as exc: raise ValueError(f"{r['sample_id']}: {exc}") from exc
            missing_groups += not bool(r.get('source_layout'))
        if missing_groups:
            warnings.warn(f"{missing_groups} samples lack source_layout; source-separated splitting needs a supplied manifest.")

    def __len__(self): return len(self.records)
    def visualize_sample(self, index, prediction=None):
        """Display a verified 100px GT box and optional prediction for debugging."""
        from .visualize import visualize_localization_result
        r = self.records[index]
        gx, gy = float(r['center_x']), float(r['center_y'])
        prediction = prediction or {'center_x': gx, 'center_y': gy, 'bbox': None, 'confidence': None}
        error = float(np.hypot(prediction['center_x']-gx, prediction['center_y']-gy))
        return visualize_localization_result(
            self.root / r['search'], self.root / r['reference'], (gx, gy), prediction,
            confidence=prediction.get('confidence'), error=error,
        )
    def __getitem__(self, index):
        r = self.records[index]; search = _read_gray(self.root / r['search']); reference = _read_gray(self.root / r['reference'])
        if reference.shape != (self.config.reference_size, self.config.reference_size):
            reference = cv2.resize(reference, (self.config.reference_size, self.config.reference_size), interpolation=cv2.INTER_AREA)
        return {'search': search.astype(np.float32) / 255.0, 'reference': reference.astype(np.float32) / 255.0,
                'center': np.array([float(r['center_x']), float(r['center_y'])], dtype=np.float32), 'meta': r}


def split_by_source_layout(dataset, seed=2026, ratios=(.70, .15, .15)):
    """Return record lists with no layout group appearing in more than one split."""
    grouped = defaultdict(list)
    for record in dataset.records:
        if not record.get('source_layout'):
            raise ValueError('Cannot safely split: source_layout is missing. Add it to metadata rather than randomly splitting crops.')
        grouped[record['source_layout']].append(record)
    groups = sorted(grouped); rng = np.random.default_rng(seed); rng.shuffle(groups)
    if len(groups) < 3:
        raise ValueError(f'Need at least three source_layout groups for train/validation/test, found {len(groups)}')
    n = len(groups); a, b = max(1, round(n * ratios[0])), max(1, round(n * (ratios[0] + ratios[1])))
    return [[r for g in selected for r in grouped[g]] for selected in (groups[:a], groups[a:b], groups[b:])]


