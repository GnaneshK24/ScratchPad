"""Evaluate the classical SEM localizer on the current dataset formats."""
import argparse, csv, json
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
from .config import LocalizationConfig
from .dataset import SEMLocalizationDataset
from .inference import localize
from .visualize import save_localization_result

def metrics(rows):
    errors = np.asarray([r['error_px'] for r in rows], dtype=float)
    return {'samples': len(rows), **{f'accuracy_at_{p}px': float(np.mean(errors <= p)) for p in (2, 5, 10, 20, 50)}, 'mean_error_px': float(errors.mean()), 'median_error_px': float(np.median(errors)), 'p90_error_px': float(np.percentile(errors, 90)), 'p95_error_px': float(np.percentile(errors, 95)), 'max_error_px': float(errors.max()), 'mean_confidence': float(np.mean([r['confidence'] for r in rows])), 'false_localization_rate': float(np.mean(errors > 10))}

def main():
    p = argparse.ArgumentParser(); p.add_argument('--dataset-dir', required=True); p.add_argument('--output-dir', default='evaluation'); p.add_argument('--visualize-count', type=int, default=20); a = p.parse_args()
    ds = SEMLocalizationDataset(a.dataset_dir, LocalizationConfig()); rows = []; images = []
    print(f'Evaluating {len(ds)} sample(s) from {a.dataset_dir}')
    for item in tqdm(ds, total=len(ds), desc='Evaluating localization', unit='sample', dynamic_ncols=True):
        meta = item['meta']; search = cv2.imread(str(ds.root/meta['search']), cv2.IMREAD_GRAYSCALE); reference = cv2.imread(str(ds.root/meta['reference']), cv2.IMREAD_GRAYSCALE)
        result = localize(search, reference); gx, gy = float(meta['center_x']), float(meta['center_y']); error = float(np.hypot(result['center_x']-gx, result['center_y']-gy))
        row = {'sample_id': meta['sample_id'], 'source_layout': meta.get('source_layout'), 'process_type': meta.get('process_type'), 'noise_mode': meta.get('noise_mode'), 'gt_x': gx, 'gt_y': gy, 'pred_x': result['center_x'], 'pred_y': result['center_y'], 'error_px': error, 'confidence': result['confidence'], 'score': result['score'], 'second_score': result['second_score'], 'peak_ratio': result['peak_ratio'], 'rotation': result['rotation'], 'scale': result['scale'], 'low_confidence': result['low_confidence'], 'top_k': json.dumps(result['top_k']), 'search_path': meta['search'], 'reference_path': meta['reference']}
        rows.append(row); images.append((search, result, gx, gy, error))
    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    for index in np.argsort([r['error_px'] for r in rows])[-min(a.visualize_count, len(rows)):]:
        search, result, gx, gy, error = images[int(index)]; save_localization_result(search, out/'visualizations'/f'{rows[int(index)]["sample_id"]}.png', (gx, gy), result, result['confidence'], error)
    with (out/'predictions.csv').open('w', newline='') as f: writer = csv.DictWriter(f, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    groups = {'combined': rows}
    for key in ('process_type', 'noise_mode', 'source_layout'):
        for value in sorted({str(r.get(key)) for r in rows}): groups[f'{key}:{value}'] = [r for r in rows if str(r.get(key)) == value]
    (out/'metrics.json').write_text(json.dumps({name: metrics(group) for name, group in groups.items()}, indent=2)); print(json.dumps(metrics(rows), indent=2))
if __name__ == '__main__': main()
