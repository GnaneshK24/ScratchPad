"""Build a source-layout-labelled FinFET benchmark without changing the generator."""
import argparse, json, sys
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from dataset_generator import FinFETSEMDatasetGenerator

def generator_for(image, seed, noise_mode):
    generator = object.__new__(FinFETSEMDatasetGenerator)
    generator.input_dir = Path('.'); generator.seed = seed
    generator.noise_mode = generator._normalize_noise_mode(noise_mode)
    generator.noise_params = {}; generator.base_images = [image]; generator.ground_truth_data = []
    return generator

def main():
    p = argparse.ArgumentParser(description='Create an immutable, source-layout-labelled FinFET benchmark.')
    p.add_argument('--architecture', choices=('finfet','dram'), required=True); p.add_argument('--base-images-dir', default='finfet_base_images')
    p.add_argument('--output-dir', required=True); p.add_argument('--samples-per-layout', type=int, default=50); p.add_argument('--seed', type=int, default=42)
    p.add_argument('--noise-modes', nargs='+', default=['clean','low','medium','high']); a = p.parse_args()
    if a.architecture == 'dram': raise SystemExit('DRAM benchmark blocked: current generator does not embed its reference in the search; see LOCALIZATION_OPTIMIZATION_AUDIT.md.')
    files = sorted(Path(a.base_images_dir).glob('*.png'))
    if len(files) < 3: raise SystemExit('Need at least three base layouts for source-disjoint validation/test.')
    out = Path(a.output_dir); images = out/'images'; images.mkdir(parents=True, exist_ok=True); records=[]
    total = len(files) * a.samples_per_layout
    print(f'Base layouts: {len(files)} loaded from {a.base_images_dir}')
    print(f'Generating {total} benchmark pairs in {out}')
    with tqdm(total=total, desc='Generating benchmark', unit='pair', dynamic_ncols=True) as progress:
        for layout_index, layout in enumerate(files):
            base=cv2.imread(str(layout),cv2.IMREAD_GRAYSCALE)
            if base is None: continue
            progress.set_postfix_str(f'layout={layout.name}')
            for i in range(a.samples_per_layout):
                mode=a.noise_modes[i % len(a.noise_modes)]; seed=a.seed + layout_index*100000 + i
                np.random.seed(seed); reference, search, meta=generator_for(base, seed, mode).generate_image_pair(i)
                sample_id=f'{layout_index:03d}_{i:05d}'; ref=f'images/reference_{sample_id}.png'; sea=f'images/search_{sample_id}.png'
                cv2.imwrite(str(out/ref), reference); cv2.imwrite(str(out/sea), search)
                records.append({'pair_id':sample_id,'reference_path':ref,'search_path':sea,'ground_truth_center':meta['ground_truth_center'],'source_layout':layout.name,'noise_mode':mode,'architecture':'FinFET','seed':seed,'rotation_angle':meta.get('rotation_angle')})
                progress.update(1)
    manifest={'format':'driftsense_benchmark_v1','architecture':'FinFET','seed':a.seed,'pairs':records}
    (out/'annotations.json').write_text(json.dumps(manifest,indent=2)); (out/'benchmark_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'Created immutable benchmark with {len(records)} pairs: {out}')
if __name__ == '__main__': main()
