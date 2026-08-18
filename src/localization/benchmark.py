"""Benchmark candidate recall and final classical-localization accuracy on frozen data."""
import argparse, csv, json, time
from collections import defaultdict
from pathlib import Path
import cv2, numpy as np
from tqdm import tqdm
from .classical_matcher import ClassicalMatcherConfig, ClassicalSEMLocalizer
from .dataset import SEMLocalizationDataset
from .config import LocalizationConfig
from .visualize import save_localization_result

def config_from(path):
    if not path: return ClassicalMatcherConfig()
    values=json.loads(Path(path).read_text()); values.pop('selection_metrics',None); values.pop('tuning_metadata',None)
    for key in ('rotation_range','scales'):
        if key in values: values[key]=tuple(values[key])
    return ClassicalMatcherConfig(**values)
def recall(candidates, gt, k, tolerance):
    return any(np.hypot(x['center_x']-gt[0],x['center_y']-gt[1])<=tolerance for x in candidates[:k])
def summary(rows):
    errors=np.asarray([r['error_px'] for r in rows],float); result={'samples':len(rows),**{f'accuracy_at_{p}px':float(np.mean(errors<=p)) for p in (1,2,5,10,20)},'mean_error_px':float(errors.mean()),'median_error_px':float(np.median(errors)),'p90_error_px':float(np.percentile(errors,90)),'p95_error_px':float(np.percentile(errors,95)),'max_error_px':float(errors.max()),**{f'error_over_{p}px':float(np.mean(errors>p)) for p in (20,50,100,200)},'mean_inference_time_ms':float(np.mean([r['inference_time_ms'] for r in rows])),'p95_inference_time_ms':float(np.percentile([r['inference_time_ms'] for r in rows],95))}
    for k in (1,5,10,20,50,100):
        for tol in (2,5,10,20): result[f'candidate_recall_at_{k}_within_{tol}px']=float(np.mean([r[f'recall_{k}_{tol}'] for r in rows]))
    for representation in ('intensity','local','gradient','edge','distance','structure','fused_full','fused_coarse'):
        for k in (10,20,50): result[f'{representation}_candidate_recall_at_{k}_within_10px']=float(np.mean([r[f'{representation}_recall_{k}_10'] for r in rows]))
    result['ambiguous_rate' ]=float(np.mean([r['low_confidence'] for r in rows])); return result
def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True);p.add_argument('--config');p.add_argument('--output-dir',default='benchmark_results');p.add_argument('--limit',type=int);p.add_argument('--source-layouts',nargs='+');p.add_argument('--save-failures',action='store_true');p.add_argument('--disable-center-tie',action='store_true',help='Score-only ranking for controlled comparison; default uses the official conditional centre tie rule.');a=p.parse_args()
    ds=SEMLocalizationDataset(a.dataset,LocalizationConfig());
    if a.source_layouts:
        allowed=set(a.source_layouts); ds.records=[r for r in ds.records if r.get('source_layout') in allowed]
        if not ds.records: raise SystemExit('No records match --source-layouts.')
    config=config_from(a.config)
    if a.disable_center_tie:
        values=dict(config.__dict__); values['use_center_tie_break']=False; config=ClassicalMatcherConfig(**values)
    matcher=ClassicalSEMLocalizer(config); rows=[]; out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    total=len(ds) if a.limit is None else min(a.limit,len(ds))
    print(f'Evaluating {total} sample(s) from {a.dataset}')
    for index in tqdm(range(total), desc='Evaluating localization', unit='sample', dynamic_ncols=True):
        item=ds[index]; meta=item['meta']; search=cv2.imread(str(ds.root/meta['search']),0); reference=cv2.imread(str(ds.root/meta['reference']),0); gt=(float(meta['center_x']),float(meta['center_y']))
        diagnostics=matcher.candidate_diagnostics(search,reference,100); result=matcher.localize(search,reference); error=float(np.hypot(result['center_x']-gt[0],result['center_y']-gt[1]))
        row={'sample_id':meta['sample_id'],'source_layout':meta.get('source_layout'),'architecture':meta.get('process_type'),'noise_mode':meta.get('noise_mode'),'seed':meta.get('seed'),'gt_x':gt[0],'gt_y':gt[1],'pred_x':result['center_x'],'pred_y':result['center_y'],'error_px':error,'inference_time_ms':result['inference_time_ms'],'confidence':result['confidence'],'low_confidence':result['low_confidence'],'score':result['score'],'second_score':result['second_score'],'peak_ratio':result['peak_ratio'],'rotation':result['rotation'],'scale':result['scale']}
        for k in (1,5,10,20,50,100):
            for tol in (2,5,10,20): row[f'recall_{k}_{tol}']=recall(diagnostics['fused_full'],gt,k,tol)
        for representation, candidates in diagnostics.items():
            for k in (10,20,50): row[f'{representation}_recall_{k}_10']=recall(candidates,gt,k,10)
        rows.append(row)
        if a.save_failures and (error > 20 or result['low_confidence']):
            folder=out/('failures' if error > 20 else 'ambiguous')/str(meta['sample_id']); folder.mkdir(parents=True,exist_ok=True)
            cv2.imwrite(str(folder/'search.png'),search); cv2.imwrite(str(folder/'reference.png'),reference)
            save_localization_result(search,folder/'visualization.png',gt,result,result['confidence'],error)
            (folder/'metadata.json').write_text(json.dumps({**row,'top_k':result['top_k']},indent=2))
    with (out/'predictions.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    groups={'combined':rows}
    for key in ('architecture','noise_mode','source_layout','seed'):
        for value in {str(r.get(key)) for r in rows}: groups[f'{key}:{value}']=[r for r in rows if str(r.get(key))==value]
    metrics={name:summary(group) for name,group in groups.items()}; (out/'metrics.json').write_text(json.dumps(metrics,indent=2)); print(json.dumps(metrics['combined'],indent=2))
if __name__=='__main__': main()





