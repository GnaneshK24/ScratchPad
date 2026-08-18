"""Random-search tuning for the classical SEM matcher; test data is never accepted here."""
import argparse, json, random, subprocess, sys
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--validation-dataset',required=True);p.add_argument('--output-dir',default='tuning');p.add_argument('--trials',type=int,default=24);p.add_argument('--seed',type=int,default=42);p.add_argument('--limit',type=int);p.add_argument('--source-layouts',nargs='+');a=p.parse_args(); rng=random.Random(a.seed); out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True); trials=[]
 for i in range(a.trials):
  weights=[rng.uniform(.05,.35) for _ in range(9)]; total=sum(weights); weights=[v/total for v in weights]
  config={'intensity_weight':weights[0],'contrast_weight':weights[1],'gradient_weight':weights[2],'gx_weight':weights[3],'gy_weight':weights[4],'edge_weight':weights[5],'shape_weight':weights[6],'local_structure_weight':weights[7],'bandpass_weight':weights[8],'top_k':rng.choice([40,50,60]),'verify_top_k':rng.choice([20,30,40]),'coarse_peaks_per_representation':rng.choice([32,48,64]),'nms_radius':rng.choice([15,20,25,35]),'refinement_window':rng.choice([50,70,90]),'rotation_range':[-3.,3.],'rotation_step':rng.choice([1.,1.5,2.]),'scales':rng.choice([[.97,1.,1.03],[.95,.975,1.,1.025,1.05]])}
  path=out/f'trial_{i:03d}.json';path.write_text(json.dumps(config,indent=2)); run=out/f'trial_{i:03d}'; command=[sys.executable,'-m','src.localization.benchmark','--dataset',a.validation_dataset,'--config',str(path),'--output-dir',str(run)];
  if a.limit: command += ['--limit',str(a.limit)]
  if a.source_layouts: command += ['--source-layouts',*a.source_layouts]
  subprocess.run(command,check=True); metrics=json.loads((run/'metrics.json').read_text())['combined']; trials.append({'config':config,'metrics':metrics})
 trials.sort(key=lambda item:(item['metrics']['accuracy_at_5px'],item['metrics']['accuracy_at_10px'],item['metrics']['accuracy_at_2px'],item['metrics']['candidate_recall_at_20_within_10px'],-item['metrics']['p95_error_px']),reverse=True); best=trials[0]; best['config']['selection_metrics']=best['metrics'];best['config']['tuning_metadata']={'seed':a.seed,'trials':a.trials,'validation_dataset':a.validation_dataset};(out/'best_classical.json').write_text(json.dumps(best['config'],indent=2));(out/'trials.json').write_text(json.dumps(trials,indent=2));print(json.dumps(best,indent=2))
if __name__=='__main__': main()


