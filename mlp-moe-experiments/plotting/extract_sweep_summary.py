"""
Extract summary statistics from a 6D sweep results directory.
Saves a compressed JSON with train/val loss+acc+top5 for each multiplier combination.
"""
import json, gzip, argparse, sys
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', type=str, required=True)
parser.add_argument('--out', type=str, default=None, help='Output .json.gz path')
args = parser.parse_args()

results_dir = Path(args.results_dir)
out_path = Path(args.out) if args.out else results_dir / 'sweep_summary.json.gz'

METRICS = ['train_loss', 'train_acc', 'train_top5_acc', 'val_loss', 'val_acc', 'val_top5_acc']

# Find all npz files (they live in <config>/stats/*.npz)
npz_files = list(results_dir.rglob('*.npz'))
print(f"Found {len(npz_files)} npz files in {results_dir}", flush=True)

summary = {}  # key -> {metric: [values...]}

for i, f in enumerate(npz_files):
    if i % 1000 == 0:
        print(f"  {i}/{len(npz_files)}", flush=True)
    try:
        data = np.load(f, allow_pickle=True)
        meta = data['metadata'].item()
        a = meta['args']

        key = {
            'init_std': float(a.get('init_std_mult', 1.0)),
            'lr_in':    float(a.get('lr_mult_in', 1.0)),
            'lr_out':   float(a.get('lr_mult_out', 1.0)),
            'lr_router':float(a.get('lr_mult_router', 1.0)),
            'lr_exp1':  float(a.get('lr_mult_expert1', 1.0)),
            'lr_exp2':  float(a.get('lr_mult_expert2', 1.0)),
            'seed':     int(a.get('seed', 42)),
            'N':        int(a.get('N', 0)),
        }
        key_str = json.dumps(key, sort_keys=True)

        entry = {}
        for m in METRICS:
            if m in data and len(data[m]) > 0:
                entry[m] = data[m].tolist()
            else:
                entry[m] = []
        summary[key_str] = entry
    except Exception as e:
        print(f"  Warning: failed on {f.name}: {e}", flush=True)

print(f"Saving {len(summary)} entries to {out_path}", flush=True)
with gzip.open(out_path, 'wt', encoding='utf-8') as fh:
    json.dump(summary, fh)
print("Done.", flush=True)
