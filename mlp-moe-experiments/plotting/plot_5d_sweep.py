import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import itertools
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--results_dir', type=str, required=True)
parser.add_argument('--config', type=str, default=None)
args = parser.parse_args()

# Load results
results_dir = Path(args.results_dir)
if not (results_dir / 'stats').exists():
    if args.config:
        results_dir = results_dir / args.config / 'stats'
    else:
        config_dirs = [d for d in results_dir.iterdir() if d.is_dir() and (d / 'stats').exists()]
        if config_dirs:
            results_dir = config_dirs[0] / 'stats'
        else:
            raise FileNotFoundError(f"No stats directory in {results_dir}")
else:
    results_dir = results_dir / 'stats'

# Load all files
print("Loading results...")
all_files = {}
for f in results_dir.glob("*.npz"):
    data = np.load(f, allow_pickle=True)
    meta = data['metadata'].item()
    a = meta['args']
    
    # Get multipliers (default to 1.0 if not present)
    init_std = a.get('init_std_mult', 1.0)
    lr_in = a.get('lr_mult_in', 1.0)
    lr_out = a.get('lr_mult_out', 1.0)
    lr_router = a.get('lr_mult_router', 1.0)
    lr_exp1 = a.get('lr_mult_expert1', 1.0)
    lr_exp2 = a.get('lr_mult_expert2', 1.0)
    seed = a['seed']
    
    key = (init_std, lr_in, lr_out, lr_router, lr_exp1, lr_exp2, seed)

    # Prefer val_top5_acc for multi-class datasets, fallback to val_acc
    if 'val_top5_acc' in data and len(data['val_top5_acc']) > 0:
        val_acc = data['val_top5_acc'][-1]
        val_top1 = data['val_acc'][-1] if 'val_acc' in data and len(data['val_acc']) > 0 else np.nan
    else:
        val_acc = data['val_acc'][-1] if 'val_acc' in data and len(data['val_acc']) > 0 else np.nan
        val_top1 = val_acc

    train_acc = data['train_acc'][-1] if len(data['train_acc']) > 0 else np.nan
    train_top5 = data['train_top5_acc'][-1] if 'train_top5_acc' in data and len(data['train_top5_acc']) > 0 else np.nan
    all_files[key] = {'val_acc': val_acc, 'val_top1': val_top1, 'train_acc': train_acc, 'train_top5': train_top5}

print(f"Loaded {len(all_files)} result files")

# Extract unique values
init_stds = sorted(set(k[0] for k in all_files.keys()))
lr_ins = sorted(set(k[1] for k in all_files.keys()))
lr_outs = sorted(set(k[2] for k in all_files.keys()))
lr_routers = sorted(set(k[3] for k in all_files.keys()))
lr_exp1s = sorted(set(k[4] for k in all_files.keys()))
lr_exp2s = sorted(set(k[5] for k in all_files.keys()))

print(f"Init std: {init_stds}")
print(f"LR in: {lr_ins}")
print(f"LR out: {lr_outs}")
print(f"LR router: {lr_routers}")
print(f"LR expert1: {lr_exp1s}")
print(f"LR expert2: {lr_exp2s}")

# Aggregate over seeds
results = {}
for init_std, lr_in, lr_out, lr_router, lr_exp1, lr_exp2 in itertools.product(
        init_stds, lr_ins, lr_outs, lr_routers, lr_exp1s, lr_exp2s):
    vals = [all_files[k] for k in all_files.keys() 
            if abs(k[0]-init_std)<1e-6 and abs(k[1]-lr_in)<1e-6 and abs(k[2]-lr_out)<1e-6 and
               abs(k[3]-lr_router)<1e-6 and abs(k[4]-lr_exp1)<1e-6 and abs(k[5]-lr_exp2)<1e-6]
    if vals:
        val_accs = [v['val_acc'] for v in vals if not np.isnan(v['val_acc'])]
        val_top1s = [v['val_top1'] for v in vals if not np.isnan(v['val_top1'])]
        train_accs = [v['train_acc'] for v in vals if not np.isnan(v['train_acc'])]
        train_top5s = [v['train_top5'] for v in vals if not np.isnan(v['train_top5'])]
        results[(init_std, lr_in, lr_out, lr_router, lr_exp1, lr_exp2)] = {
            'val_acc': np.mean(val_accs) if val_accs else np.nan,
            'val_top1': np.mean(val_top1s) if val_top1s else np.nan,
            'train_acc': np.mean(train_accs) if train_accs else np.nan,
            'train_top5': np.mean(train_top5s) if train_top5s else np.nan,
        }

# Find optimal
valid_results = {k: v for k, v in results.items() if not np.isnan(v['val_acc'])}
if not valid_results:
    raise ValueError("All results have NaN val_acc")

best_val_acc = max(r['val_acc'] for r in valid_results.values())
best_config = [k for k, v in valid_results.items() if v['val_acc'] == best_val_acc][0]
opt_init, opt_in, opt_out, opt_router, opt_exp1, opt_exp2 = best_config
opt_val_top1 = valid_results[best_config]['val_top1']

# Print optimal hyperparameters with both top-5 (val_acc) and top-1 if available
if not np.isnan(opt_val_top1) and abs(opt_val_top1 - best_val_acc) > 1e-6:
    print(f"\nOptimal: init_std={opt_init}, lr_in={opt_in}, lr_out={opt_out}, lr_router={opt_router}, lr_exp1={opt_exp1}, lr_exp2={opt_exp2}")
    print(f"  val_top5_acc={best_val_acc:.4f}, val_top1_acc={opt_val_top1:.4f}")
else:
    print(f"\nOptimal: init_std={opt_init}, lr_in={opt_in}, lr_out={opt_out}, lr_router={opt_router}, lr_exp1={opt_exp1}, lr_exp2={opt_exp2}, val_acc={best_val_acc:.4f}")

# Create 6x6 grid plot (each cell is a 5x5 heatmap)
metric_configs = [
    ('train_top5', 'Train Top-5 Acc', '6d_sweep_train_top5_acc.png'),
    ('val_top1', 'Val Top-1 Acc', '6d_sweep_val_top1_acc.png'),
    ('val_acc', 'Val Top-5 Acc', '6d_sweep_val_top5_acc.png'),
    ('train_acc', 'Train Acc', '6d_sweep_train_acc.png'),
]

param_names = ['init_std', 'lr_in', 'lr_out', 'lr_router', 'lr_exp1', 'lr_exp2']
all_vals = [init_stds, lr_ins, lr_outs, lr_routers, lr_exp1s, lr_exp2s]
opt_vals = [opt_init, opt_in, opt_out, opt_router, opt_exp1, opt_exp2]

for metric, title, filename in metric_configs:
    # Skip top1 plot if top1 == top5 (no val_top5_acc in data)
    if metric == 'val_top1':
        all_top1 = [v['val_top1'] for v in results.values() if not np.isnan(v['val_top1'])]
        all_top5 = [v['val_acc'] for v in results.values() if not np.isnan(v['val_acc'])]
        if all_top1 and all_top5 and np.allclose(all_top1, all_top5[:len(all_top1)], atol=1e-6):
            print(f"Skipping {filename} (top1 == top5, no separate top5 data)")
            continue
    if metric == 'train_top5':
        if all(np.isnan(v['train_top5']) for v in results.values()):
            print(f"Skipping {filename} (no train_top5_acc in data)")
            continue

    fig, axes = plt.subplots(6, 6, figsize=(30, 30))
    fig.suptitle(f'{title} - 6D Sweep', fontsize=24)

    for row in range(6):
        for col in range(6):
            if row == col:
                axes[row, col].axis('off')
                continue

            vals1, vals2 = all_vals[row], all_vals[col]
            grid = np.full((len(vals1), len(vals2)), np.nan)

            for i, v1 in enumerate(vals1):
                for j, v2 in enumerate(vals2):
                    params = list(opt_vals)
                    params[row] = v1
                    params[col] = v2
                    key = tuple(params)
                    if key in results:
                        grid[i, j] = results[key][metric]

            im = axes[row, col].imshow(grid, cmap='viridis', aspect='auto', origin='lower')
            if col == 0:
                axes[row, col].set_ylabel(param_names[row], fontsize=10)
            if row == 5:
                axes[row, col].set_xlabel(param_names[col], fontsize=10)
            axes[row, col].set_xticks(range(len(vals2)))
            axes[row, col].set_yticks(range(len(vals1)))
            axes[row, col].set_xticklabels([f"{v:.1g}" for v in vals2], fontsize=6, rotation=45)
            axes[row, col].set_yticklabels([f"{v:.1g}" for v in vals1], fontsize=6)
            plt.colorbar(im, ax=axes[row, col], fraction=0.046, pad=0.04)

    plt.tight_layout()
    output_path = results_dir.parent / filename
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved {output_path}")
    plt.close()

# Keep backwards-compatible symlink names
for old, new in [('6d_sweep_val_acc.png', '6d_sweep_val_top5_acc.png')]:
    old_path = results_dir.parent / old
    new_path = results_dir.parent / new
    if new_path.exists() and not old_path.exists():
        import shutil
        shutil.copy2(new_path, old_path)
