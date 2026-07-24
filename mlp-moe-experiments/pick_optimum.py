"""Pick the argmax multiplier setting from a 6D sweep and write it into tuned_multipliers.py.

Reads all `stats/*.npz` files under `<sweep_results>/<config>/stats/`, extracts multipliers
from each run's metadata, picks the run with the highest validation accuracy
(val_top5_acc if present else val_acc), and appends a sentinel-bracketed override to
tuned_multipliers.py that mutates TUNED_MULTIPLIERS at import time. Re-running the picker
replaces the prior block for the same (config, regime_key) pair.
"""

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import numpy as np


MULT_KEYS = (
    'init_std_mult',
    'lr_mult_in',
    'lr_mult_out',
    'lr_mult_router',
    'lr_mult_expert1',
    'lr_mult_expert2',
)


def _best_acc(data):
    if 'val_top5_acc' in data.files and len(data['val_top5_acc']) > 0:
        arr = np.asarray(data['val_top5_acc'], dtype=float)
    elif 'val_acc' in data.files and len(data['val_acc']) > 0:
        arr = np.asarray(data['val_acc'], dtype=float)
    else:
        return float('nan')
    arr = arr[np.isfinite(arr)]
    return float(arr.max()) if arr.size else float('nan')


def _run_multipliers(data):
    md = data['metadata'].item()
    args = md['args']
    out = {'base_lr': float(args['eta'])}
    for k in MULT_KEYS:
        out[k] = float(args[k])
    return out


def scan(sweep_dir: Path, config: str):
    stats_dir = sweep_dir / config / 'stats'
    files = sorted(glob.glob(str(stats_dir / 'nn_*.npz')))
    if not files:
        raise SystemExit(f'no stats files under {stats_dir}')
    runs = []
    for f in files:
        try:
            d = np.load(f, allow_pickle=True)
            acc = _best_acc(d)
            if not np.isfinite(acc):
                continue
            mults = _run_multipliers(d)
            runs.append({'file': f, 'best_acc': acc, **mults})
        except Exception as e:
            print(f'skip {f}: {e}', file=sys.stderr)
    if not runs:
        raise SystemExit('no finite-accuracy runs found')
    runs.sort(key=lambda r: r['best_acc'], reverse=True)
    return runs


def sentinel(config: str, key: str):
    return f'# --- BEGIN auto-pick {config}/{key} ---', f'# --- END auto-pick {config}/{key} ---'


def upsert_override(tm_path: Path, config: str, regime_key: str, entry: dict):
    begin, end = sentinel(config, regime_key)
    body = tm_path.read_text()
    block_lines = [
        '',
        begin,
        f"TUNED_MULTIPLIERS.setdefault({config!r}, {{}})[{regime_key!r}] = {{",
    ]
    for k in ('base_lr', *MULT_KEYS):
        block_lines.append(f'    {k!r}: {entry[k]!r},')
    block_lines.append(f"    'best_acc': {entry['best_acc']!r},")
    block_lines.append('}')
    block_lines.append(end)
    new_block = '\n'.join(block_lines) + '\n'

    pat = re.compile(re.escape(begin) + r'.*?' + re.escape(end) + r'\n?', re.DOTALL)
    if pat.search(body):
        body = pat.sub(new_block.strip('\n') + '\n', body)
    else:
        body = body.rstrip() + '\n' + new_block
    tm_path.write_text(body)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sweep_results', required=True, type=Path)
    ap.add_argument('--config', required=True)
    ap.add_argument('--regime_key', required=True,
                    help="tuned_multipliers.py sub-dict key, e.g. 'soft_no_shared_ll1overN'")
    ap.add_argument('--tuned_multipliers_path', required=True, type=Path)
    ap.add_argument('--dump_json', type=Path, default=None)
    ap.add_argument('--top_n', type=int, default=10, help='how many top runs to print')
    args = ap.parse_args()

    runs = scan(args.sweep_results, args.config)
    best = runs[0]

    print(f'scanned {len(runs)} finite runs from {args.sweep_results}/{args.config}/stats')
    print(f'top {args.top_n}:')
    for r in runs[: args.top_n]:
        print(
            f"  acc={r['best_acc']:.4f}  base_lr={r['base_lr']}  "
            + '  '.join(f'{k[len("lr_mult_"):] if k.startswith("lr_mult_") else k}={r[k]}' for k in MULT_KEYS)
        )

    entry = {k: best[k] for k in ('base_lr', *MULT_KEYS)}
    entry['best_acc'] = round(best['best_acc'], 4)

    upsert_override(args.tuned_multipliers_path, args.config, args.regime_key, entry)
    print(f"\nwrote override block to {args.tuned_multipliers_path}")
    print(f"  key: {args.config!r} -> {args.regime_key!r}")

    if args.dump_json:
        args.dump_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'config': args.config,
            'regime_key': args.regime_key,
            'entry': entry,
            'source_file': best['file'],
            'n_runs_scanned': len(runs),
            'top_runs': runs[: args.top_n],
        }
        args.dump_json.write_text(json.dumps(payload, indent=2, default=str))
        print(f'wrote {args.dump_json}')


if __name__ == '__main__':
    main()
