#!/usr/bin/env python3
"""Plot learning rate sweep results."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from label_utils import format_label, format_subplot_title, get_figure_title

try:
    from tueplots import bundles
    plt.rcParams.update(bundles.icml2022())
    plt.rcParams.update({"figure.dpi": 300, "text.usetex": False})
    onefigsize = bundles.icml2022()['figure.figsize']
except ImportError:
    onefigsize = (3.25, 2.0)

def load_lr_sweep_data(results_dir, separate_routing_and_init=False, widths=None):
    """Load all LR sweep results

    Args:
        results_dir: Path to results directory
        separate_routing_and_init: If True, create separate "virtual configs" for
            different routing modes (soft/topk) and router initializations (zero/mup/etc).
            This allows plotting them as distinct configs for ablation studies.
    """
    results_path = Path(results_dir)
    data = {}
    routing_meta = {}  # config_name -> {'routing': set, 'rinit': set}

    # Reserved directory names that should not be treated as config directories
    RESERVED_DIRS = {'plots', 'rcc', 'figures', 'tables'}

    for config_dir in results_path.iterdir():
        if not config_dir.is_dir() or config_dir.name.startswith('.'):
            continue

        base_config_name = config_dir.name

        # Skip reserved directory names
        if base_config_name in RESERVED_DIRS:
            continue

        stats_dir = config_dir / 'stats'
        if not stats_dir.exists():
            continue

        for f in glob.glob(str(stats_dir / 'nn_N*.npz')):
            npz_data = np.load(f, allow_pickle=True)
            basename = os.path.basename(f)
            parts = basename.split('_')

            # Extract N, eta, seed
            N = int(parts[1][1:])
            if widths is not None and N not in widths:
                continue
            eta = None
            seed = 42

            for part in parts:
                if part.startswith('eta'):
                    eta = float(part.replace('eta', ''))
                if part.startswith('seed'):
                    seed = int(part.replace('seed', '').replace('.npz', ''))

            if eta is None:
                continue

            # Determine effective config name (with routing/init suffixes if requested)
            config_name = base_config_name
            if separate_routing_and_init:
                # Extract routing mode and router_init from filename
                routing_suffix = None
                rinit_suffix = None
                for part in parts:
                    if part == 'soft' or part.startswith('k'):
                        routing_suffix = part
                    if part.startswith('rinit'):
                        rinit_suffix = part

                # Build virtual config name
                if routing_suffix:
                    config_name += f"_{routing_suffix}"
                if rinit_suffix:
                    config_name += f"_{rinit_suffix}"

            if config_name not in data:
                data[config_name] = {}
            if N not in data[config_name]:
                data[config_name][N] = {}
            if eta not in data[config_name][N]:
                data[config_name][N][eta] = []

            data[config_name][N][eta].append({
                'train_loss': npz_data['train_loss'] if 'train_loss' in npz_data else [],
                'val_loss': npz_data['val_loss'] if 'val_loss' in npz_data else [],
                'train_acc': npz_data['train_acc'] if 'train_acc' in npz_data else [],
                'val_acc': npz_data['val_acc'] if 'val_acc' in npz_data else [],
                'train_top5_acc': npz_data['train_top5_acc'] if 'train_top5_acc' in npz_data else [],
                'val_top5_acc': npz_data['val_top5_acc'] if 'val_top5_acc' in npz_data else [],
                'seed': seed
            })

            # Track routing/rinit metadata per config (for title generation)
            if config_name not in routing_meta:
                routing_meta[config_name] = {'routing': set(), 'rinit': set()}
            for part in parts:
                if part == 'soft' or (part.startswith('k') and part[1:].isdigit()):
                    routing_meta[config_name]['routing'].add(part)
                if part.startswith('rinit'):
                    routing_meta[config_name]['rinit'].add(part)

    return data, routing_meta

def _routing_title_suffix(config, routing_meta):
    """Return ' — top-K, router init=X' suffix when all data for config shares one routing/rinit.

    Suppressed when config name already encodes routing/init (format_label handles it).
    """
    import re as _re
    # format_label already added routing/init info when these suffixes appear in the config name
    if '_rinit' in config or '_soft' in config or _re.search(r'_k\d', config):
        return ''
    meta = routing_meta.get(config, {})
    routing_set = meta.get('routing', set())
    rinit_set = meta.get('rinit', set())
    parts = []
    if len(routing_set) == 1:
        r = next(iter(routing_set))
        parts.append('soft' if r == 'soft' else f'top-{r[1:]}')
    if len(rinit_set) == 1:
        ri = next(iter(rinit_set))
        if ri == 'rinitzero':
            parts.append('router init=0')
        elif ri == 'rinitmup':
            parts.append('router init=1/N')
        elif ri == 'rinitntp':
            parts.append('router init=1/√N')
    return (' — ' + ', '.join(parts)) if parts else ''


def plot_lr_sweep(data, results_dir, metric='val_loss', timestep=-1, routing_meta=None):
    """Plot LR sweep results with 2-sigma confidence bands, darker for larger N"""
    configs = sorted(data.keys())

    for config in configs:
        Ns = sorted(data[config].keys())

        fig, ax = plt.subplots(figsize=(onefigsize[0]*1.5, onefigsize[1]*1.2))

        # Color map: darker for larger N
        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(Ns)))

        # Track all means across widths for proper ylim calculation
        all_means = []
        max_per_width = []

        for idx, N in enumerate(Ns):
            etas = sorted(data[config][N].keys())

            # Aggregate across seeds
            means = []
            stds = []

            for eta in etas:
                values = []
                for run in data[config][N][eta]:
                    if metric in run and len(run[metric]) > 0:
                        val = run[metric][timestep] if timestep < len(run[metric]) else run[metric][-1]
                        values.append(val)

                if values:
                    means.append(np.mean(values))
                    stds.append(np.std(values))
                else:
                    means.append(np.nan)
                    stds.append(0)

            means = np.array(means)
            stds = np.array(stds)

            # Track for ylim calculation
            all_means.extend(means[~np.isnan(means)])
            if len(means[~np.isnan(means)]) > 0:
                max_per_width.append(np.nanmax(means))

            # Plot with 2-sigma confidence bands
            color = colors[idx]
            ax.plot(etas, means, 'o-', color=color, linewidth=2, markersize=6, label=f'N={N}')
            ax.fill_between(etas, means - 2*stds, means + 2*stds, alpha=0.2, color=color)

        ax.set_xscale('log')
        if 'loss' in metric:
            ax.set_yscale('log')
        elif 'acc' in metric:
            # Set ylim based on global max and min of per-width maxes
            if all_means:
                global_max = np.max(all_means)
                min_of_maxes = np.min(max_per_width) if max_per_width else global_max

                # Upper limit: 1.05 * global_max but capped at 1.0
                ylim_upper = min(1.05 * global_max, 1.0)

                # Lower limit: at most 0.8 * min_of_maxes
                ylim_lower = min(0.8 * min_of_maxes, 0.95 * min_of_maxes)

                ax.set_ylim(ylim_lower, ylim_upper)
        ax.set_xlabel('Learning Rate')
        ylabel = metric.replace('_', ' ').title().replace('Acc', 'Accuracy')
        ax.set_ylabel(ylabel)
        suffix = _routing_title_suffix(config, routing_meta or {})
        ax.set_title(get_figure_title(config, results_dir) + suffix)
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        output_path = os.path.join(results_dir, f'lr_sweep_{config}_{metric}_t{timestep}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {output_path}")

def plot_lr_sweep_heatmap(data, results_dir, metric='val_loss', timestep=-1, routing_meta=None):
    """Plot LR sweep as heatmap"""
    configs = sorted(data.keys())
    
    for config in configs:
        Ns = sorted(data[config].keys())
        all_etas = set()
        for N in Ns:
            all_etas.update(data[config][N].keys())
        etas = sorted(all_etas)
        
        # Build matrix
        matrix = np.full((len(Ns), len(etas)), np.nan)
        
        for i, N in enumerate(Ns):
            for j, eta in enumerate(etas):
                if eta in data[config][N]:
                    values = []
                    for run in data[config][N][eta]:
                        if metric in run and len(run[metric]) > 0:
                            val = run[metric][timestep] if timestep < len(run[metric]) else run[metric][-1]
                            values.append(val)
                    if values:
                        matrix[i, j] = np.mean(values)
        
        # Plot
        fig, ax = plt.subplots(figsize=(onefigsize[0]*1.5, onefigsize[1]*1.2))
        
        if 'loss' in metric:
            im = ax.imshow(matrix, aspect='auto', cmap='viridis_r', norm=plt.matplotlib.colors.LogNorm())
        else:
            im = ax.imshow(matrix, aspect='auto', cmap='viridis')
        
        ax.set_xticks(range(len(etas)))
        ax.set_xticklabels([f'{eta:.4f}' for eta in etas], rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(len(Ns)))
        ax.set_yticklabels([f'N={N}' for N in Ns])
        ax.set_xlabel('Learning Rate')
        ax.set_ylabel('Width')
        suffix = _routing_title_suffix(config, routing_meta or {})
        ax.set_title(f'{format_subplot_title(config)}{suffix} - {metric.replace("_", " ").title()}')
        
        plt.colorbar(im, ax=ax)
        
        output_path = os.path.join(results_dir, f'lr_sweep_heatmap_{config}_{metric}_t{timestep}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved {output_path}")

def plot_lr_sweep_comparison(data, config_pairs, results_dir, metric='val_acc', timestep=-1):
    """Plot side-by-side comparison of config pairs"""
    from datetime import datetime

    for pair in config_pairs:
        if ':' not in pair:
            print(f"Warning: Invalid pair format '{pair}', expected 'config1:config2'")
            continue

        config1, config2 = pair.split(':', 1)

        if config1 not in data or config2 not in data:
            print(f"Warning: Config(s) not found: {config1}, {config2}")
            continue

        Ns = sorted(set(data[config1].keys()) & set(data[config2].keys()))
        if not Ns:
            print(f"Warning: No common widths for {config1} vs {config2}")
            continue

        fig, axes = plt.subplots(1, 2, figsize=(2*onefigsize[0]*1.5, onefigsize[1]*1.2), sharey=True)

        # Color map: darker for larger N
        colors = plt.cm.Blues(np.linspace(0.4, 0.95, len(Ns)))

        for config_idx, (config, ax) in enumerate([(config1, axes[0]), (config2, axes[1])]):
            all_means = []
            max_per_width = []

            for N_idx, N in enumerate(Ns):
                etas = sorted(data[config][N].keys())
                means = []
                stds = []

                for eta in etas:
                    values = []
                    for run in data[config][N][eta]:
                        if metric in run and len(run[metric]) > 0:
                            val = run[metric][timestep] if timestep < len(run[metric]) else run[metric][-1]
                            if not np.isnan(val):
                                values.append(val)

                    if values:
                        means.append(np.mean(values))
                        stds.append(np.std(values))
                    else:
                        means.append(np.nan)
                        stds.append(0)

                means = np.array(means)
                stds = np.array(stds)

                # Track all means and max per width
                valid_mask = ~np.isnan(means)
                if valid_mask.any():
                    all_means.extend(means[valid_mask])
                    if 'acc' in metric:
                        max_per_width.append(np.max(means[valid_mask]))
                    else:
                        max_per_width.append(np.min(means[valid_mask]))

                ax.plot(etas, means, 'o-', color=colors[N_idx], label=f'N={N}',
                       linewidth=2, markersize=6)
                ax.fill_between(etas, means - 2*stds, means + 2*stds,
                               alpha=0.2, color=colors[N_idx])

            # Set y-limits based on global max and min of per-width maxes
            if all_means:
                if 'acc' in metric:
                    global_max = np.max(all_means)
                    min_of_maxes = np.min(max_per_width) if max_per_width else global_max

                    # Upper limit: 1.05 * global_max but capped at 1.0
                    ylim_upper = min(1.05 * global_max, 1.0)

                    # Lower limit: at most 0.8 * min_of_maxes
                    ylim_lower = min(0.8 * min_of_maxes, 0.95 * min_of_maxes)

                    ax.set_ylim([ylim_lower, ylim_upper])
                elif 'loss' in metric:
                    global_min = np.min(all_means)
                    max_of_mins = np.max(max_per_width) if max_per_width else global_min
                    ax.set_ylim([0.8 * global_min, 1.2 * max_of_mins])

            ax.set_xscale('log')
            if 'loss' in metric and config_idx == 1:  # Only log scale y for loss on second plot
                ax.set_yscale('log')

            ax.set_xlabel('Learning Rate')
            if config_idx == 0:
                ylabel = metric.replace('_', ' ').title().replace('Acc', 'Accuracy')
                ax.set_ylabel(ylabel)

            ax.set_title(get_figure_title(config, results_dir))
            ax.legend(fontsize=8, loc='best')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        date_str = datetime.now().strftime('%m%d')
        metric_name = metric.replace('_', '')
        output_path = os.path.join(results_dir, f'lr_sweep_comparison_{config1}_vs_{config2}_{metric_name}_{date_str}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved comparison plot: {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', required=True, help='Results directory')
    parser.add_argument('--metrics', nargs='+', default=['val_loss', 'val_acc', 'train_loss', 'train_acc', 'val_top5_acc', 'train_top5_acc'],
                       help='Metrics to plot')
    parser.add_argument('--timesteps', type=int, nargs='+', default=[-1],
                       help='Timesteps to plot (-1 for final)')
    parser.add_argument('--comparison', action='store_true',
                       help='Enable comparison mode to plot config pairs side-by-side')
    parser.add_argument('--config_pairs', action='append', type=str,
                       help='Config pair to compare in format "config1:config2". '
                            'Can be specified multiple times for multiple comparisons.')
    parser.add_argument('--separate_routing_and_init', action='store_true',
                       help='Create separate virtual configs for different routing modes and router inits (for ablation studies)')
    parser.add_argument('--widths', type=int, nargs='+', default=None,
                       help='Only include these widths (e.g. --widths 256 1024 4096)')
    args = parser.parse_args()

    print(f"Loading data from {args.results_dir}...")
    data, routing_meta = load_lr_sweep_data(args.results_dir, separate_routing_and_init=args.separate_routing_and_init, widths=args.widths)

    if not data:
        print("No data found!")
        return

    # Comparison mode
    if args.comparison:
        if not args.config_pairs:
            print("Error: --comparison requires --config_pairs")
            return

        print("Plotting LR sweep comparisons...")
        for metric in args.metrics:
            for timestep in args.timesteps:
                print(f"  {metric} at t={timestep}")
                plot_lr_sweep_comparison(data, args.config_pairs, args.results_dir, metric, timestep)
        print(f"\n✓ LR sweep comparison plots saved to {args.results_dir}/")

    # Basic mode
    else:
        print("Plotting LR sweep results...")
        for metric in args.metrics:
            has_data = any(
                metric in run and len(run[metric]) > 0
                for config in data.values()
                for N_data in config.values()
                for eta_data in N_data.values()
                for run in eta_data
            )
            if not has_data:
                print(f"  {metric}: no data, skipping")
                continue
            for timestep in args.timesteps:
                print(f"  {metric} at t={timestep}")
                plot_lr_sweep(data, args.results_dir, metric, timestep, routing_meta=routing_meta)
                plot_lr_sweep_heatmap(data, args.results_dir, metric, timestep, routing_meta=routing_meta)
        print(f"\n✓ LR sweep plots saved to {args.results_dir}/")

if __name__ == '__main__':
    main()
