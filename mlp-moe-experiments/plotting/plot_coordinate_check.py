#!/usr/bin/env python3
"""Plot effective and propagating updates for coordinate check."""

import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from pathlib import Path

try:
    from tueplots import bundles
    plt.rcParams.update(bundles.icml2022())
    plt.rcParams.update({"figure.dpi": 300, "text.usetex": False})
    onefigsize = bundles.icml2022()['figure.figsize']
except ImportError:
    onefigsize = (3.25, 2.0)

from label_utils import (
    format_label, get_optimizer_from_config, format_subplot_title, get_scaling_regime,
    get_figure_title,
)


def set_shared_ylim(axes, margin=0.02):
    """Set shared y-axis limits for all axes based on global min/max.

    Args:
        axes: List of matplotlib axes objects
        margin: Fraction of range to add as margin (default 0.02 = 2%)
    """
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    # Collect all y-values from all axes
    all_ydata = []
    for ax in axes:
        for line in ax.get_lines():
            ydata = line.get_ydata()
            # Filter out inf/nan values
            ydata_clean = ydata[np.isfinite(ydata)]
            if len(ydata_clean) > 0:
                all_ydata.extend(ydata_clean)

    if len(all_ydata) > 0:
        global_min = np.min(all_ydata)
        global_max = np.max(all_ydata)

        # Handle log scale - need to work in log space
        if axes[0].get_yscale() == 'log' and global_min > 0:
            log_min = np.log10(global_min)
            log_max = np.log10(global_max)
            log_range = log_max - log_min
            ylim_min = 10 ** (log_min - margin * log_range)
            ylim_max = 10 ** (log_max + margin * log_range)
        else:
            # Linear scale
            data_range = global_max - global_min
            ylim_min = global_min - margin * data_range
            ylim_max = global_max + margin * data_range

        # Apply to all axes
        for ax in axes:
            try:
                ax.set_ylim(ylim_min, ylim_max)
            except:
                pass  # Skip if ylim setting fails

def get_plot_save_path(results_dir, config_name, filename, is_joint_plot=True):
    """Determine where to save a plot.

    Args:
        results_dir: Base results directory
        config_name: Name of the config (or None for joint plots)
        filename: Filename for the plot
        is_joint_plot: Whether this is a joint plot (multiple configs) or individual

    Returns:
        Path object for where to save the plot
    """
    if is_joint_plot or config_name is None:
        # Joint plots go to rcc/ subdirectory
        plot_dir = Path(results_dir) / 'rcc'
    else:
        # Individual plots go to config_name/plots/ subdirectory
        plot_dir = Path(results_dir) / config_name / 'plots'

    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir / filename

# Global config order used across all plotting functions
# Our muP-MoE configs (with router init 0):
#   - fixed_E_mup_multfree, fixed_E_mup_adam
#   - mup_allscaling_multfree, mup_adam_allscaling_ours
#   - mup_bottleneck_ours, mup_bottleneck_adam_multfree
# Baselines (with router init 1/N or standard expert init):
#   - fixed_E_mup_multfree (router init 1/N), fixed_E_mup_adam (router init 1/N)
#   - mup_stdinit_allscaling_multfree, mup_adam_allscaling_stdinit_ours
#   - mup_bottleneck_stdinit, mup_bottleneck_adam_stdinit_multfree
CONFIG_ORDER = [
    # Fixed-E configs (ours + baselines)
    'fixed_E_ntp', 'fixed_E_mup', 'fixed_E_mup_multfree', 'fixed_E_mup_largerouterlr', 'fixed_E_mup_largerouterlr_nonasympQ', 'fixed_E_mup_adam',
    # Legacy expert scaling configs
    'ntp', 'mup_largerouter', 'mup_smallrouter', 'mup_smallrouter_singlelr',
    # All-scaling configs (ours + baselines)
    'ntp_allscaling', 'mup_allscaling', 'mup_allscaling_nonasympQ', 'mup_clt_smallinlr_allscaling_multfree',
    'mup_allscaling_multfree', 'mup_stdinit_allscaling_multfree', 'mup_smallinputlr_allscaling_multfree',
    'mup_heuristic_allscaling_multfree', 'mup_adam_allscaling_ours', 'mup_adam_allscaling_stdinit_ours',
    'mup_adam_globaleps_allscaling_multfree', 'sp_adam_allscaling_multfree', 'mup_heur_adam_allscaling_multfree', 'sp_allscaling_multfree',
    # Bottleneck configs (ours + baselines)
    'mup_bottleneck_multfree', 'mup_bottleneck_ours', 'mup_bottleneck_stdinit',
    'mup_bottleneck_largeexpertin', 'mup_bottleneck_largeexpertin_allclt', 'mup_bottleneck_largeexpertin_sepagg',
    'mup_bottleneck_jiang_sgd', 'mup_bottleneck_heuristic_multfree',
    'mup_bottleneck_adam_multfree', 'mup_bottleneck_adam_stdinit_multfree',
    'mup_bottleneck_adam_globaleps_multfree', 'mup_adam_jiang_bottleneck_multfree'
]

def filter_configs_by_regime_and_optimizer(all_data, config_order=None):
    """Filter configs by scaling regime and optimizer type.

    Args:
        all_data: Dictionary of all loaded data
        config_order: Optional custom config order (defaults to global CONFIG_ORDER)

    Returns:
        Dictionary with keys:
            - 'fixed_expert_sgd': Fixed-E configs with SGD
            - 'fixed_expert_adam': Fixed-E configs with Adam
            - 'allscaling_sgd': All-scaling configs with SGD
            - 'allscaling_adam': All-scaling configs with Adam
            - 'bottleneck_sgd': Bottleneck configs with SGD
            - 'bottleneck_adam': Bottleneck configs with Adam
            - 'expert_scaling': Other configs (neither fixed_E, allscaling, nor bottleneck)
    """
    if config_order is None:
        config_order = CONFIG_ORDER

    return {
        'fixed_expert_sgd': [c for c in config_order if c in all_data and 'fixed_E' in c and get_optimizer_from_config(c) == 'sgd'],
        'fixed_expert_adam': [c for c in config_order if c in all_data and 'fixed_E' in c and get_optimizer_from_config(c) == 'adam'],
        'allscaling_sgd': [c for c in config_order if c in all_data and 'allscaling' in c and get_optimizer_from_config(c) == 'sgd'],
        'allscaling_adam': [c for c in config_order if c in all_data and 'allscaling' in c and get_optimizer_from_config(c) == 'adam'],
        'bottleneck_sgd': [c for c in config_order if c in all_data and 'bottleneck' in c and get_optimizer_from_config(c) == 'sgd'],
        'bottleneck_adam': [c for c in config_order if c in all_data and 'bottleneck' in c and get_optimizer_from_config(c) == 'adam'],
        'expert_scaling': [c for c in config_order if c in all_data and 'fixed_E' not in c and 'allscaling' not in c and 'bottleneck' not in c]
    }

def organize_configs_by_group(all_data, config_order=None):
    """Organize configs into simple regime groups (fixed_E, expert_scaling, allscaling, bottleneck).

    This is a simplified version of filter_configs_by_regime_and_optimizer that doesn't
    separate by optimizer type. Used by newer plotting functions.

    Args:
        all_data: Dictionary of all loaded data
        config_order: Optional custom config order (defaults to global CONFIG_ORDER)

    Returns:
        List of tuples: [(config_list, group_name), ...]
    """
    if config_order is None:
        config_order = CONFIG_ORDER

    # Include any config that is in all_data but not in config_order
    extra_configs = [c for c in all_data if c not in config_order]
    full_order = config_order + extra_configs

    # Organize into groups
    fixed_expert   = [c for c in full_order if c in all_data and 'fixed_E' in c]
    expert_scaling = [c for c in full_order if c in all_data and 'fixed_E' not in c and 'allscaling' not in c and 'bottleneck' not in c]
    allscaling     = [c for c in full_order if c in all_data and 'allscaling' in c]
    bottleneck     = [c for c in full_order if c in all_data and 'bottleneck' in c]

    return [
        (fixed_expert, 'fixed'),
        (expert_scaling, 'scaling'),
        (allscaling, 'allscaling'),
        (bottleneck, 'bottleneck')
    ]

def load_all_data(results_dir, separate_routing_and_init=False):
    """Load all results from all config directories, grouping by seed.

    Args:
        results_dir: Path to results directory
        separate_routing_and_init: If True, create separate "virtual configs" for
            different routing modes (soft/topk) and router initializations (zero/mup/etc).
            This allows plotting them as distinct configs for ablation studies.
    """
    results_path = Path(results_dir)
    all_data = {}

    # Reserved directory names that should not be treated as config directories
    RESERVED_DIRS = {'plots', 'rcc', 'figures', 'tables'}

    for config_dir in results_path.iterdir():
        if not config_dir.is_dir() or config_dir.name.startswith('.'):
            continue

        base_config_name = config_dir.name

        # Skip reserved directory names to prevent nested plots/plots/ structure
        if base_config_name in RESERVED_DIRS:
            continue

        stats_dir = config_dir / 'stats'
        if not stats_dir.exists():
            continue

        for f in glob.glob(str(stats_dir / 'nn_N*.npz')):
            basename = os.path.basename(f)
            # Extract N and seed from filename
            parts = basename.split('_')
            N = int(parts[1][1:])
            seed = None
            for part in parts:
                if part.startswith('seed'):
                    seed = int(part.replace('seed', '').replace('.npz', ''))
                    break
            if seed is None:
                seed = 42  # default

            data = np.load(f, allow_pickle=True)
            metadata = data['metadata'].item() if 'metadata' in data else {}
            config_args = metadata.get('args', {}) if isinstance(metadata, dict) else {}

            # Determine effective config name (with routing/init suffixes if requested)
            config_name = base_config_name
            if separate_routing_and_init:
                # Extract routing mode and router_init from filename or metadata
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
            
            # Initialize config entry if needed
            if config_name not in all_data:
                all_data[config_name] = {}
            if N not in all_data[config_name]:
                all_data[config_name][N] = {}

            all_data[config_name][N][seed] = {
                'effective_updates': data['effective_updates'].tolist() if 'effective_updates' in data else [],
                'propagating_updates': data['propagating_updates'].tolist() if 'propagating_updates' in data else [],
                'total_effective_updates_per_layer': data['total_effective_updates_per_layer'].tolist() if 'total_effective_updates_per_layer' in data else [],
                'grad_norms': data['grad_norms'].tolist() if 'grad_norms' in data else [],
                'weight_rms_norms': data['weight_rms_norms'].tolist() if 'weight_rms_norms' in data else [],
                'h_L_rms_diff': data['h_L_rms_diff'].tolist() if 'h_L_rms_diff' in data else [],
                'h_L_rms': data['h_L_rms'].tolist() if 'h_L_rms' in data else [],
                'activation_rms': data['activation_rms'].tolist() if 'activation_rms' in data else [],
                'output_grad_rms': data['output_grad_rms'].tolist() if 'output_grad_rms' in data else [],
                'losses': data['losses'].tolist() if 'losses' in data else [],
                'train_loss': data['train_loss'].tolist() if 'train_loss' in data else [],
                'train_acc':  data['train_acc'].tolist()  if 'train_acc'  in data else [],
                'psi_norm':   data['psi_norm'].tolist()   if 'psi_norm'   in data else [],
                'config': config_args
            }

            # Extract decomposition terms from nested dict structure
            # h^agg decomposition: hagg_decomposition is a list of dicts with keys {base, propagating, effective, cross}
            if 'hagg_decomposition' in data:
                hagg_data = data['hagg_decomposition'].tolist() if isinstance(data['hagg_decomposition'], np.ndarray) else data['hagg_decomposition']
                if hagg_data and len(hagg_data) > 0:
                    all_data[config_name][N][seed]['hagg_decomp_base'] = [d.get('base', np.nan) if isinstance(d, dict) else np.nan for d in hagg_data]
                    all_data[config_name][N][seed]['hagg_decomp_propagating'] = [d.get('propagating', np.nan) if isinstance(d, dict) else np.nan for d in hagg_data]
                    all_data[config_name][N][seed]['hagg_decomp_effective'] = [d.get('effective', np.nan) if isinstance(d, dict) else np.nan for d in hagg_data]
                    all_data[config_name][N][seed]['hagg_decomp_cross'] = [d.get('cross', np.nan) if isinstance(d, dict) else np.nan for d in hagg_data]
                else:
                    all_data[config_name][N][seed]['hagg_decomp_base'] = None
                    all_data[config_name][N][seed]['hagg_decomp_propagating'] = None
                    all_data[config_name][N][seed]['hagg_decomp_effective'] = None
                    all_data[config_name][N][seed]['hagg_decomp_cross'] = None
            else:
                # Fallback: try loading from old format (if exists)
                all_data[config_name][N][seed]['hagg_decomp_base'] = data['hagg_decomp_base'].tolist() if 'hagg_decomp_base' in data else None
                all_data[config_name][N][seed]['hagg_decomp_propagating'] = data['hagg_decomp_propagating'].tolist() if 'hagg_decomp_propagating' in data else None
                all_data[config_name][N][seed]['hagg_decomp_effective'] = data['hagg_decomp_effective'].tolist() if 'hagg_decomp_effective' in data else None
                all_data[config_name][N][seed]['hagg_decomp_cross'] = data['hagg_decomp_cross'].tolist() if 'hagg_decomp_cross' in data else None

            # Expert gradient to h^1 decomposition: expert_grad_h1_decomposition is a list of dicts
            if 'expert_grad_h1_decomposition' in data:
                grad_data = data['expert_grad_h1_decomposition'].tolist() if isinstance(data['expert_grad_h1_decomposition'], np.ndarray) else data['expert_grad_h1_decomposition']
                if grad_data and len(grad_data) > 0:
                    all_data[config_name][N][seed]['grad_h1_decomp_base'] = [d.get('base', np.nan) if isinstance(d, dict) else np.nan for d in grad_data]
                    all_data[config_name][N][seed]['grad_h1_decomp_propagating'] = [d.get('propagating', np.nan) if isinstance(d, dict) else np.nan for d in grad_data]
                    all_data[config_name][N][seed]['grad_h1_decomp_effective'] = [d.get('effective', np.nan) if isinstance(d, dict) else np.nan for d in grad_data]
                    all_data[config_name][N][seed]['grad_h1_decomp_cross'] = [d.get('cross', np.nan) if isinstance(d, dict) else np.nan for d in grad_data]
                else:
                    all_data[config_name][N][seed]['grad_h1_decomp_base'] = None
                    all_data[config_name][N][seed]['grad_h1_decomp_propagating'] = None
                    all_data[config_name][N][seed]['grad_h1_decomp_effective'] = None
                    all_data[config_name][N][seed]['grad_h1_decomp_cross'] = None
            else:
                # Fallback: try loading from old format (if exists)
                all_data[config_name][N][seed]['grad_h1_decomp_base'] = data['grad_h1_decomp_base'].tolist() if 'grad_h1_decomp_base' in data else None
                all_data[config_name][N][seed]['grad_h1_decomp_propagating'] = data['grad_h1_decomp_propagating'].tolist() if 'grad_h1_decomp_propagating' in data else None
                all_data[config_name][N][seed]['grad_h1_decomp_effective'] = data['grad_h1_decomp_effective'].tolist() if 'grad_h1_decomp_effective' in data else None
                all_data[config_name][N][seed]['grad_h1_decomp_cross'] = data['grad_h1_decomp_cross'].tolist() if 'grad_h1_decomp_cross' in data else None

            # Additional decomposition-related metrics
            all_data[config_name][N][seed]['loss_grad_rms'] = data['loss_grad_rms'].tolist() if 'loss_grad_rms' in data else None
            all_data[config_name][N][seed]['expert_grad_to_h1'] = data['expert_grad_to_h1'].tolist() if 'expert_grad_to_h1' in data else None
    
    return all_data

def plot_effective_updates(all_data, results_dir, T_total_override=None):
    """Plot effective updates for each layer"""
    # Get filtered config groups
    config_groups = filter_configs_by_regime_and_optimizer(all_data)
    fixed_expert_sgd = config_groups['fixed_expert_sgd']
    fixed_expert_adam = config_groups['fixed_expert_adam']
    allscaling_sgd = config_groups['allscaling_sgd']
    allscaling_adam = config_groups['allscaling_adam']
    bottleneck_sgd = config_groups['bottleneck_sgd']
    bottleneck_adam = config_groups['bottleneck_adam']
    expert_scaling = config_groups['expert_scaling']


    for config_group, group_name in [
        (fixed_expert_sgd, 'fixed_E_sgd'),
        (fixed_expert_adam, 'fixed_E_adam'),
        (expert_scaling, 'other'),
        (allscaling_sgd, 'allscaling_sgd'),
        (allscaling_adam, 'allscaling_adam'),
        (bottleneck_sgd, 'bottleneck_sgd'),
        (bottleneck_adam, 'bottleneck_adam')
    ]:
        print(f"Processing {group_name}: {config_group}")
        if not config_group:
            print(f"Skipping {group_name} - empty")
            continue
        
        # Get layer names from first config
        first_config = config_group[0]
        N_first = min(all_data[first_config].keys())
        first_seed = list(all_data[first_config][N_first].keys())[0]
        eff_upd = all_data[first_config][N_first][first_seed].get('effective_updates')
        if eff_upd is None or len(eff_upd) == 0:
            continue
        if isinstance(eff_upd, np.ndarray):
            eff_upd = eff_upd.tolist()
        if len(eff_upd) <= 1:
            continue
        
        sample_entry = eff_upd[1]
        is_nested = isinstance(sample_entry, dict) and 'raw' in sample_entry
        if is_nested:
            layer_names = list(sample_entry['raw'].keys())
            data_types = ['raw', 'normalized']
        else:
            layer_names = list(sample_entry.keys()) if isinstance(sample_entry, dict) else []
            data_types = ['raw']
        
        if not layer_names:
            continue
        
        # Plot for each layer and data type
        for layer_name in layer_names:
            for data_type in data_types:
                n_configs = len(config_group)
                
                if n_configs == 1:
                    # Single config: one plot with legend
                    fig, ax = plt.subplots(1, 1, figsize=onefigsize)
                    config = config_group[0]
                    Ns = sorted(all_data[config].keys())
                    first_seed = list(all_data[config][Ns[0]].keys())[0]
                    T_total = T_total_override if T_total_override is not None else all_data[config][Ns[0]][first_seed].get('config', {}).get('T', len(all_data[config][Ns[0]][first_seed].get('losses', [])))
                    timesteps = [2, 50, T_total//5, T_total-1] if T_total > 10 else list(range(1, T_total))
                    blues = plt.cm.Blues(np.linspace(0.4, 0.95, len(timesteps)))
                    
                    for t_idx, t in enumerate(timesteps):
                        seed_data_dict = {}
                        for N in Ns:
                            for seed, seed_data in all_data[config][N].items():
                                eff_upd_N = seed_data.get('effective_updates')
                                if eff_upd_N is None:
                                    continue
                                if isinstance(eff_upd_N, np.ndarray):
                                    eff_upd_N = eff_upd_N.tolist()
                                if t < len(eff_upd_N):
                                    update_dict = eff_upd_N[t]
                                    if is_nested and data_type in update_dict:
                                        update_dict = update_dict[data_type]
                                    if isinstance(update_dict, dict) and layer_name in update_dict:
                                        if seed not in seed_data_dict:
                                            seed_data_dict[seed] = {}
                                        seed_data_dict[seed][N] = update_dict[layer_name]
                        
                        from scipy import stats
                        for seed, seed_points in seed_data_dict.items():
                            if len(seed_points) >= 2:
                                widths = np.array(list(seed_points.keys()))
                                values = np.array(list(seed_points.values()))
                                ax.plot(widths, values, 'o', color=blues[t_idx], markersize=4, alpha=0.5)
                                log_widths, log_values = np.log10(widths), np.log10(values)
                                try:
                                    slope, intercept, _, _, _ = stats.linregress(log_widths, log_values)
                                    x_fit = np.logspace(np.log10(min(widths)), np.log10(max(widths)), 100)
                                    ax.plot(x_fit, 10**intercept * x_fit**slope, '-', color=blues[t_idx], alpha=0.5, linewidth=1.5)
                                except:
                                    pass
                        
                        if seed_data_dict:
                            all_widths = sorted(set(N for seed_points in seed_data_dict.values() for N in seed_points.keys()))
                            mean_values = []
                            for N in all_widths:
                                vals = [seed_points[N] for seed_points in seed_data_dict.values() if N in seed_points]
                                if vals:
                                    mean_values.append(np.mean(vals))
                            
                            if len(all_widths) >= 2:
                                widths_arr = np.array(all_widths)
                                values_arr = np.array(mean_values)
                                log_widths = np.log10(widths_arr)
                                log_values = np.log10(values_arr)
                                try:
                                    slope, intercept, _, _, _ = stats.linregress(log_widths, log_values)
                                    alpha_fit = np.round(slope, 2)
                                    C_fit = 10**intercept
                                    x_fit = np.logspace(np.log10(min(all_widths)), np.log10(max(all_widths)), 100)
                                    ax.plot(x_fit, C_fit * x_fit**alpha_fit, '-', color=blues[t_idx], label=f't={t} ({alpha_fit:.2f})', linewidth=2)
                                except:
                                    pass
                    
                    ax.set_xscale('log'); ax.set_yscale('log')
                    ax.set_xlabel('Width N')
                    if data_type == 'normalized':
                        ax.set_ylabel(r'$||\Delta W_t (x_t/||x_t||_{\mathrm{RMS}})||_{\mathrm{RMS}}$')
                    else:
                        ax.set_ylabel(r'$||\Delta W_t x_t||_{\mathrm{RMS}}$')
                    ax.set_title(format_subplot_title(config))
                    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
                
                else:
                    # Multiple configs: subplots
                    fig, axes = plt.subplots(1, n_configs, figsize=(n_configs*onefigsize[0], onefigsize[1]), sharey=True)
                    
                    for ax_idx, config in enumerate(config_group):
                        ax = axes[ax_idx]
                        Ns = sorted(all_data[config].keys())
                        first_seed = list(all_data[config][Ns[0]].keys())[0]
                        T_total = T_total_override if T_total_override is not None else all_data[config][Ns[0]][first_seed].get('config', {}).get('T', len(all_data[config][Ns[0]][first_seed].get('losses', [])))
                        timesteps = [2, 50, T_total//5, T_total-1] if T_total > 10 else list(range(1, T_total))
                        blues = plt.cm.Blues(np.linspace(0.4, 0.95, len(timesteps)))
                        
                        for t_idx, t in enumerate(timesteps):
                            seed_data_dict = {}
                            for N in Ns:
                                for seed, seed_data in all_data[config][N].items():
                                    eff_upd_N = seed_data.get('effective_updates')
                                    if eff_upd_N is None:
                                        continue
                                    if isinstance(eff_upd_N, np.ndarray):
                                        eff_upd_N = eff_upd_N.tolist()
                                    if t < len(eff_upd_N):
                                        update_dict = eff_upd_N[t]
                                        if is_nested and data_type in update_dict:
                                            update_dict = update_dict[data_type]
                                        if isinstance(update_dict, dict) and layer_name in update_dict:
                                            if seed not in seed_data_dict:
                                                seed_data_dict[seed] = {}
                                            seed_data_dict[seed][N] = update_dict[layer_name]
                            
                            from scipy import stats
                            for seed, seed_points in seed_data_dict.items():
                                if len(seed_points) >= 2:
                                    widths = np.array(list(seed_points.keys()))
                                    values = np.array(list(seed_points.values()))
                                    ax.plot(widths, values, 'o', color=blues[t_idx], markersize=4, alpha=0.5)
                                    log_widths, log_values = np.log10(widths), np.log10(values)
                                    try:
                                        slope, intercept, _, _, _ = stats.linregress(log_widths, log_values)
                                        x_fit = np.logspace(np.log10(min(widths)), np.log10(max(widths)), 100)
                                        ax.plot(x_fit, 10**intercept * x_fit**slope, '-', color=blues[t_idx], alpha=0.5, linewidth=1.5)
                                    except:
                                        pass
                            
                            if seed_data_dict:
                                all_widths = sorted(set(N for seed_points in seed_data_dict.values() for N in seed_points.keys()))
                                mean_values = []
                                for N in all_widths:
                                    vals = [seed_points[N] for seed_points in seed_data_dict.values() if N in seed_points]
                                    if vals:
                                        mean_values.append(np.mean(vals))
                                
                                if len(all_widths) >= 2:
                                    widths_arr = np.array(all_widths)
                                    values_arr = np.array(mean_values)
                                    log_widths = np.log10(widths_arr)
                                    log_values = np.log10(values_arr)
                                    try:
                                        slope, intercept, _, _, _ = stats.linregress(log_widths, log_values)
                                        alpha_fit = np.round(slope, 2)
                                        C_fit = 10**intercept
                                        x_fit = np.logspace(np.log10(min(all_widths)), np.log10(max(all_widths)), 100)
                                        label = f't={t} ({alpha_fit:.2f})' if ax_idx == n_configs - 1 else None
                                        ax.plot(x_fit, C_fit * x_fit**alpha_fit, '-', color=blues[t_idx], label=label, linewidth=2)
                                    except:
                                        pass
                        
                        ax.set_xscale('log'); ax.set_yscale('log')
                        ax.set_xlabel('Width N')
                        if ax_idx == 0:
                            if data_type == 'normalized':
                                ax.set_ylabel(r'$||\Delta W_t (x_t/||x_t||_{\mathrm{RMS}})||_{\mathrm{RMS}}$')
                            else:
                                ax.set_ylabel(r'$||\Delta W_t x_t||_{\mathrm{RMS}}$')
                        ax.set_title(format_subplot_title(config))
                        if ax_idx == n_configs - 1:
                            ax.legend(fontsize=8, loc='best')
                        ax.grid(True, alpha=0.3)

                    # Set shared ylims across all subplots
                    set_shared_ylim(axes)

                # Determine config name for single-config plots
                single_config_name = config_group[0] if n_configs == 1 else None

                plt.tight_layout()
                save_path = get_plot_save_path(results_dir, single_config_name,
                                               f'coordinate_updates_{layer_name}_{data_type}_{group_name}.png',
                                               is_joint_plot=(n_configs > 1))
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()

def plot_total_effective_updates_per_layer(all_data, results_dir, T_total_override=None):
    """Plot total effective updates per layer for each layer"""
    # Get filtered config groups
    config_groups = filter_configs_by_regime_and_optimizer(all_data)
    fixed_expert_sgd = config_groups['fixed_expert_sgd']
    fixed_expert_adam = config_groups['fixed_expert_adam']
    allscaling_sgd = config_groups['allscaling_sgd']
    allscaling_adam = config_groups['allscaling_adam']
    bottleneck_sgd = config_groups['bottleneck_sgd']
    bottleneck_adam = config_groups['bottleneck_adam']
    expert_scaling = config_groups['expert_scaling']


    for config_group, group_name in [
        (fixed_expert_sgd, 'fixed_E_sgd'),
        (fixed_expert_adam, 'fixed_E_adam'),
        (expert_scaling, 'other'),
        (allscaling_sgd, 'allscaling_sgd'),
        (allscaling_adam, 'allscaling_adam'),
        (bottleneck_sgd, 'bottleneck_sgd'),
        (bottleneck_adam, 'bottleneck_adam')
    ]:
        print(f"Processing {group_name}: {config_group}")
        if not config_group:
            print(f"Skipping {group_name} - empty")
            continue
        
        # Get layer names from first config
        first_config = config_group[0]
        N_first = min(all_data[first_config].keys())
        first_seed = list(all_data[first_config][N_first].keys())[0]
        eff_upd = all_data[first_config][N_first][first_seed].get('total_effective_updates_per_layer')
        if eff_upd is None or len(eff_upd) == 0:
            continue
        if isinstance(eff_upd, np.ndarray):
            eff_upd = eff_upd.tolist()
        if len(eff_upd) <= 1:
            continue
        
        sample_entry = eff_upd[1]
        is_nested = isinstance(sample_entry, dict) and 'raw' in sample_entry
        if is_nested:
            layer_names = list(sample_entry['raw'].keys())
            data_types = ['raw', 'normalized']
        else:
            layer_names = list(sample_entry.keys()) if isinstance(sample_entry, dict) else []
            data_types = ['raw']
        
        if not layer_names:
            continue
        
        # Plot for each layer and data type
        for layer_name in layer_names:
            for data_type in data_types:
                n_configs = len(config_group)
                fig, axes = plt.subplots(1, n_configs, figsize=(n_configs*onefigsize[0], onefigsize[1]), sharey=True)
                if n_configs == 1:
                    axes = [axes]
                
                for ax_idx, config in enumerate(config_group):
                    ax = axes[ax_idx]
                    Ns = sorted(all_data[config].keys())
                    first_seed = list(all_data[config][Ns[0]].keys())[0]
                    T_total = T_total_override if T_total_override is not None else all_data[config][Ns[0]][first_seed].get('config', {}).get('T', len(all_data[config][Ns[0]][first_seed].get('losses', [])))
                    timesteps = [2, 50, T_total//5, T_total-1] if T_total > 10 else list(range(1, T_total))
                    blues = plt.cm.Blues(np.linspace(0.4, 0.95, len(timesteps)))
                    
                    for t_idx, t in enumerate(timesteps):
                        # Collect all seed data
                        seed_data_dict = {}  # {seed: {N: value}}
                        for N in Ns:
                            for seed, seed_data in all_data[config][N].items():
                                eff_upd_N = seed_data.get('total_effective_updates_per_layer')
                                if eff_upd_N is None:
                                    continue
                                if isinstance(eff_upd_N, np.ndarray):
                                    eff_upd_N = eff_upd_N.tolist()
                                if t < len(eff_upd_N):
                                    update_dict = eff_upd_N[t]
                                    if is_nested and data_type in update_dict:
                                        update_dict = update_dict[data_type]
                                    if isinstance(update_dict, dict) and layer_name in update_dict:
                                        if seed not in seed_data_dict:
                                            seed_data_dict[seed] = {}
                                        seed_data_dict[seed][N] = update_dict[layer_name]
                        
                        # Plot individual seeds with alpha=0.5
                        from scipy import stats
                        for seed, seed_points in seed_data_dict.items():
                            if len(seed_points) >= 2:
                                widths = np.array(list(seed_points.keys()))
                                values = np.array(list(seed_points.values()))
                                # Plot points
                                ax.plot(widths, values, 'o', color=blues[t_idx], markersize=4, alpha=0.5)
                                # Fit and plot line
                                log_widths, log_values = np.log10(widths), np.log10(values)
                                try:
                                    slope, intercept, _, _, _ = stats.linregress(log_widths, log_values)
                                    x_fit = np.logspace(np.log10(min(widths)), np.log10(max(widths)), 100)
                                    ax.plot(x_fit, 10**intercept * x_fit**slope, '-', color=blues[t_idx], 
                                           alpha=0.5, linewidth=1.5)
                                except:
                                    pass
                        
                        # Compute and plot mean fit
                        if seed_data_dict:
                            all_widths = sorted(set(N for seed_points in seed_data_dict.values() for N in seed_points.keys()))
                            mean_values = []
                            for N in all_widths:
                                vals = [seed_points[N] for seed_points in seed_data_dict.values() if N in seed_points]
                                if vals:
                                    mean_values.append(np.mean(vals))
                            
                            if len(all_widths) >= 2:
                                widths_arr = np.array(all_widths)
                                values_arr = np.array(mean_values)
                                log_widths = np.log10(widths_arr)
                                log_values = np.log10(values_arr)
                                try:
                                    slope, intercept, _, _, _ = stats.linregress(log_widths, log_values)
                                    alpha_fit = np.round(slope, 2)
                                    C_fit = 10**intercept
                                    x_fit = np.logspace(np.log10(min(all_widths)), np.log10(max(all_widths)), 100)
                                    ax.plot(x_fit, C_fit * x_fit**alpha_fit, '-', color=blues[t_idx], 
                                           label=f't={t} ({alpha_fit:.2f})', linewidth=2)
                                except:
                                    pass
                    
                    ax.set_xscale('log'); ax.set_yscale('log')
                    ax.set_xlabel('Width N')
                    if data_type == 'normalized':
                        ax.set_ylabel(r'$||\Delta W_t (x_t/||x_t||_{\mathrm{RMS}})||_{\mathrm{RMS}}$')
                    else:
                        ax.set_ylabel(r'$||\Delta W_t x_t||_{\mathrm{RMS}}$')
                    ax.set_title(format_subplot_title(config))
                    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

                # Set shared ylims for all subplots
                set_shared_ylim(axes)

                # Determine config name for single-config plots
                single_config_name = config_group[0] if n_configs == 1 else None

                plt.tight_layout()
                save_path = get_plot_save_path(results_dir, single_config_name,
                                               f'total_effective_updates_per_layer_{layer_name}_{data_type}_{group_name}.png',
                                               is_joint_plot=(n_configs > 1))
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()


def plot_gradient_norms(all_data, results_dir, T_total_override=None):
    """Plot gradient RMS norms for each layer"""
    # Get filtered config groups
    config_groups = filter_configs_by_regime_and_optimizer(all_data)
    fixed_expert_sgd = config_groups['fixed_expert_sgd']
    fixed_expert_adam = config_groups['fixed_expert_adam']
    allscaling_sgd = config_groups['allscaling_sgd']
    allscaling_adam = config_groups['allscaling_adam']
    bottleneck_sgd = config_groups['bottleneck_sgd']
    bottleneck_adam = config_groups['bottleneck_adam']
    expert_scaling = config_groups['expert_scaling']

    for config_group, group_name in [
        (fixed_expert_sgd, 'fixed_E_sgd'),
        (fixed_expert_adam, 'fixed_E_adam'),
        (expert_scaling, 'other'),
        (allscaling_sgd, 'allscaling_sgd'),
        (allscaling_adam, 'allscaling_adam'),
        (bottleneck_sgd, 'bottleneck_sgd'),
        (bottleneck_adam, 'bottleneck_adam')
    ]:

        if not config_group:
            continue
        
        first_config = config_group[0]
        N_first = min(all_data[first_config].keys())
        first_seed = list(all_data[first_config][N_first].keys())[0]
        grad_norms = all_data[first_config][N_first][first_seed].get('grad_norms')
        if grad_norms is None or len(grad_norms) == 0:
            continue
        if isinstance(grad_norms, np.ndarray):
            grad_norms = grad_norms.tolist()
        if len(grad_norms) <= 1:
            continue
        
        layer_names = list(grad_norms[1].keys()) if isinstance(grad_norms[1], dict) else []
        if not layer_names:
            continue
        
        for layer_name in layer_names:
            n_configs = len(config_group)
            fig, axes = plt.subplots(1, n_configs, figsize=(n_configs*onefigsize[0], onefigsize[1]), sharey=True)
            if n_configs == 1:
                axes = [axes]
            
            for ax_idx, config in enumerate(config_group):
                ax = axes[ax_idx]
                Ns = sorted(all_data[config].keys())
                first_seed = list(all_data[config][Ns[0]].keys())[0]
                T_total = T_total_override if T_total_override is not None else all_data[config][Ns[0]][first_seed].get('config', {}).get('T', len(all_data[config][Ns[0]][first_seed].get('losses', [])))
                timesteps = [2, 50, T_total//5, T_total-1] if T_total > 10 else list(range(1, T_total))
                blues = plt.cm.Blues(np.linspace(0.4, 0.95, len(timesteps)))
                
                for t_idx, t in enumerate(timesteps):
                    # Collect all seed data
                    seed_data_dict = {}  # {seed: {N: value}}
                    for N in Ns:
                        for seed, seed_data in all_data[config][N].items():
                            grad_N = seed_data.get('grad_norms')
                            if grad_N is None:
                                continue
                            if isinstance(grad_N, np.ndarray):
                                grad_N = grad_N.tolist()
                            if t < len(grad_N) and isinstance(grad_N[t], dict) and layer_name in grad_N[t]:
                                if seed not in seed_data_dict:
                                    seed_data_dict[seed] = {}
                                seed_data_dict[seed][N] = grad_N[t][layer_name]
                    
                    # Plot individual seeds with alpha=0.5
                    from scipy import stats
                    for seed, seed_points in seed_data_dict.items():
                        if len(seed_points) >= 2:
                            widths = np.array(list(seed_points.keys()))
                            values = np.array(list(seed_points.values()))
                            # Plot points
                            ax.plot(widths, values, 'o', color=blues[t_idx], markersize=4, alpha=0.5)
                            # Fit and plot line
                            log_widths, log_values = np.log10(widths), np.log10(values)
                            try:
                                slope, intercept, _, _, _ = stats.linregress(log_widths, log_values)
                                x_fit = np.logspace(np.log10(min(widths)), np.log10(max(widths)), 100)
                                ax.plot(x_fit, 10**intercept * x_fit**slope, '-', color=blues[t_idx], 
                                       alpha=0.5, linewidth=1.5)
                            except:
                                pass
                    
                    # Compute and plot mean fit
                    if seed_data_dict:
                        all_widths = sorted(set(N for seed_points in seed_data_dict.values() for N in seed_points.keys()))
                        mean_values = []
                        for N in all_widths:
                            vals = [seed_points[N] for seed_points in seed_data_dict.values() if N in seed_points]
                            if vals:
                                mean_values.append(np.mean(vals))
                        
                        if len(all_widths) >= 2:
                            widths_arr = np.array(all_widths)
                            values_arr = np.array(mean_values)
                            log_widths = np.log10(widths_arr)
                            log_values = np.log10(values_arr)
                            try:
                                slope, intercept, _, _, _ = stats.linregress(log_widths, log_values)
                                alpha_fit = np.round(slope, 2)
                                C_fit = 10**intercept
                                x_fit = np.logspace(np.log10(min(all_widths)), np.log10(max(all_widths)), 100)
                                ax.plot(x_fit, C_fit * x_fit**alpha_fit, '-', color=blues[t_idx], 
                                       label=f't={t} ({alpha_fit:.2f})', linewidth=2)
                            except:
                                pass
                
                ax.set_xscale('log'); ax.set_yscale('log')
                ax.set_xlabel('Width N')
                ax.set_ylabel(r'$||\nabla_{' + layer_name.replace('_', r'\_') + r'}||_{\mathrm{RMS}}$')
                ax.set_title(format_subplot_title(config))
                ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

            # Set shared ylims for all subplots
            set_shared_ylim(axes)

            # Determine config name for single-config plots
            single_config_name = config_group[0] if n_configs == 1 else None

            plt.tight_layout()
            save_path = get_plot_save_path(results_dir, single_config_name,
                                           f'gradient_norms_{layer_name}_{group_name}.png',
                                           is_joint_plot=(n_configs > 1))
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

def plot_propagating_updates(all_data, results_dir, T_total_override=None):
    """Plot propagating updates W_0 Δx for each layer"""
    # Get filtered config groups
    config_groups = filter_configs_by_regime_and_optimizer(all_data)
    fixed_expert_sgd = config_groups['fixed_expert_sgd']
    fixed_expert_adam = config_groups['fixed_expert_adam']
    allscaling_sgd = config_groups['allscaling_sgd']
    allscaling_adam = config_groups['allscaling_adam']
    bottleneck_sgd = config_groups['bottleneck_sgd']
    bottleneck_adam = config_groups['bottleneck_adam']
    expert_scaling = config_groups['expert_scaling']


    for config_group, group_name in [
        (fixed_expert_sgd, 'fixed_E_sgd'),
        (fixed_expert_adam, 'fixed_E_adam'),
        (expert_scaling, 'other'),
        (allscaling_sgd, 'allscaling_sgd'),
        (allscaling_adam, 'allscaling_adam'),
        (bottleneck_sgd, 'bottleneck_sgd'),
        (bottleneck_adam, 'bottleneck_adam')
    ]:
        print(f"Processing {group_name}: {config_group}")
        if not config_group:
            print(f"Skipping {group_name} - empty")
            continue
        
        # Get layer names from first config
        first_config = config_group[0]
        N_first = min(all_data[first_config].keys())
        first_seed = list(all_data[first_config][N_first].keys())[0]
        prop_upd = all_data[first_config][N_first][first_seed].get('propagating_updates')
        if prop_upd is None or len(prop_upd) == 0:
            continue
        if isinstance(prop_upd, np.ndarray):
            prop_upd = prop_upd.tolist()
        if len(prop_upd) <= 1:
            continue
        
        sample_entry = prop_upd[1]
        is_nested = isinstance(sample_entry, dict) and 'raw' in sample_entry
        if is_nested:
            layer_names = list(sample_entry['raw'].keys())
            data_types = ['raw', 'normalized']
        else:
            layer_names = list(sample_entry.keys()) if isinstance(sample_entry, dict) else []
            data_types = ['raw']
        
        if not layer_names:
            continue
        
        # Plot for each layer and data type
        for layer_name in layer_names:
            for data_type in data_types:
                n_configs = len(config_group)
                fig, axes = plt.subplots(1, n_configs, figsize=(n_configs*onefigsize[0], onefigsize[1]), sharey=True)
                if n_configs == 1:
                    axes = [axes]
                
                for ax_idx, config in enumerate(config_group):
                    ax = axes[ax_idx]
                    Ns = sorted(all_data[config].keys())
                    first_seed = list(all_data[config][Ns[0]].keys())[0]
                    T_total = T_total_override if T_total_override is not None else all_data[config][Ns[0]][first_seed].get('config', {}).get('T', len(all_data[config][Ns[0]][first_seed].get('losses', [])))
                    timesteps = [2, 50, T_total//5, T_total-1] if T_total > 10 else list(range(1, T_total))
                    blues = plt.cm.Blues(np.linspace(0.4, 0.95, len(timesteps)))
                    
                    for t_idx, t in enumerate(timesteps):
                        # Collect all seed data
                        seed_data_dict = {}  # {seed: {N: value}}
                        for N in Ns:
                            for seed, seed_data in all_data[config][N].items():
                                prop_upd_N = seed_data.get('propagating_updates')
                                if prop_upd_N is None:
                                    continue
                                if isinstance(prop_upd_N, np.ndarray):
                                    prop_upd_N = prop_upd_N.tolist()
                                if t < len(prop_upd_N):
                                    update_dict = prop_upd_N[t]
                                    if is_nested and data_type in update_dict:
                                        update_dict = update_dict[data_type]
                                    if isinstance(update_dict, dict) and layer_name in update_dict:
                                        if seed not in seed_data_dict:
                                            seed_data_dict[seed] = {}
                                        seed_data_dict[seed][N] = update_dict[layer_name]
                        
                        # Plot individual seeds with alpha=0.5
                        from scipy import stats
                        for seed, seed_points in seed_data_dict.items():
                            if len(seed_points) >= 2:
                                widths = np.array(list(seed_points.keys()))
                                values = np.array(list(seed_points.values()))
                                # Plot points
                                ax.plot(widths, values, 'o', color=blues[t_idx], markersize=4, alpha=0.5)
                                # Fit and plot line
                                log_widths, log_values = np.log10(widths), np.log10(values)
                                try:
                                    slope, intercept, _, _, _ = stats.linregress(log_widths, log_values)
                                    x_fit = np.logspace(np.log10(min(widths)), np.log10(max(widths)), 100)
                                    ax.plot(x_fit, 10**intercept * x_fit**slope, '-', color=blues[t_idx], 
                                           alpha=0.5, linewidth=1.5)
                                except:
                                    pass
                        
                        # Compute and plot mean fit
                        if seed_data_dict:
                            all_widths = sorted(set(N for seed_points in seed_data_dict.values() for N in seed_points.keys()))
                            mean_values = []
                            for N in all_widths:
                                vals = [seed_points[N] for seed_points in seed_data_dict.values() if N in seed_points]
                                if vals:
                                    mean_values.append(np.mean(vals))
                            
                            if len(all_widths) >= 2:
                                widths_arr = np.array(all_widths)
                                values_arr = np.array(mean_values)
                                log_widths = np.log10(widths_arr)
                                log_values = np.log10(values_arr)
                                try:
                                    slope, intercept, _, _, _ = stats.linregress(log_widths, log_values)
                                    alpha_fit = np.round(slope, 2)
                                    C_fit = 10**intercept
                                    x_fit = np.logspace(np.log10(min(all_widths)), np.log10(max(all_widths)), 100)
                                    ax.plot(x_fit, C_fit * x_fit**alpha_fit, '-', color=blues[t_idx], 
                                           label=f't={t} ({alpha_fit:.2f})', linewidth=2)
                                except:
                                    pass
                    
                    ax.set_xscale('log'); ax.set_yscale('log')
                    ax.set_xlabel('Width N')
                    if data_type == 'normalized':
                        ax.set_ylabel(r'$||W_0 (\Delta x_t/||\Delta x_t||_{\mathrm{RMS}})||_{\mathrm{RMS}}$')
                    else:
                        ax.set_ylabel(r'$||W_0 \Delta x_t||_{\mathrm{RMS}}$')
                    ax.set_title(format_subplot_title(config))
                    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

                # Set shared ylims for all subplots
                set_shared_ylim(axes)

                # Determine config name for single-config plots
                single_config_name = config_group[0] if n_configs == 1 else None

                plt.tight_layout()
                save_path = get_plot_save_path(results_dir, single_config_name,
                                               f'coordinate_propagating_{layer_name}_{data_type}_{group_name}.png',
                                               is_joint_plot=(n_configs > 1))
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()


def plot_weight_rms_norms(all_data, results_dir, T_total_override=None):
    """Plot weight RMS norms for each layer"""
    # Get filtered config groups
    config_groups = filter_configs_by_regime_and_optimizer(all_data)
    fixed_expert_sgd = config_groups['fixed_expert_sgd']
    fixed_expert_adam = config_groups['fixed_expert_adam']
    allscaling_sgd = config_groups['allscaling_sgd']
    allscaling_adam = config_groups['allscaling_adam']
    bottleneck_sgd = config_groups['bottleneck_sgd']
    bottleneck_adam = config_groups['bottleneck_adam']
    expert_scaling = config_groups['expert_scaling']

    for config_group, group_name in [
        (fixed_expert_sgd, 'fixed_E_sgd'),
        (fixed_expert_adam, 'fixed_E_adam'),
        (expert_scaling, 'other'),
        (allscaling_sgd, 'allscaling_sgd'),
        (allscaling_adam, 'allscaling_adam'),
        (bottleneck_sgd, 'bottleneck_sgd'),
        (bottleneck_adam, 'bottleneck_adam')
    ]:
        if not config_group:
            continue

        first_config = config_group[0]
        N_first = min(all_data[first_config].keys())
        first_seed = list(all_data[first_config][N_first].keys())[0]
        weight_norms = all_data[first_config][N_first][first_seed].get('weight_rms_norms')
        if weight_norms is None or len(weight_norms) == 0:
            continue
        if isinstance(weight_norms, np.ndarray):
            weight_norms = weight_norms.tolist()
        if len(weight_norms) <= 1:
            continue

        layer_names = list(weight_norms[1].keys()) if isinstance(weight_norms[1], dict) else []
        # Filter out per-expert stats (mean/std) to get only main layer names
        layer_names = [ln for ln in layer_names if not (ln.endswith('_mean') or ln.endswith('_std'))]
        if not layer_names:
            continue

        for layer_name in layer_names:
            n_configs = len(config_group)
            fig, axes = plt.subplots(1, n_configs, figsize=(n_configs*onefigsize[0], onefigsize[1]), sharey=True)
            if n_configs == 1:
                axes = [axes]

            for ax_idx, config in enumerate(config_group):
                ax = axes[ax_idx]
                Ns = sorted(all_data[config].keys())
                first_seed = list(all_data[config][Ns[0]].keys())[0]
                T_total = T_total_override if T_total_override is not None else all_data[config][Ns[0]][first_seed].get('config', {}).get('T', len(all_data[config][Ns[0]][first_seed].get('losses', [])))
                timesteps = [2, 50, T_total//5, T_total-1] if T_total > 10 else list(range(1, T_total))
                blues = plt.cm.Blues(np.linspace(0.4, 0.95, len(timesteps)))

                for t_idx, t in enumerate(timesteps):
                    # Collect all seed data
                    seed_data_dict = {}  # {seed: {N: value}}
                    for N in Ns:
                        for seed, seed_data in all_data[config][N].items():
                            weight_N = seed_data.get('weight_rms_norms')
                            if weight_N is None:
                                continue
                            if isinstance(weight_N, np.ndarray):
                                weight_N = weight_N.tolist()
                            if t < len(weight_N) and isinstance(weight_N[t], dict) and layer_name in weight_N[t]:
                                if seed not in seed_data_dict:
                                    seed_data_dict[seed] = {}
                                seed_data_dict[seed][N] = weight_N[t][layer_name]

                    # Plot individual seeds with alpha=0.5
                    from scipy import stats
                    for seed, seed_points in seed_data_dict.items():
                        if len(seed_points) >= 2:
                            widths = np.array(list(seed_points.keys()))
                            values = np.array(list(seed_points.values()))
                            # Plot points
                            ax.plot(widths, values, 'o', color=blues[t_idx], markersize=4, alpha=0.5)
                            # Fit and plot line
                            log_widths, log_values = np.log10(widths), np.log10(values)
                            try:
                                slope, intercept, _, _, _ = stats.linregress(log_widths, log_values)
                                x_fit = np.logspace(np.log10(min(widths)), np.log10(max(widths)), 100)
                                ax.plot(x_fit, 10**intercept * x_fit**slope, '-', color=blues[t_idx],
                                       alpha=0.5, linewidth=1.5)
                            except:
                                pass

                    # Compute and plot mean fit
                    if seed_data_dict:
                        all_widths = sorted(set(N for seed_points in seed_data_dict.values() for N in seed_points.keys()))
                        mean_values = []
                        for N in all_widths:
                            vals = [seed_points[N] for seed_points in seed_data_dict.values() if N in seed_points]
                            if vals:
                                mean_values.append(np.mean(vals))

                        if len(all_widths) >= 2:
                            widths_arr = np.array(all_widths)
                            values_arr = np.array(mean_values)
                            log_widths = np.log10(widths_arr)
                            log_values = np.log10(values_arr)
                            try:
                                slope, intercept, _, _, _ = stats.linregress(log_widths, log_values)
                                alpha_fit = np.round(slope, 2)
                                C_fit = 10**intercept
                                x_fit = np.logspace(np.log10(min(all_widths)), np.log10(max(all_widths)), 100)
                                ax.plot(x_fit, C_fit * x_fit**alpha_fit, '-', color=blues[t_idx],
                                       label=f't={t} ({alpha_fit:.2f})', linewidth=2)
                            except:
                                pass

                ax.set_xscale('log'); ax.set_yscale('log')
                ax.set_xlabel('Width N')
                ax.set_ylabel(r'$||' + layer_name.replace('_', r'\_') + r'||_{\mathrm{RMS}}$')
                ax.set_title(format_subplot_title(config))
                ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

            # Set shared ylims for all subplots
            set_shared_ylim(axes)

            # Determine config name for single-config plots
            single_config_name = config_group[0] if n_configs == 1 else None

            plt.tight_layout()
            save_path = get_plot_save_path(results_dir, single_config_name,
                                           f'weight_rms_norms_{layer_name}_{group_name}.png',
                                           is_joint_plot=(n_configs > 1))
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()


def compute_exponent_table(all_data, results_dir, timesteps=None):
    """Compute exponent tables for specified timesteps.
    
    Args:
        timesteps: List of integer timesteps (e.g., [2, 3, 50]). T-1 is always appended if not present.
    """
    """Compute exponents at t=2 and t=T-1 and save as tables."""
    from scipy import stats
    import pandas as pd
    
    if not all_data:
        print("No data found, skipping exponent tables")
        return
    
    row_order = [
        'fixed_E_ntp', 'fixed_E_mup', 'fixed_E_mup_multfree', 'fixed_E_mup_largerouterlr', 'fixed_E_mup_largerouterlr_nonasympQ', 'fixed_E_mup_adam',
        'ntp', 'mup_largerouter', 'mup_smallrouter', 'mup_smallrouter_singlelr',
        'ntp_allscaling', 'mup_allscaling', 'mup_allscaling_nonasympQ', 'mup_clt_smallinlr_allscaling_multfree', 'mup_lln_allscaling_multfree', 'mup_smallinputlr_allscaling_multfree', 'mup_heuristic_allscaling_multfree', 'mup_adam_allscaling_ours', 'mup_adam_globaleps_allscaling_multfree', 'sp_adam_allscaling_multfree', 'mup_heur_adam_allscaling_multfree', 'sp_allscaling_multfree'
    ]
    
    layer_base_order = ['W_in', 'Q', 'W_exp1', 'W_exp2', 'W_out']
    col_order = [f"{layer}_normalized" for layer in layer_base_order] + [f"{layer}_raw" for layer in layer_base_order]

    for update_type in ['effective_updates', 'propagating_updates', 'total_effective_updates_per_layer', 'grad_norms']:
        # Get T from first config
        first_config = list(all_data.keys())[0]
        first_N = min(all_data[first_config].keys())
        first_seed = list(all_data[first_config][first_N].keys())[0]
        T_total = all_data[first_config][first_N][first_seed].get('config', {}).get('T', len(all_data[first_config][first_N][first_seed].get('losses', [])))
        
        # Use provided timesteps or default to [2, 3, 50]
        if timesteps is None:
            timesteps_to_generate = [2, 50, 200]
        else:
            timesteps_to_generate = list(timesteps)
        
        # Always append T-1 if not already present
        if T_total - 1 not in timesteps_to_generate:
            timesteps_to_generate.append(T_total - 1)
        
        # Generate CSVs for requested timesteps
        for t_val in timesteps_to_generate:
            # Skip if t_val >= T_total
            if t_val >= T_total:
                print(f"Skipping t={t_val} (>= T_total={T_total})")
                continue
            
            t_func = lambda T, tv=t_val: min(tv, T-1)
            exponent_data = {}
            
            for config in all_data.keys():
                exponent_data[config] = {}
                Ns = sorted(all_data[config].keys())
                if len(Ns) < 2:
                    continue
                
                N_first = Ns[0]
                # Get first seed's data to check structure
                first_seed = list(all_data[config][N_first].keys())[0]
                updates = all_data[config][N_first][first_seed].get(update_type)
                if updates is None or len(updates) == 0:
                    continue
                if isinstance(updates, np.ndarray):
                    updates = updates.tolist()
                if len(updates) <= 1:
                    continue
                
                sample_entry = updates[1]
                is_nested = isinstance(sample_entry, dict) and 'raw' in sample_entry
                if is_nested:
                    layer_names = list(sample_entry['raw'].keys())
                    data_types = ['raw', 'normalized']
                elif isinstance(sample_entry, dict):
                    layer_names = list(sample_entry.keys())
                    data_types = ['raw']
                else:
                    layer_names = []
                    data_types = []
                
                if not layer_names:
                    continue
                
                t = t_func(T_total)
                
                for layer_name in layer_names:
                    for data_type in data_types:
                        layer_key = f"{layer_name}_{data_type}"
                        # Collect exponents across all seeds
                        exponents_by_seed = []
                        for N in Ns:
                            if N not in all_data[config]:
                                continue
                            for seed, seed_data in all_data[config][N].items():
                                upd_N = seed_data.get(update_type)
                                if upd_N is None:
                                    continue
                                if isinstance(upd_N, np.ndarray):
                                    upd_N = upd_N.tolist()
                                if t < len(upd_N):
                                    update_dict = upd_N[t]
                                    if is_nested and data_type in update_dict:
                                        update_dict = update_dict[data_type]
                                    if isinstance(update_dict, dict) and layer_name in update_dict:
                                        # Store (N, value, seed) for later regression
                                        if seed not in [e[2] for e in exponents_by_seed if e[0] == N]:
                                            exponents_by_seed.append((N, update_dict[layer_name], seed))
                        
                        # Group by seed and compute exponent for each seed
                        if len(exponents_by_seed) >= 2:
                            seeds = list(set([e[2] for e in exponents_by_seed]))
                            seed_exponents = []
                            for seed in seeds:
                                seed_points = [(e[0], e[1]) for e in exponents_by_seed if e[2] == seed]
                                if len(seed_points) >= 2:
                                    widths_arr = np.array([p[0] for p in seed_points])
                                    values_arr = np.array([p[1] for p in seed_points])
                                    valid_mask = np.isfinite(values_arr) & np.isfinite(widths_arr) & (values_arr > 1e-20) & (widths_arr > 0)
                                    
                                    if np.sum(valid_mask) >= 2:
                                        widths_valid = widths_arr[valid_mask]
                                        values_valid = values_arr[valid_mask]
                                        log_widths = np.log10(widths_valid)
                                        log_values = np.log10(values_valid)
                                        
                                        try:
                                            slope, _, _, _, _ = stats.linregress(log_widths, log_values)
                                            if np.isfinite(slope):
                                                seed_exponents.append(slope)
                                        except:
                                            pass
                            
                            if seed_exponents:
                                # Store mean and std
                                exponent_data[config][layer_key] = {
                                    'mean': np.mean(seed_exponents),
                                    'std': np.std(seed_exponents),
                                    'n': len(seed_exponents)
                                }
            
            if exponent_data:
                all_layers = set()
                for config_layers in exponent_data.values():
                    all_layers.update(config_layers.keys())
                
                ordered_cols = [col for col in col_order if col in all_layers]
                remaining_cols = sorted([col for col in all_layers if col not in col_order])
                ordered_cols.extend(remaining_cols)
                
                ordered_configs = [cfg for cfg in row_order if cfg in exponent_data]
                remaining_configs = sorted([cfg for cfg in exponent_data.keys() if cfg not in row_order])
                ordered_configs.extend(remaining_configs)
                
                table_data = {}
                for config in ordered_configs:
                    row_data = []
                    for layer in ordered_cols:
                        val_dict = exponent_data[config].get(layer, {})
                        if isinstance(val_dict, dict) and 'mean' in val_dict:
                            mean = val_dict['mean']
                            std = val_dict['std']
                            n = val_dict['n']
                            row_data.append(f"{mean:.2f}±{std:.2f}" if n > 1 else f"{mean:.2f}")
                        else:
                            row_data.append(np.nan)
                    table_data[config] = row_data
                
                df = pd.DataFrame(table_data, index=ordered_cols).T
                
                csv_path = os.path.join(results_dir, f'exponent_table_{update_type}_t{t_val}.csv')
                df.to_csv(csv_path)
                print(f"Saved exponent table to {csv_path}")
                
                print(f"\n{update_type.replace('_', ' ').title()} Exponents at t={t_val}:")
                print(df.to_string())
                print()


_LAYER_ORDER = ['W_in', 'Q', 'W_exp1', 'W_exp2', 'W_out']
_LAYER_COLORS = {
    'W_in':   'tab:blue',
    'Q':      'tab:orange',
    'W_exp1': 'tab:green',
    'W_exp2': 'tab:red',
    'W_out':  'tab:purple',
}
_LAYER_DISPLAY = {
    'W_in': 'Input', 'Q': 'Router',
    'W_exp1': 'Expert In', 'W_exp2': 'Expert Out', 'W_out': 'Output',
}

_EXPTRAJ_UPDATE_TYPES = [
    ('effective_updates',                 ['normalized', 'raw']),
    ('propagating_updates',               ['raw']),
    ('total_effective_updates_per_layer', ['normalized', 'raw']),
    ('grad_norms',                        ['raw']),
    ('weight_rms_norms',                  ['raw']),
]

_EXPTRAJ_YLABEL = {
    ('effective_updates', 'normalized'):                 r'Exponent($\|\delta W^l x^l\|_\mathrm{rms} / \|x^l\|_\mathrm{rms}$)',
    ('effective_updates', 'raw'):                        r'Exponent($\|\delta W^l x^l\|_\mathrm{rms}$)',
    ('propagating_updates', 'raw'):                      r'Prop. Exponent($\|W^l_0 \Delta x^l\|_\mathrm{rms}$)',
    ('total_effective_updates_per_layer', 'normalized'): r'Eff. Exponent($\|\Delta W^l x^l\|_\mathrm{rms} / \|x^l\|_\mathrm{rms}$)',
    ('total_effective_updates_per_layer', 'raw'):        r'Eff. Exponent($\|\Delta W^l x^l\|_\mathrm{rms}$)',
    ('grad_norms', 'raw'):                               r'Gradnorm Exponent($\|\nabla_{W^l} L\|_\mathrm{rms}$)',
    ('weight_rms_norms', 'raw'):                         r'Weight norm Exponent($\|W^l\|_\mathrm{rms}$)',
}


def plot_exponent_trajectories(all_data, results_dir, timestep_stride=5):
    """For each RCC metric type, plot the log-log exponent (vs N) of every layer
    as a solid colored line across training steps.

    Layout: one file per (update_type, data_type, group), one subplot per config.
    Each subplot has:
      - x-axis: training step
      - y-axis: exponent (slope of log(metric) vs log(N))
      - one solid line per layer, colored by _LAYER_COLORS

    Saved to rcc/exponent_trajectory_{update_type}_{data_type}_{group}.png
    """
    from scipy import stats as sp_stats

    rcc_dir = os.path.join(results_dir, 'rcc')
    os.makedirs(rcc_dir, exist_ok=True)

    for update_type, data_types in _EXPTRAJ_UPDATE_TYPES:
        # Determine T from data
        T = None
        sample_entry = None
        for config in all_data:
            for N in all_data[config]:
                for seed, sd in all_data[config][N].items():
                    vals = sd.get(update_type)
                    if vals and len(vals) > 1:
                        T = len(vals)
                        sample_entry = vals[1]
                        break
                if T is not None:
                    break
            if T is not None:
                break
        if T is None or sample_entry is None:
            continue

        is_nested = isinstance(sample_entry, dict) and 'raw' in sample_entry

        for data_type in data_types:
            # Resolve layer names from sample entry
            if is_nested:
                layer_names = [l for l in _LAYER_ORDER if l in (sample_entry.get(data_type) or {})]
            elif isinstance(sample_entry, dict):
                if data_type == 'normalized':
                    continue  # non-nested data has no normalized variant
                layer_names = [l for l in _LAYER_ORDER if l in sample_entry]
            else:
                continue

            if not layer_names:
                continue

            timesteps = list(range(0, T, timestep_stride))

            # Pre-compute exponent trajectories: {config: {layer: array(T/stride)}}
            traj = {}
            for config in all_data:
                Ns = sorted(all_data[config].keys())
                if len(Ns) < 2:
                    continue
                traj[config] = {}
                for layer in layer_names:
                    exp_over_time = []
                    std_over_time = []
                    valid_ts = []
                    for t in timesteps:
                        points_by_seed = {}
                        for N in Ns:
                            for seed, sd in all_data[config][N].items():
                                upd = sd.get(update_type)
                                if upd is None or t >= len(upd):
                                    continue
                                entry = upd[t]
                                if is_nested:
                                    entry = entry.get(data_type, {})
                                val = entry.get(layer) if isinstance(entry, dict) else None
                                if val is None or not np.isfinite(val) or val <= 1e-20:
                                    continue
                                if seed not in points_by_seed:
                                    points_by_seed[seed] = []
                                points_by_seed[seed].append((N, val))

                        seed_exps = []
                        for pts in points_by_seed.values():
                            if len(pts) < 2:
                                continue
                            ws = np.array([p[0] for p in pts])
                            vs = np.array([p[1] for p in pts])
                            valid = np.isfinite(np.log10(vs)) & (vs > 0) & (ws > 0)
                            if valid.sum() < 2:
                                continue
                            try:
                                slope, *_ = sp_stats.linregress(np.log10(ws[valid]), np.log10(vs[valid]))
                                if np.isfinite(slope):
                                    seed_exps.append(slope)
                            except Exception:
                                pass

                        if seed_exps:
                            exp_over_time.append(np.mean(seed_exps))
                            std_over_time.append(np.std(seed_exps) if len(seed_exps) > 1 else 0.0)
                            valid_ts.append(t)

                    traj[config][layer] = (valid_ts, exp_over_time, std_over_time)

            if not traj:
                continue

            # One figure per group
            for config_group, group_name in organize_configs_by_group(all_data):
                group_configs = [c for c in config_group if c in traj]
                if not group_configs:
                    continue

                n_configs = len(group_configs)
                fig, axes = plt.subplots(
                    1, n_configs,
                    figsize=(n_configs * onefigsize[0], onefigsize[1]),
                    sharey=True,
                )
                if n_configs == 1:
                    axes = [axes]

                for ax_idx, config in enumerate(group_configs):
                    ax = axes[ax_idx]
                    for layer in layer_names:
                        ts_vals, exp_vals, std_vals = traj[config].get(layer, ([], [], []))
                        if not ts_vals:
                            continue
                        color = _LAYER_COLORS.get(layer, 'black')
                        ts_arr = np.array(ts_vals)
                        exp_arr = np.array(exp_vals)
                        std_arr = np.array(std_vals)
                        ax.plot(ts_arr, exp_arr, color=color, linewidth=1.5,
                                label=_LAYER_DISPLAY.get(layer, layer))
                        ax.fill_between(ts_arr, exp_arr - 2 * std_arr, exp_arr + 2 * std_arr,
                                        color=color, alpha=0.35)

                    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
                    ax.set_xlabel('Step')
                    if ax_idx == 0:
                        ax.set_ylabel(_EXPTRAJ_YLABEL.get((update_type, data_type), 'Exponent'))
                    ax.set_title(format_subplot_title(config))
                    _ylim = (-3, 0.5) if update_type == 'grad_norms' else (-1, 1)
                    ax.set_ylim(*_ylim)
                    from matplotlib.ticker import MultipleLocator
                    ax.yaxis.set_major_locator(MultipleLocator(0.5))
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=6, loc='lower center')

                plt.tight_layout()
                save_path = os.path.join(
                    rcc_dir,
                    f'exponent_trajectory_{update_type}_{data_type}_{group_name}.png'
                )
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"Saved exponent trajectory to {save_path}")


_DECOMP_TRAJ_TYPES = [
    ('hagg_decomp', {
        'base':        r'Init ($W^{3,i}_0 h^{2,i}_0$)',
        'propagating': r'Prop. ($W^{3,i}_0 \Delta h^{2,i}$)',
        'effective':   r'Eff.-init. ($\Delta W^{3,i} h^{2,i}_0$)',
        'cross':       r'Eff.-prop. ($\Delta W^{3,i} \Delta h^{2,i}$)',
    }),
    ('grad_h1_decomp', {
        'base':        r'Init ($(W^{2,i}_0)^\top g^i_0$)',
        'propagating': r'Prop. ($(W^{2,i}_0)^\top \Delta g^i$)',
        'effective':   r'Eff.-init. ($(\Delta W^{2,i})^\top g^i_0$)',
        'cross':       r'Eff.-prop. ($(\Delta W^{2,i})^\top \Delta g^i$)',
    }),
]
_DECOMP_TRAJ_YLABEL = {
    'hagg_decomp':     r'Exponent($h^\mathrm{agg}$ decomp.)',
    'grad_h1_decomp':  r'Exponent($\nabla_{h^1} L$ decomp.)',
}
_DECOMP_TERM_COLORS = {
    'base':        "#636363",   # gray
    'propagating': '#b8860b',   # dark yellow
    'effective':   "#90ee90ac",   # light green
    'cross':       "#1d4f20",   # dark green
}


def _compute_layer_traj(all_data, config, update_type, data_type, timestep_stride=5):
    """Compute per-layer exponent trajectory for one config.

    Returns {layer: (ts_list, exp_list, std_list)}.
    """
    from scipy import stats as sp_stats

    if config not in all_data:
        return {}
    Ns = sorted(all_data[config].keys())
    if len(Ns) < 2:
        return {}

    T = None
    sample_entry = None
    for N in Ns:
        for seed, sd in all_data[config][N].items():
            vals = sd.get(update_type)
            if vals and len(vals) > 1:
                T = len(vals)
                sample_entry = vals[1]
                break
        if T is not None:
            break
    if T is None or sample_entry is None:
        return {}

    is_nested = isinstance(sample_entry, dict) and 'raw' in sample_entry
    if is_nested:
        layer_names = [l for l in _LAYER_ORDER if l in (sample_entry.get(data_type) or {})]
    elif isinstance(sample_entry, dict):
        if data_type == 'normalized':
            return {}
        layer_names = [l for l in _LAYER_ORDER if l in sample_entry]
    else:
        return {}
    if not layer_names:
        return {}

    timesteps = list(range(0, T, timestep_stride))
    result = {}
    for layer in layer_names:
        exp_over_time, std_over_time, valid_ts = [], [], []
        for t in timesteps:
            pts_by_seed = {}
            for N in Ns:
                for seed, sd in all_data[config][N].items():
                    upd = sd.get(update_type)
                    if upd is None or t >= len(upd):
                        continue
                    entry = upd[t]
                    if is_nested:
                        entry = entry.get(data_type, {})
                    val = entry.get(layer) if isinstance(entry, dict) else None
                    if val is None or not np.isfinite(val) or val <= 1e-20:
                        continue
                    pts_by_seed.setdefault(seed, []).append((N, val))
            seed_exps = []
            for pts in pts_by_seed.values():
                if len(pts) < 2:
                    continue
                ws = np.array([p[0] for p in pts])
                vs = np.array([p[1] for p in pts])
                valid = np.isfinite(np.log10(vs)) & (vs > 0) & (ws > 0)
                if valid.sum() < 2:
                    continue
                try:
                    slope, *_ = sp_stats.linregress(np.log10(ws[valid]), np.log10(vs[valid]))
                    if np.isfinite(slope):
                        seed_exps.append(slope)
                except Exception:
                    pass
            if seed_exps:
                exp_over_time.append(np.mean(seed_exps))
                std_over_time.append(np.std(seed_exps) if len(seed_exps) > 1 else 0.0)
                valid_ts.append(t)
        result[layer] = (valid_ts, exp_over_time, std_over_time)
    return result


def _compute_decomp_traj(all_data, config, decomp_type, timestep_stride=5):
    """Compute decomposition term exponent trajectory for one config.

    Returns {term: (ts_list, exp_list, std_list)}.
    """
    from scipy import stats as sp_stats

    if config not in all_data:
        return {}
    Ns = sorted(all_data[config].keys())
    if len(Ns) < 2:
        return {}

    term_labels = None
    for dt, tl in _DECOMP_TRAJ_TYPES:
        if dt == decomp_type:
            term_labels = tl
            break
    if term_labels is None:
        return {}

    terms = list(term_labels.keys())
    data_keys = [f'{decomp_type}_{t}' for t in terms]

    T = None
    for N in Ns:
        for sd in all_data[config][N].values():
            v = sd.get(data_keys[0])
            if v and len(v) > 0:
                T = len(v)
                break
        if T is not None:
            break
    if T is None:
        return {}

    timesteps_nbase = [t for t in range(0, T, timestep_stride) if t > 0]
    result = {}
    for term, data_key in zip(terms, data_keys):
        ts_to_use = timesteps_nbase[:1] if term == 'base' else timesteps_nbase
        exp_vals, std_vals, valid_ts = [], [], []
        for t in ts_to_use:
            pts_by_seed = {}
            for N in Ns:
                for seed, sd in all_data[config][N].items():
                    vals = sd.get(data_key)
                    if vals is None or t >= len(vals):
                        continue
                    v = vals[t]
                    if not np.isfinite(v) or v <= 1e-20:
                        continue
                    pts_by_seed.setdefault(seed, []).append((N, v))
            seed_exps = []
            for pts in pts_by_seed.values():
                if len(pts) < 2:
                    continue
                ws = np.array([p[0] for p in pts])
                vs = np.array([p[1] for p in pts])
                mask = (ws > 0) & (vs > 0)
                if mask.sum() < 2:
                    continue
                try:
                    slope, *_ = sp_stats.linregress(np.log10(ws[mask]), np.log10(vs[mask]))
                    if np.isfinite(slope):
                        seed_exps.append(slope)
                except Exception:
                    pass
            if seed_exps:
                exp_vals.append(np.mean(seed_exps))
                std_vals.append(np.std(seed_exps) if len(seed_exps) > 1 else 0.0)
                valid_ts.append(0 if term == 'base' else t)
        result[term] = (valid_ts, exp_vals, std_vals)
    return result


def plot_decomp_exponent_trajectories(all_data, results_dir, timestep_stride=5):
    """Plot scaling exponents (log-log slope vs N) of decomposition terms over training steps.

    For hagg_decomp and grad_h1_decomp, one file per (decomp_type, group):
      rcc/exponent_trajectory_{decomp_type}_{group}.png

    Layout: one subplot per config.  Each subplot has:
      - x-axis: training step
      - y-axis: exponent (slope of log(term) vs log(N))
      - one solid colored line per term (propagating / effective / cross)
      - the base term shown as a single 'x' marker at t=0 only, since it
        captures the initialization scale and is conceptually static.
    """
    from scipy import stats as sp_stats

    rcc_dir = os.path.join(results_dir, 'rcc')
    os.makedirs(rcc_dir, exist_ok=True)

    for decomp_type, term_labels in _DECOMP_TRAJ_TYPES:
        terms = list(term_labels.keys())          # base, propagating, effective, cross
        data_keys = [f'{decomp_type}_{t}' for t in terms]

        # Check data availability
        has_data = any(
            any(sd.get(data_keys[0]) is not None for sd in all_data[c][N].values())
            for c in all_data for N in all_data[c]
        )
        if not has_data:
            continue

        # Determine T
        T = None
        for c in all_data:
            for N in all_data[c]:
                for sd in all_data[c][N].values():
                    v = sd.get(data_keys[0])
                    if v and len(v) > 0:
                        T = len(v)
                        break
                if T is not None:
                    break
            if T is not None:
                break
        if T is None:
            continue

        timesteps_all  = list(range(0, T, timestep_stride))
        timesteps_nbase = [t for t in timesteps_all if t > 0]   # lines skip t=0

        # Pre-compute exponent trajectory per config per term
        traj = {}   # {config: {term: (ts_list, exp_list)}}
        for config in all_data:
            Ns = sorted(all_data[config].keys())
            if len(Ns) < 2:
                continue
            traj[config] = {}
            for term, data_key in zip(terms, data_keys):
                # Base term is constant (W0 h0); t=0 is logged as {} so use the
                # first valid non-zero timestep and force x-position to 0.
                ts_to_use = timesteps_nbase[:1] if term == 'base' else timesteps_nbase
                exp_vals, std_vals, valid_ts = [], [], []
                for t in ts_to_use:
                    pts_by_seed = {}
                    for N in Ns:
                        for seed, sd in all_data[config][N].items():
                            vals = sd.get(data_key)
                            if vals is None or t >= len(vals):
                                continue
                            v = vals[t]
                            if not np.isfinite(v) or v <= 1e-20:
                                continue
                            pts_by_seed.setdefault(seed, []).append((N, v))

                    seed_exps = []
                    for pts in pts_by_seed.values():
                        if len(pts) < 2:
                            continue
                        ws = np.array([p[0] for p in pts])
                        vs = np.array([p[1] for p in pts])
                        mask = (ws > 0) & (vs > 0)
                        if mask.sum() < 2:
                            continue
                        try:
                            slope, *_ = sp_stats.linregress(
                                np.log10(ws[mask]), np.log10(vs[mask])
                            )
                            if np.isfinite(slope):
                                seed_exps.append(slope)
                        except Exception:
                            pass

                    if seed_exps:
                        exp_vals.append(np.mean(seed_exps))
                        std_vals.append(np.std(seed_exps) if len(seed_exps) > 1 else 0.0)
                        valid_ts.append(0 if term == 'base' else t)

                traj[config][term] = (valid_ts, exp_vals, std_vals)

        if not traj:
            continue

        # One figure per group
        for config_group, group_name in organize_configs_by_group(all_data):
            group_configs = [c for c in config_group if c in traj]
            if not group_configs:
                continue

            n_configs = len(group_configs)
            fig, axes = plt.subplots(
                1, n_configs,
                figsize=(n_configs * onefigsize[0], onefigsize[1]),
                sharey=True,
            )
            if n_configs == 1:
                axes = [axes]

            for ax_idx, config in enumerate(group_configs):
                ax = axes[ax_idx]
                for term in terms:
                    ts_vals, exp_vals, std_vals = traj[config].get(term, ([], [], []))
                    if not ts_vals:
                        continue
                    color = _DECOMP_TERM_COLORS.get(term, 'black')
                    label = term_labels[term]
                    ts_arr = np.array(ts_vals)
                    exp_arr = np.array(exp_vals)
                    std_arr = np.array(std_vals)
                    if term == 'base':
                        ax.scatter(ts_arr, exp_arr, marker='x', s=60,
                                   color=color, linewidths=1.5, zorder=5, label=label)
                    else:
                        ax.plot(ts_arr, exp_arr, color=color, linewidth=1.5, label=label)
                        ax.fill_between(ts_arr, exp_arr - 2 * std_arr, exp_arr + 2 * std_arr,
                                        color=color, alpha=0.35)

                ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
                ax.set_xlabel('Step')
                if ax_idx == 0:
                    ax.set_ylabel(_DECOMP_TRAJ_YLABEL.get(decomp_type, 'Exponent'))
                ax.set_title(format_subplot_title(config))
                ax.set_ylim(-2, 1)
                from matplotlib.ticker import MultipleLocator
                ax.yaxis.set_major_locator(MultipleLocator(0.5))
                ax.grid(True, alpha=0.3)
                _leg_loc = 'upper center' if decomp_type == 'grad_h1_decomp' else 'lower center'
                ax.legend(fontsize=6, loc=_leg_loc, frameon=True,
                          title='Decomp. term', title_fontsize=8)

            plt.tight_layout()
            save_path = os.path.join(
                rcc_dir,
                f'exponent_trajectory_{decomp_type}_{group_name}.png'
            )
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved decomp exponent trajectory to {save_path}")


def compute_decomp_exponent_tables(all_data, results_dir, timesteps=None):
    """Compute exponent tables for hagg_decomp and grad_h1_decomp terms.

    For each decomposition type, saves one CSV per timestep:
      exponent_table_hagg_decomp_t{ts}.csv
      exponent_table_grad_h1_decomp_t{ts}.csv

    Rows = configs, columns = terms (base, propagating, effective, cross).
    Values are the log-log slopes of each term vs N across widths.
    """
    from scipy import stats
    import pandas as pd

    if not all_data:
        return

    first_config = list(all_data.keys())[0]
    first_N = min(all_data[first_config].keys())
    first_seed = list(all_data[first_config][first_N].keys())[0]
    T_total = all_data[first_config][first_N][first_seed].get('config', {}).get(
        'T', len(all_data[first_config][first_N][first_seed].get('losses', [])) or 1000
    )

    if timesteps is None:
        timesteps_to_use = [2, 50, 200]
    else:
        timesteps_to_use = list(timesteps)
    if T_total - 1 not in timesteps_to_use:
        timesteps_to_use.append(T_total - 1)

    DECOMP_TYPES = [
        ('hagg_decomp',
         ['hagg_decomp_base', 'hagg_decomp_propagating', 'hagg_decomp_effective', 'hagg_decomp_cross'],
         ['base', 'propagating', 'effective', 'cross']),
        ('grad_h1_decomp',
         ['grad_h1_decomp_base', 'grad_h1_decomp_propagating', 'grad_h1_decomp_effective', 'grad_h1_decomp_cross'],
         ['base', 'propagating', 'effective', 'cross']),
    ]

    for decomp_type, data_keys, term_names in DECOMP_TYPES:
        # Check if any data exists for this decomp type
        has_data = any(
            any(sd.get(data_keys[0]) is not None for sd in all_data[c][N].values())
            for c in all_data for N in all_data[c]
        )
        if not has_data:
            continue

        for t_val in timesteps_to_use:
            if t_val >= T_total:
                continue

            exponent_data = {}
            for config in all_data.keys():
                Ns = sorted(all_data[config].keys())
                if len(Ns) < 2:
                    continue

                row = {}
                for data_key, term_name in zip(data_keys, term_names):
                    points_by_seed = {}
                    for N in Ns:
                        for seed, seed_data in all_data[config][N].items():
                            vals = seed_data.get(data_key)
                            if vals is None or t_val >= len(vals):
                                continue
                            val = vals[t_val]
                            if not np.isfinite(val) or val <= 1e-20:
                                continue
                            if seed not in points_by_seed:
                                points_by_seed[seed] = []
                            points_by_seed[seed].append((N, val))

                    seed_exponents = []
                    for seed, pts in points_by_seed.items():
                        if len(pts) < 2:
                            continue
                        widths = np.array([p[0] for p in pts])
                        values = np.array([p[1] for p in pts])
                        log_w = np.log10(widths)
                        log_v = np.log10(values)
                        try:
                            slope, _, _, _, _ = stats.linregress(log_w, log_v)
                            if np.isfinite(slope):
                                seed_exponents.append(slope)
                        except Exception:
                            pass

                    if seed_exponents:
                        mean = np.mean(seed_exponents)
                        std = np.std(seed_exponents)
                        n = len(seed_exponents)
                        row[term_name] = f"{mean:.2f}±{std:.2f}" if n > 1 else f"{mean:.2f}"
                    else:
                        row[term_name] = np.nan

                if row:
                    exponent_data[config] = row

            if exponent_data:
                df = pd.DataFrame(exponent_data, index=term_names).T
                csv_path = os.path.join(results_dir, f'exponent_table_{decomp_type}_t{t_val}.csv')
                df.to_csv(csv_path)
                print(f"Saved decomp exponent table to {csv_path}")


def plot_h_L_coordinate_check(all_data, results_dir):
    """Plot h^L RMS difference vs width at different timesteps."""
    from scipy import stats

    # Get filtered config groups
    config_groups = filter_configs_by_regime_and_optimizer(all_data)
    fixed_expert_sgd = config_groups['fixed_expert_sgd']
    fixed_expert_adam = config_groups['fixed_expert_adam']
    allscaling_sgd = config_groups['allscaling_sgd']
    allscaling_adam = config_groups['allscaling_adam']
    bottleneck_sgd = config_groups['bottleneck_sgd']
    bottleneck_adam = config_groups['bottleneck_adam']
    expert_scaling = config_groups['expert_scaling']

    for config_group, group_name in [
        (fixed_expert_sgd, 'fixed_E_sgd'),
        (fixed_expert_adam, 'fixed_E_adam'),
        (expert_scaling, 'other'),
        (allscaling_sgd, 'allscaling_sgd'),
        (allscaling_adam, 'allscaling_adam'),
        (bottleneck_sgd, 'bottleneck_sgd'),
        (bottleneck_adam, 'bottleneck_adam')
    ]:
        if not config_group:
            continue

        n_configs = len(config_group)
        fig, axes = plt.subplots(1, n_configs, figsize=(n_configs*onefigsize[0], onefigsize[1]), sharey=True)
        if n_configs == 1:
            axes = [axes]

        for ax_idx, config in enumerate(config_group):
            ax = axes[ax_idx]
            Ns = sorted(all_data[config].keys())
            if len(Ns) < 2:
                continue

            # Get h_L_rms from first seed
            first_seed = list(all_data[config][Ns[0]].keys())[0]
            h_L_data = all_data[config][Ns[0]][first_seed].get('h_L_rms')
            if h_L_data is None:
                continue
            
            T_total = all_data[config][Ns[0]][first_seed].get('config', {}).get('T', len(h_L_data))
            timesteps = [2, 50, T_total//5, T_total-1] if T_total > 10 else list(range(1, T_total))
            blues = plt.cm.Blues(np.linspace(0.4, 0.95, len(timesteps)))
            
            for t_idx, t in enumerate(timesteps):
                # Collect all seed data
                seed_data_dict = {}  # {seed: {N: value}}
                for N in Ns:
                    for seed, seed_data in all_data[config][N].items():
                        h_L_rms = seed_data.get('h_L_rms')
                        if h_L_rms is not None and t < len(h_L_rms):
                            if seed not in seed_data_dict:
                                seed_data_dict[seed] = {}
                            seed_data_dict[seed][N] = h_L_rms[t]
                
                # Plot individual seeds with alpha=0.5
                for seed, seed_points in seed_data_dict.items():
                    if len(seed_points) >= 2:
                        widths = np.array(list(seed_points.keys()))
                        values = np.array(list(seed_points.values()))
                        # Plot points
                        ax.plot(widths, values, 'o', color=blues[t_idx], markersize=4, alpha=0.5)
                        # Fit and plot line
                        log_widths, log_values = np.log10(widths), np.log10(values)
                        try:
                            slope, intercept, _, _, _ = stats.linregress(log_widths, log_values)
                            x_fit = np.logspace(np.log10(min(widths)), np.log10(max(widths)), 100)
                            ax.plot(x_fit, 10**intercept * x_fit**slope, '-', color=blues[t_idx], 
                                   alpha=0.5, linewidth=1.5)
                        except:
                            pass
                
                # Compute and plot mean fit
                if seed_data_dict:
                    all_widths = sorted(set(N for seed_points in seed_data_dict.values() for N in seed_points.keys()))
                    mean_values = []
                    for N in all_widths:
                        vals = [seed_points[N] for seed_points in seed_data_dict.values() if N in seed_points]
                        if vals:
                            mean_values.append(np.mean(vals))
                    
                    if len(all_widths) >= 2:
                        widths_arr = np.array(all_widths)
                        h_L_arr = np.array(mean_values)
                        log_widths = np.log10(widths_arr)
                        log_h_L = np.log10(h_L_arr)
                        
                        try:
                            slope, intercept, _, _, _ = stats.linregress(log_widths, log_h_L)
                            alpha_fit = np.round(slope, 2)
                            C_fit = 10**intercept
                            x_fit = np.logspace(np.log10(min(all_widths)), np.log10(max(all_widths)), 100)
                            ax.plot(x_fit, C_fit * x_fit**alpha_fit, '-', color=blues[t_idx], 
                                   label=f't={t} ({alpha_fit:.2f})', linewidth=2)
                        except:
                            pass
            
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xlabel('Width N')
            ax.set_ylabel(r'$\mathrm{RMS}(h^L)$')
            ax.set_title(format_subplot_title(config))
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # Set shared ylims for all subplots
        set_shared_ylim(axes)

        # Determine config name for single-config plots
        single_config_name = config_group[0] if n_configs == 1 else None

        plt.tight_layout()
        save_path = get_plot_save_path(results_dir, single_config_name,
                                       f'h_L_coordinate_check_{group_name}.png',
                                       is_joint_plot=(n_configs > 1))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def plot_activation_rms(all_data, results_dir):
    """Plot RMS of all activations vs width at different timesteps."""
    from scipy import stats

    # Get filtered config groups
    config_groups = filter_configs_by_regime_and_optimizer(all_data)
    fixed_expert_sgd = config_groups['fixed_expert_sgd']
    fixed_expert_adam = config_groups['fixed_expert_adam']
    allscaling_sgd = config_groups['allscaling_sgd']
    allscaling_adam = config_groups['allscaling_adam']
    bottleneck_sgd = config_groups['bottleneck_sgd']
    bottleneck_adam = config_groups['bottleneck_adam']

    expert_scaling = [c for c in CONFIG_ORDER if c in all_data and 'fixed_E' not in c and 'allscaling' not in c and 'bottleneck' not in c]

    for config_group, group_name in [
        (fixed_expert_sgd, 'fixed_E_sgd'),
        (fixed_expert_adam, 'fixed_E_adam'),
        (expert_scaling, 'other'),
        (allscaling_sgd, 'allscaling_sgd'),
        (allscaling_adam, 'allscaling_adam'),
        (bottleneck_sgd, 'bottleneck_sgd'),
        (bottleneck_adam, 'bottleneck_adam')
    ]:
        if not config_group:
            continue
        
        # Get activation names from first config
        first_config = config_group[0]
        first_N = min(all_data[first_config].keys())
        first_seed = list(all_data[first_config][first_N].keys())[0]
        act_rms_data = all_data[first_config][first_N][first_seed].get('activation_rms')
        if not act_rms_data or len(act_rms_data) == 0:
            continue
        
        # Get activation layer names
        if isinstance(act_rms_data[0], dict):
            activation_names = list(act_rms_data[0].keys())
        else:
            continue
        
        # Plot each activation layer
        for act_name in activation_names:
            n_configs = len(config_group)
            fig, axes = plt.subplots(1, n_configs, figsize=(n_configs*onefigsize[0], onefigsize[1]), sharey=True)
            if n_configs == 1:
                axes = [axes]
            
            for ax_idx, config in enumerate(config_group):
                ax = axes[ax_idx]
                Ns = sorted(all_data[config].keys())
                if len(Ns) < 2:
                    continue
                
                first_seed = list(all_data[config][Ns[0]].keys())[0]
                T_total = all_data[config][Ns[0]][first_seed].get('config', {}).get('T', 100)
                timesteps = [2, 50, T_total//5, T_total-1] if T_total > 10 else list(range(1, T_total))
                blues = plt.cm.Blues(np.linspace(0.4, 0.95, len(timesteps)))
                
                for t_idx, t in enumerate(timesteps):
                    seed_data_dict = {}
                    for N in Ns:
                        for seed, seed_data in all_data[config][N].items():
                            act_rms = seed_data.get('activation_rms')
                            if act_rms and t < len(act_rms) and isinstance(act_rms[t], dict) and act_name in act_rms[t]:
                                if seed not in seed_data_dict:
                                    seed_data_dict[seed] = {}
                                seed_data_dict[seed][N] = act_rms[t][act_name]
                    
                    # Plot individual seeds
                    for seed, seed_points in seed_data_dict.items():
                        if len(seed_points) >= 2:
                            widths = np.array(list(seed_points.keys()))
                            values = np.array(list(seed_points.values()))
                            ax.plot(widths, values, 'o', color=blues[t_idx], markersize=4, alpha=0.5)
                            log_widths, log_values = np.log10(widths), np.log10(values)
                            try:
                                slope, intercept, _, _, _ = stats.linregress(log_widths, log_values)
                                x_fit = np.logspace(np.log10(min(widths)), np.log10(max(widths)), 100)
                                ax.plot(x_fit, 10**intercept * x_fit**slope, '-', color=blues[t_idx], alpha=0.5, linewidth=1.5)
                            except:
                                pass
                    
                    # Plot mean fit
                    if seed_data_dict:
                        all_widths = sorted(set(N for seed_points in seed_data_dict.values() for N in seed_points.keys()))
                        mean_values = [np.mean([seed_points[N] for seed_points in seed_data_dict.values() if N in seed_points]) 
                                     for N in all_widths if any(N in seed_points for seed_points in seed_data_dict.values())]
                        
                        if len(all_widths) >= 2 and len(mean_values) >= 2:
                            widths_arr = np.array(all_widths[:len(mean_values)])
                            values_arr = np.array(mean_values)
                            log_widths, log_values = np.log10(widths_arr), np.log10(values_arr)
                            try:
                                slope, intercept, _, _, _ = stats.linregress(log_widths, log_values)
                                alpha_fit = np.round(slope, 2)
                                x_fit = np.logspace(np.log10(min(all_widths)), np.log10(max(all_widths)), 100)
                                ax.plot(x_fit, 10**intercept * x_fit**alpha_fit, '-', color=blues[t_idx], 
                                       label=f't={t} ({alpha_fit:.2f})', linewidth=2)
                            except:
                                pass
                
                ax.set_xscale('log'); ax.set_yscale('log')
                ax.set_xlabel('Width N')
                ax.set_ylabel(f'RMS({act_name})')
                ax.set_title(format_subplot_title(config))
                ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

            # Set shared ylims for all subplots
            set_shared_ylim(axes)

            # Determine config name for single-config plots
            single_config_name = config_group[0] if n_configs == 1 else None

            plt.tight_layout()
            save_path = get_plot_save_path(results_dir, single_config_name,
                                           f'activation_rms_{act_name}_{group_name}.png',
                                           is_joint_plot=(n_configs > 1))
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

def plot_hagg_decomp_rcc(all_data, results_dir, T_total_override=None):
    """RCC-style log-log plots of h^agg decomposition terms vs width N.

    One figure per term (base / propagating / effective / cross), each following
    the standard Blues-timestep convention used by the other RCC plots.
    Saved as rcc/hagg_decomp_{term}_{group}.png.
    """
    from scipy import stats

    TERMS = [
        ('hagg_decomp_base',        'base',        r'$\mathrm{RMS}(\sum_i \phi_i W^{3,i}_0 h^{2,i}_0)$'),
        ('hagg_decomp_propagating', 'propagating', r'$\mathrm{RMS}(\sum_i \phi_i W^{3,i}_0 \Delta h^{2,i})$'),
        ('hagg_decomp_effective',   'effective',   r'$\mathrm{RMS}(\sum_i \phi_i \Delta W^{3,i} h^{2,i}_0)$'),
        ('hagg_decomp_cross',       'cross',       r'$\mathrm{RMS}(\sum_i \phi_i \Delta W^{3,i} \Delta h^{2,i})$'),
    ]

    # Use helper function to organize configs by group
    for config_group, group_name in organize_configs_by_group(all_data):
        if not config_group:
            continue

        has_data = any(
            any(sd.get('hagg_decomp_base') for sd in all_data[c][N].values())
            for c in config_group for N in all_data[c]
        )
        if not has_data:
            continue

        n_configs = len(config_group)

        for key, term_name, ylabel in TERMS:
            fig, axes = plt.subplots(1, n_configs, figsize=(n_configs * onefigsize[0], onefigsize[1]), sharey=True)
            if n_configs == 1:
                axes = [axes]

            for ax_idx, config in enumerate(config_group):
                ax = axes[ax_idx]
                Ns = sorted(all_data[config].keys())
                if len(Ns) < 2:
                    continue

                first_seed = list(all_data[config][Ns[0]].keys())[0]
                ref_data = all_data[config][Ns[0]][first_seed].get(key)
                if not ref_data:
                    continue

                T_total = T_total_override if T_total_override is not None else \
                    all_data[config][Ns[0]][first_seed].get('config', {}).get('T', len(ref_data))
                timesteps = [2, 50, T_total//5, T_total-1] if T_total > 10 else list(range(2, T_total))
                blues = plt.cm.Blues(np.linspace(0.4, 0.95, len(timesteps)))

                for t_idx, t in enumerate(timesteps):
                    seed_data_dict = {}
                    for N in Ns:
                        for seed, seed_data in all_data[config][N].items():
                            arr = seed_data.get(key)
                            if arr is not None and t < len(arr):
                                seed_data_dict.setdefault(seed, {})[N] = arr[t]

                    # Individual seeds (faint)
                    for seed, seed_points in seed_data_dict.items():
                        if len(seed_points) >= 2:
                            widths = np.array(list(seed_points.keys()))
                            values = np.array(list(seed_points.values()), dtype=float)
                            mask = np.isfinite(values) & (values > 0)
                            if mask.sum() >= 2:
                                ax.plot(widths[mask], values[mask], 'o', color=blues[t_idx], markersize=4, alpha=0.5)
                                try:
                                    slope, intercept, *_ = stats.linregress(np.log10(widths[mask]), np.log10(values[mask]))
                                    x_fit = np.logspace(np.log10(widths[mask].min()), np.log10(widths[mask].max()), 100)
                                    ax.plot(x_fit, 10 ** intercept * x_fit ** slope, '-', color=blues[t_idx], alpha=0.5, linewidth=1.5)
                                except Exception:
                                    pass

                    # Mean fit across seeds
                    if seed_data_dict:
                        all_widths = sorted(set(N for sp in seed_data_dict.values() for N in sp))
                        mean_values = [np.mean([sp[N] for sp in seed_data_dict.values() if N in sp]) for N in all_widths]
                        widths_arr = np.array(all_widths)
                        values_arr = np.array(mean_values, dtype=float)
                        mask = np.isfinite(values_arr) & (values_arr > 0)
                        if mask.sum() >= 2:
                            ax.plot(widths_arr[mask], values_arr[mask], 'o', color=blues[t_idx], markersize=4)
                            try:
                                slope, intercept, *_ = stats.linregress(np.log10(widths_arr[mask]), np.log10(values_arr[mask]))
                                alpha_fit = round(slope, 2)
                                x_fit = np.logspace(np.log10(widths_arr[mask].min()), np.log10(widths_arr[mask].max()), 100)
                                ax.plot(x_fit, 10 ** intercept * x_fit ** alpha_fit, '-',
                                        color=blues[t_idx], label=f't={t} ({alpha_fit:.2f})', linewidth=2)
                            except Exception:
                                pass

                # t=0 'x' marker for the base term (initial scale, constant across time)
                if term_name == 'base':
                    seed_data_t0 = {}
                    for N in Ns:
                        for seed, seed_data in all_data[config][N].items():
                            arr = seed_data.get(key)
                            if arr is not None and len(arr) > 0:
                                seed_data_t0.setdefault(seed, {})[N] = arr[0]
                    for seed, seed_points in seed_data_t0.items():
                        if len(seed_points) >= 2:
                            widths = np.array(list(seed_points.keys()))
                            values = np.array(list(seed_points.values()), dtype=float)
                            mask = np.isfinite(values) & (values > 0)
                            if mask.sum() >= 2:
                                ax.plot(widths[mask], values[mask], 'x', color='black',
                                        markersize=5, alpha=0.5, linewidth=1.2)
                    if seed_data_t0:
                        all_w = sorted(set(N for sp in seed_data_t0.values() for N in sp))
                        mean_v = [np.mean([sp[N] for sp in seed_data_t0.values() if N in sp]) for N in all_w]
                        widths_arr = np.array(all_w)
                        values_arr = np.array(mean_v, dtype=float)
                        mask = np.isfinite(values_arr) & (values_arr > 0)
                        if mask.sum() >= 2:
                            ax.plot(widths_arr[mask], values_arr[mask], 'x', color='black',
                                    markersize=6, linewidth=1.5)
                            try:
                                slope, intercept, *_ = stats.linregress(
                                    np.log10(widths_arr[mask]), np.log10(values_arr[mask]))
                                alpha_fit = round(slope, 2)
                                x_fit = np.logspace(np.log10(widths_arr[mask].min()),
                                                    np.log10(widths_arr[mask].max()), 100)
                                ax.plot(x_fit, 10**intercept * x_fit**alpha_fit, '--',
                                        color='black', label=f't=0 ({alpha_fit:.2f})', linewidth=2)
                            except Exception:
                                pass

                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlabel('Width N')
                if ax_idx == 0:
                    ax.set_ylabel(ylabel)
                ax.set_title(format_label(config))
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            rcc_dir = os.path.join(results_dir, 'rcc')
            os.makedirs(rcc_dir, exist_ok=True)
            plt.savefig(os.path.join(rcc_dir, f'hagg_decomp_{term_name}_{group_name}.png'), dpi=300, bbox_inches='tight')
            plt.close()

def plot_grad_h1_decomp_rcc(all_data, results_dir, T_total_override=None):
    """RCC-style log-log plots of the expert-induced ∇_{h^1} L decomposition vs width N.

    The four terms come from writing W^{2,i}(t) = W^{2,i}(0) + ΔW^{2,i}(t) and
    g_{μ,i}(t) = g_{μ,i}(0) + Δg_{μ,i}(t), where g_{μ,i} = ∂L/∂h^{2,in}_{μ,i}:

      base:        sum_i g_{μ,i}(0)   @ W^{2,i}(0)^T
      propagating: sum_i Δg_{μ,i}(t)  @ W^{2,i}(0)^T
      effective:   sum_i g_{μ,i}(0)   @ ΔW^{2,i}(t)^T
      cross:       sum_i Δg_{μ,i}(t)  @ ΔW^{2,i}(t)^T

    Saved as rcc/grad_h1_decomp_{term}_{group}.png.
    """
    from scipy import stats

    TERMS = [
        ('grad_h1_decomp_base',
         'base',
         r'$\mathrm{RMS}\!\left(\sum_i (W^{2,i}_0)^\top g^i_0\right)$'),
        ('grad_h1_decomp_propagating',
         'propagating',
         r'$\mathrm{RMS}\!\left(\sum_i (W^{2,i}_0)^\top \Delta g^i\right)$'),
        ('grad_h1_decomp_effective',
         'effective',
         r'$\mathrm{RMS}\!\left(\sum_i (\Delta W^{2,i})^\top g^i_0\right)$'),
        ('grad_h1_decomp_cross',
         'cross',
         r'$\mathrm{RMS}\!\left(\sum_i (\Delta W^{2,i})^\top \Delta g^i\right)$'),
    ]

    # Use helper function to organize configs by group
    for config_group, group_name in organize_configs_by_group(all_data):
        if not config_group:
            continue

        has_data = any(
            any(sd.get('grad_h1_decomp_base') for sd in all_data[c][N].values())
            for c in config_group for N in all_data[c]
        )
        if not has_data:
            continue

        n_configs = len(config_group)

        for key, term_name, ylabel in TERMS:
            fig, axes = plt.subplots(1, n_configs, figsize=(n_configs * onefigsize[0], onefigsize[1]), sharey=True)
            if n_configs == 1:
                axes = [axes]

            for ax_idx, config in enumerate(config_group):
                ax = axes[ax_idx]
                Ns = sorted(all_data[config].keys())
                if len(Ns) < 2:
                    continue

                first_seed = list(all_data[config][Ns[0]].keys())[0]
                ref_data = all_data[config][Ns[0]][first_seed].get(key)
                if not ref_data:
                    continue

                T_total = T_total_override if T_total_override is not None else \
                    all_data[config][Ns[0]][first_seed].get('config', {}).get('T', len(ref_data))
                timesteps = [2, 50, T_total//5, T_total-1] if T_total > 10 else list(range(2, T_total))
                blues = plt.cm.Blues(np.linspace(0.4, 0.95, len(timesteps)))

                for t_idx, t in enumerate(timesteps):
                    seed_data_dict = {}
                    for N in Ns:
                        for seed, seed_data in all_data[config][N].items():
                            arr = seed_data.get(key)
                            if arr is not None and t < len(arr):
                                seed_data_dict.setdefault(seed, {})[N] = arr[t]

                    # Individual seeds (faint)
                    for seed, seed_points in seed_data_dict.items():
                        if len(seed_points) >= 2:
                            widths = np.array(list(seed_points.keys()))
                            values = np.array(list(seed_points.values()), dtype=float)
                            mask = np.isfinite(values) & (values > 0)
                            if mask.sum() >= 2:
                                ax.plot(widths[mask], values[mask], 'o', color=blues[t_idx], markersize=4, alpha=0.5)
                                try:
                                    slope, intercept, *_ = stats.linregress(np.log10(widths[mask]), np.log10(values[mask]))
                                    x_fit = np.logspace(np.log10(widths[mask].min()), np.log10(widths[mask].max()), 100)
                                    ax.plot(x_fit, 10 ** intercept * x_fit ** slope, '-', color=blues[t_idx], alpha=0.5, linewidth=1.5)
                                except Exception:
                                    pass

                    # Mean fit across seeds
                    if seed_data_dict:
                        all_widths = sorted(set(N for sp in seed_data_dict.values() for N in sp))
                        mean_values = [np.mean([sp[N] for sp in seed_data_dict.values() if N in sp]) for N in all_widths]
                        widths_arr = np.array(all_widths)
                        values_arr = np.array(mean_values, dtype=float)
                        mask = np.isfinite(values_arr) & (values_arr > 0)
                        if mask.sum() >= 2:
                            ax.plot(widths_arr[mask], values_arr[mask], 'o', color=blues[t_idx], markersize=4)
                            try:
                                slope, intercept, *_ = stats.linregress(np.log10(widths_arr[mask]), np.log10(values_arr[mask]))
                                alpha_fit = round(slope, 2)
                                x_fit = np.logspace(np.log10(widths_arr[mask].min()), np.log10(widths_arr[mask].max()), 100)
                                ax.plot(x_fit, 10 ** intercept * x_fit ** alpha_fit, '-',
                                        color=blues[t_idx], label=f't={t} ({alpha_fit:.2f})', linewidth=2)
                            except Exception:
                                pass

                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlabel('Width N')
                if ax_idx == 0:
                    ax.set_ylabel(ylabel)
                ax.set_title(format_label(config))
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

            plt.tight_layout()
            rcc_dir = os.path.join(results_dir, 'rcc')
            os.makedirs(rcc_dir, exist_ok=True)
            plt.savefig(os.path.join(rcc_dir, f'grad_h1_decomp_{term_name}_{group_name}.png'), dpi=300, bbox_inches='tight')
            plt.close()

def plot_loss_grad_rms(all_data, results_dir, T_total_override=None):
    """Plot RMS of dL/d(logits) — gradient of loss w.r.t. model output — vs width N."""
    from scipy import stats

    # Use helper function to organize configs by group
    for config_group, group_name in organize_configs_by_group(all_data):
        if not config_group:
            continue

        has_data = any(
            any(seed_data.get('loss_grad_rms') for seed_data in all_data[c][N].values())
            for c in config_group for N in all_data[c]
        )
        if not has_data:
            continue

        n_configs = len(config_group)
        fig, axes = plt.subplots(1, n_configs, figsize=(n_configs * onefigsize[0], onefigsize[1]), sharey=True)
        if n_configs == 1:
            axes = [axes]

        for ax_idx, config in enumerate(config_group):
            ax = axes[ax_idx]
            Ns = sorted(all_data[config].keys())
            if len(Ns) < 2:
                continue

            first_seed = list(all_data[config][Ns[0]].keys())[0]
            lg_data = all_data[config][Ns[0]][first_seed].get('loss_grad_rms')
            if not lg_data:
                continue

            T_total = T_total_override if T_total_override is not None else \
                all_data[config][Ns[0]][first_seed].get('config', {}).get('T', len(lg_data))
            timesteps = [2, 50, T_total//5, T_total-1] if T_total > 10 else list(range(2, T_total))
            blues = plt.cm.Blues(np.linspace(0.4, 0.95, len(timesteps)))

            for t_idx, t in enumerate(timesteps):
                seed_data_dict = {}
                for N in Ns:
                    for seed, seed_data in all_data[config][N].items():
                        lg = seed_data.get('loss_grad_rms')
                        if lg is not None and t < len(lg):
                            if seed not in seed_data_dict:
                                seed_data_dict[seed] = {}
                            seed_data_dict[seed][N] = lg[t]

                for seed, seed_points in seed_data_dict.items():
                    if len(seed_points) >= 2:
                        widths = np.array(list(seed_points.keys()))
                        values = np.array(list(seed_points.values()), dtype=float)
                        mask = np.isfinite(values) & (values > 0)
                        if mask.sum() >= 2:
                            ax.plot(widths[mask], values[mask], 'o', color=blues[t_idx], markersize=4, alpha=0.5)
                            try:
                                slope, intercept, *_ = stats.linregress(np.log10(widths[mask]), np.log10(values[mask]))
                                x_fit = np.logspace(np.log10(widths[mask].min()), np.log10(widths[mask].max()), 100)
                                ax.plot(x_fit, 10 ** intercept * x_fit ** slope, '-', color=blues[t_idx], alpha=0.5, linewidth=1.5)
                            except Exception:
                                pass

                if seed_data_dict:
                    all_widths = sorted(set(N for sp in seed_data_dict.values() for N in sp))
                    mean_values = [np.mean([sp[N] for sp in seed_data_dict.values() if N in sp]) for N in all_widths]
                    widths_arr = np.array(all_widths)
                    values_arr = np.array(mean_values, dtype=float)
                    mask = np.isfinite(values_arr) & (values_arr > 0)
                    if mask.sum() >= 2:
                        try:
                            slope, intercept, *_ = stats.linregress(np.log10(widths_arr[mask]), np.log10(values_arr[mask]))
                            alpha_fit = round(slope, 2)
                            x_fit = np.logspace(np.log10(widths_arr[mask].min()), np.log10(widths_arr[mask].max()), 100)
                            ax.plot(x_fit, 10 ** intercept * x_fit ** alpha_fit, '-',
                                    color=blues[t_idx], label=f't={t} ({alpha_fit:.2f})', linewidth=2)
                        except Exception:
                            pass

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Width N')
            ax.set_ylabel(r'$\|\nabla_{\rm logits}\mathcal{L}\|_{\rm RMS}$')
            ax.set_title(format_label(config))
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        rcc_dir = os.path.join(results_dir, 'rcc')
        os.makedirs(rcc_dir, exist_ok=True)
        plt.savefig(os.path.join(rcc_dir, f'loss_grad_rms_{group_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_expert_grad_to_h1(all_data, results_dir, T_total_override=None):
    """Plot RMS of the summed expert gradient contributions to h1 vs width N.

    Quantity: ||sum_m dL/dh1 [through expert m]||_RMS
    Router weights are treated as constants; only the expert pathway is included.
    """
    from scipy import stats

    # Use helper function to organize configs by group
    for config_group, group_name in organize_configs_by_group(all_data):
        if not config_group:
            continue

        # Check that at least one config has data
        has_data = any(
            any(
                seed_data.get('expert_grad_to_h1')
                for seed_data in all_data[c][N].values()
            )
            for c in config_group
            for N in all_data[c]
        )
        if not has_data:
            continue

        n_configs = len(config_group)
        fig, axes = plt.subplots(1, n_configs, figsize=(n_configs * onefigsize[0], onefigsize[1]), sharey=True)
        if n_configs == 1:
            axes = [axes]

        for ax_idx, config in enumerate(config_group):
            ax = axes[ax_idx]
            Ns = sorted(all_data[config].keys())
            if len(Ns) < 2:
                continue

            first_seed = list(all_data[config][Ns[0]].keys())[0]
            eg_data = all_data[config][Ns[0]][first_seed].get('expert_grad_to_h1')
            if not eg_data:
                continue

            T_total = T_total_override if T_total_override is not None else \
                all_data[config][Ns[0]][first_seed].get('config', {}).get('T', len(eg_data))
            timesteps = [2, 50, T_total//5, T_total-1] if T_total > 10 else list(range(2, T_total))
            blues = plt.cm.Blues(np.linspace(0.4, 0.95, len(timesteps)))

            for t_idx, t in enumerate(timesteps):
                seed_data_dict = {}  # {seed: {N: value}}
                for N in Ns:
                    for seed, seed_data in all_data[config][N].items():
                        eg = seed_data.get('expert_grad_to_h1')
                        if eg is not None and t < len(eg):
                            if seed not in seed_data_dict:
                                seed_data_dict[seed] = {}
                            seed_data_dict[seed][N] = eg[t]

                # Individual seeds (light)
                for seed, seed_points in seed_data_dict.items():
                    if len(seed_points) >= 2:
                        widths = np.array(list(seed_points.keys()))
                        values = np.array(list(seed_points.values()), dtype=float)
                        mask = np.isfinite(values) & (values > 0)
                        if mask.sum() >= 2:
                            ax.plot(widths[mask], values[mask], 'o', color=blues[t_idx], markersize=4, alpha=0.5)
                            try:
                                slope, intercept, *_ = stats.linregress(np.log10(widths[mask]), np.log10(values[mask]))
                                x_fit = np.logspace(np.log10(widths[mask].min()), np.log10(widths[mask].max()), 100)
                                ax.plot(x_fit, 10 ** intercept * x_fit ** slope, '-', color=blues[t_idx], alpha=0.5, linewidth=1.5)
                            except Exception:
                                pass

                # Mean fit across seeds
                if seed_data_dict:
                    all_widths = sorted(set(N for sp in seed_data_dict.values() for N in sp))
                    mean_values = [
                        np.mean([sp[N] for sp in seed_data_dict.values() if N in sp])
                        for N in all_widths
                    ]
                    widths_arr = np.array(all_widths)
                    values_arr = np.array(mean_values, dtype=float)
                    mask = np.isfinite(values_arr) & (values_arr > 0)
                    if mask.sum() >= 2:
                        try:
                            slope, intercept, *_ = stats.linregress(np.log10(widths_arr[mask]), np.log10(values_arr[mask]))
                            alpha_fit = round(slope, 2)
                            x_fit = np.logspace(np.log10(widths_arr[mask].min()), np.log10(widths_arr[mask].max()), 100)
                            ax.plot(x_fit, 10 ** intercept * x_fit ** alpha_fit, '-',
                                    color=blues[t_idx], label=f't={t} ({alpha_fit:.2f})', linewidth=2)
                        except Exception:
                            pass

            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel('Width N')
            ax.set_ylabel(r'$\|\sum_m \nabla_{h^1}\mathcal{L}\|_{\rm RMS}$ (experts)')
            ax.set_title(format_label(config))
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        rcc_dir = os.path.join(results_dir, 'rcc')
        os.makedirs(rcc_dir, exist_ok=True)
        plt.savefig(os.path.join(rcc_dir, f'expert_grad_to_h1_{group_name}.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_h_L_hists_coordinate(all_data, results_dir):
    """Plot h^L (combined) histograms for coordinate check at final iteration."""
    # Get filtered config groups
    config_groups = filter_configs_by_regime_and_optimizer(all_data)
    fixed_expert_sgd = config_groups['fixed_expert_sgd']
    fixed_expert_adam = config_groups['fixed_expert_adam']
    allscaling_sgd = config_groups['allscaling_sgd']
    allscaling_adam = config_groups['allscaling_adam']
    bottleneck_sgd = config_groups['bottleneck_sgd']
    bottleneck_adam = config_groups['bottleneck_adam']
    expert_scaling = config_groups['expert_scaling']

    for config_group, group_name in [
        (fixed_expert_sgd, 'fixed_E_sgd'),
        (fixed_expert_adam, 'fixed_E_adam'),
        (expert_scaling, 'other'),
        (allscaling_sgd, 'allscaling_sgd'),
        (allscaling_adam, 'allscaling_adam'),
        (bottleneck_sgd, 'bottleneck_sgd'),
        (bottleneck_adam, 'bottleneck_adam')
    ]:
        if not config_group:
            continue

        n_configs = len(config_group)
        fig, axes = plt.subplots(1, n_configs, figsize=(n_configs*onefigsize[0], onefigsize[1]), sharey=False)
        if n_configs == 1:
            axes = [axes]
        
        for ax_idx, config in enumerate(config_group):
            ax = axes[ax_idx]
            Ns = sorted(all_data[config].keys())
            
            # Plot for largest N
            N_max = Ns[-1]
            first_seed = list(all_data[config][N_max].keys())[0]
            h_L_hists = all_data[config][N_max][first_seed].get('h_L_hists')
            
            if h_L_hists and len(h_L_hists) > 0:
                final_hist = h_L_hists[-1]
                if isinstance(final_hist, tuple) and len(final_hist) == 2:
                    counts, edges = final_hist
                    centers = 0.5 * (edges[:-1] + edges[1:])
                    widths = edges[1:] - edges[:-1]
                    density = counts / (counts.sum() * widths + 1e-10)
                    ax.plot(centers, density, color='darkblue', linewidth=2, label=f'N={N_max}')
            
            ax.set_xlabel('h^L Activation')
            if ax_idx == 0:
                ax.set_ylabel('Density')
            ax.set_title(format_subplot_title(config))
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Note: Histogram uses sharey=False, so each subplot has individual ylims
        # Determine config name for single-config plots
        single_config_name = config_group[0] if n_configs == 1 else None

        plt.tight_layout()
        save_path = get_plot_save_path(results_dir, single_config_name,
                                       f'h_L_hists_{group_name}.png',
                                       is_joint_plot=(n_configs > 1))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def plot_g_phi_rms_coordinate(all_data, results_dir):
    """Plot g^phi RMS (gradient w.r.t. router weights) vs width at different timesteps."""
    from scipy import stats

    # Get filtered config groups
    config_groups = filter_configs_by_regime_and_optimizer(all_data)
    fixed_expert_sgd = config_groups['fixed_expert_sgd']
    fixed_expert_adam = config_groups['fixed_expert_adam']
    allscaling_sgd = config_groups['allscaling_sgd']
    allscaling_adam = config_groups['allscaling_adam']
    bottleneck_sgd = config_groups['bottleneck_sgd']
    bottleneck_adam = config_groups['bottleneck_adam']
    expert_scaling = config_groups['expert_scaling']

    for config_group, group_name in [
        (fixed_expert_sgd, 'fixed_E_sgd'),
        (fixed_expert_adam, 'fixed_E_adam'),
        (expert_scaling, 'other'),
        (allscaling_sgd, 'allscaling_sgd'),
        (allscaling_adam, 'allscaling_adam'),
        (bottleneck_sgd, 'bottleneck_sgd'),
        (bottleneck_adam, 'bottleneck_adam')
    ]:
        if not config_group:
            continue
        
        n_configs = len(config_group)
        fig, axes = plt.subplots(1, n_configs, figsize=(n_configs*onefigsize[0], onefigsize[1]), sharey=True)
        if n_configs == 1:
            axes = [axes]
        
        for ax_idx, config in enumerate(config_group):
            ax = axes[ax_idx]
            Ns = sorted(all_data[config].keys())
            if len(Ns) < 2:
                continue
            
            # Get g_phi_rms from first seed
            first_seed = list(all_data[config][Ns[0]].keys())[0]
            g_phi_data = all_data[config][Ns[0]][first_seed].get('g_phi_rms')
            if g_phi_data is None:
                continue
            
            T_total = all_data[config][Ns[0]][first_seed].get('config', {}).get('T', len(g_phi_data))
            timesteps = [2, 50, T_total//5, T_total-1] if T_total > 10 else list(range(1, T_total))
            blues = plt.cm.Blues(np.linspace(0.4, 0.95, len(timesteps)))
            
            for t_idx, t in enumerate(timesteps):
                # Collect all seed data
                seed_data_dict = {}  # {seed: {N: value}}
                for N in Ns:
                    for seed, seed_data in all_data[config][N].items():
                        g_phi_rms = seed_data.get('g_phi_rms')
                        if g_phi_rms is not None and t < len(g_phi_rms):
                            if seed not in seed_data_dict:
                                seed_data_dict[seed] = {}
                            seed_data_dict[seed][N] = g_phi_rms[t]
                
                # Plot individual seeds with alpha=0.5
                for seed, seed_points in seed_data_dict.items():
                    if len(seed_points) >= 2:
                        widths = np.array(list(seed_points.keys()))
                        values = np.array(list(seed_points.values()))
                        # Plot points
                        ax.plot(widths, values, 'o', color=blues[t_idx], markersize=4, alpha=0.5)
                        # Fit and plot line
                        log_widths, log_values = np.log10(widths), np.log10(values)
                        try:
                            slope, intercept, _, _, _ = stats.linregress(log_widths, log_values)
                            x_fit = np.logspace(np.log10(min(widths)), np.log10(max(widths)), 100)
                            ax.plot(x_fit, 10**intercept * x_fit**slope, '-', color=blues[t_idx], 
                                   alpha=0.5, linewidth=1.5)
                        except:
                            pass
                
                # Compute and plot mean fit
                if seed_data_dict:
                    all_widths = sorted(set(N for seed_points in seed_data_dict.values() for N in seed_points.keys()))
                    mean_values = []
                    for N in all_widths:
                        vals = [seed_points[N] for seed_points in seed_data_dict.values() if N in seed_points]
                        if vals:
                            mean_values.append(np.mean(vals))
                    
                    if len(all_widths) >= 2:
                        widths_arr = np.array(all_widths)
                        g_phi_arr = np.array(mean_values)
                        log_widths = np.log10(widths_arr)
                        log_g_phi = np.log10(g_phi_arr)
                        
                        try:
                            slope, intercept, _, _, _ = stats.linregress(log_widths, log_g_phi)
                            alpha_fit = np.round(slope, 2)
                            C_fit = 10**intercept
                            x_fit = np.logspace(np.log10(min(all_widths)), np.log10(max(all_widths)), 100)
                            ax.plot(x_fit, C_fit * x_fit**alpha_fit, '-', color=blues[t_idx], 
                                   label=f't={t} ({alpha_fit:.2f})', linewidth=2)
                        except:
                            pass
            
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xlabel('Width N')
            ax.set_ylabel(r'$\mathrm{RMS}(g^\phi)$')
            ax.set_title(format_subplot_title(config))
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # Set shared ylims for all subplots
        set_shared_ylim(axes)

        # Determine config name for single-config plots
        single_config_name = config_group[0] if n_configs == 1 else None

        plt.tight_layout()
        save_path = get_plot_save_path(results_dir, single_config_name,
                                       f'g_phi_rms_coordinate_check_{group_name}.png',
                                       is_joint_plot=(n_configs > 1))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def plot_output_gradient_coordinate(all_data, results_dir):
    """Plot output gradient RMS vs width at different timesteps."""
    from scipy import stats

    # Get filtered config groups
    config_groups = filter_configs_by_regime_and_optimizer(all_data)
    fixed_expert_sgd = config_groups['fixed_expert_sgd']
    fixed_expert_adam = config_groups['fixed_expert_adam']
    allscaling_sgd = config_groups['allscaling_sgd']
    allscaling_adam = config_groups['allscaling_adam']
    bottleneck_sgd = config_groups['bottleneck_sgd']
    bottleneck_adam = config_groups['bottleneck_adam']
    expert_scaling = config_groups['expert_scaling']

    for config_group, group_name in [
        (fixed_expert_sgd, 'fixed_E_sgd'),
        (fixed_expert_adam, 'fixed_E_adam'),
        (expert_scaling, 'other'),
        (allscaling_sgd, 'allscaling_sgd'),
        (allscaling_adam, 'allscaling_adam'),
        (bottleneck_sgd, 'bottleneck_sgd'),
        (bottleneck_adam, 'bottleneck_adam')
    ]:
        if not config_group:
            continue

        n_configs = len(config_group)
        fig, axes = plt.subplots(1, n_configs, figsize=(n_configs*onefigsize[0], onefigsize[1]), sharey=True)
        if n_configs == 1:
            axes = [axes]

        for ax_idx, config in enumerate(config_group):
            ax = axes[ax_idx]
            Ns = sorted(all_data[config].keys())
            if len(Ns) < 2:
                continue

            first_seed = list(all_data[config][Ns[0]].keys())[0]
            output_grad_data = all_data[config][Ns[0]][first_seed].get('output_grad_rms')
            if output_grad_data is None:
                continue
            
            T_total = all_data[config][Ns[0]][first_seed].get('config', {}).get('T', len(output_grad_data))
            timesteps = [2, 50, T_total//5, T_total-1] if T_total > 10 else list(range(1, T_total))
            blues = plt.cm.Blues(np.linspace(0.4, 0.95, len(timesteps)))
            
            for t_idx, t in enumerate(timesteps):
                seed_data_dict = {}
                for N in Ns:
                    for seed, seed_data in all_data[config][N].items():
                        output_grad_rms = seed_data.get('output_grad_rms')
                        if output_grad_rms is not None and t < len(output_grad_rms):
                            if seed not in seed_data_dict:
                                seed_data_dict[seed] = {}
                            seed_data_dict[seed][N] = output_grad_rms[t]
                
                for seed, seed_points in seed_data_dict.items():
                    if len(seed_points) >= 2:
                        widths = np.array(list(seed_points.keys()))
                        values = np.array(list(seed_points.values()))
                        ax.plot(widths, values, 'o', color=blues[t_idx], markersize=4, alpha=0.5)
                        log_widths, log_values = np.log10(widths), np.log10(values)
                        try:
                            slope, intercept, _, _, _ = stats.linregress(log_widths, log_values)
                            x_fit = np.logspace(np.log10(min(widths)), np.log10(max(widths)), 100)
                            ax.plot(x_fit, 10**intercept * x_fit**slope, '-', color=blues[t_idx], alpha=0.5, linewidth=1.5)
                        except:
                            pass
                
                if seed_data_dict:
                    all_widths = sorted(set(N for seed_points in seed_data_dict.values() for N in seed_points.keys()))
                    mean_values = []
                    for N in all_widths:
                        vals = [seed_points[N] for seed_points in seed_data_dict.values() if N in seed_points]
                        if vals:
                            mean_values.append(np.mean(vals))
                    
                    if len(all_widths) >= 2:
                        widths_arr = np.array(all_widths)
                        values_arr = np.array(mean_values)
                        log_widths = np.log10(widths_arr)
                        log_values = np.log10(values_arr)
                        
                        try:
                            slope, intercept, _, _, _ = stats.linregress(log_widths, log_values)
                            alpha_fit = np.round(slope, 2)
                            C_fit = 10**intercept
                            x_fit = np.logspace(np.log10(min(all_widths)), np.log10(max(all_widths)), 100)
                            ax.plot(x_fit, C_fit * x_fit**alpha_fit, '-', color=blues[t_idx], 
                                   label=f't={t} ({alpha_fit:.2f})', linewidth=2)
                        except:
                            pass
            
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xlabel('Width N')
            ax.set_ylabel(r'RMS(dL/d(output))')
            ax.set_title(format_subplot_title(config))
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # Set shared ylims for all subplots
        set_shared_ylim(axes)

        # Determine config name for single-config plots
        single_config_name = config_group[0] if n_configs == 1 else None

        plt.tight_layout()
        save_path = get_plot_save_path(results_dir, single_config_name,
                                       f'output_grad_coordinate_check_{group_name}.png',
                                       is_joint_plot=(n_configs > 1))
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()


def plot_joint_overview(all_data, results_dir, timestep_stride=5):
    """Create a joint overview figure combining training dynamics and exponent trajectories.

    Layout:
      Row 1 (4 subplots): Train loss | Train acc | RMS(h^L diff) | RMS(ψ)
      Legend A: N values in rocket_r colors
      Legend B: Layer lines only (right below A)
      Row 2 (3 subplots): Prop. exp (raw) | Eff. exp (raw) | Eff. exp (norm)
      Row 3 (3 subplots): h^agg decomp exp | ∇h¹ decomp exp | Gradnorm exp (raw)
                          (decomp subplots have per-subplot legends at lower center)

    One figure per config; filename derived from results_dir basename.
    """
    import matplotlib.gridspec as gridspec
    import matplotlib.lines as mlines
    from matplotlib.ticker import MultipleLocator

    rocket = sns.color_palette("rocket_r", as_cmap=True)
    # icml2022 bundle enables constrained_layout AND autolayout; both override subplots_adjust.
    plt.rcParams.update({"figure.constrained_layout.use": False,
                          "figure.autolayout": False})

    all_configs = sorted(all_data.keys())
    if not all_configs:
        return

    basename = os.path.basename(os.path.normpath(results_dir))
    stem = basename[len('coordinate_check_'):] if basename.startswith('coordinate_check_') else basename

    for config in all_configs:
        Ns = sorted(all_data[config].keys())
        if not Ns:
            continue

        # ── figure & gridspec ─────────────────────────────────────────────
        # height_ratios: data rows are r_d, single merged legend row is r_l.
        # hspace is fraction of average subplot height; keep small to avoid
        # excess vertical whitespace. wspace must be wide enough for long ylabels
        # in rows 2-3 (3 subplots across 12 cols) not to bleed into left neighbour.
        r_d, r_l = 10, 2
        fig_w = 4 * onefigsize[0]
        fig_h = 8.1

        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = gridspec.GridSpec(4, 12, figure=fig,
                               height_ratios=[r_d, r_l, r_d, r_d])
        fig.subplots_adjust(left=0.08, right=0.99, bottom=0.06, top=0.88,
                            hspace=0.55, wspace=1.25)

        # Row 0 — training dynamics (4 subplots, 3 cols each)
        ax_tl = fig.add_subplot(gs[0, 0:3])
        ax_ta = fig.add_subplot(gs[0, 3:6])
        ax_hl = fig.add_subplot(gs[0, 6:9])
        ax_ps = fig.add_subplot(gs[0, 9:12])

        # Row 1 — single combined legend row (N colours + layer colours side by side)
        ax_legA = fig.add_subplot(gs[1, :6])
        ax_legA.axis('off')
        ax_legB = fig.add_subplot(gs[1, 6:])
        ax_legB.axis('off')

        # Row 2 — layer exponent trajectories (3 subplots, 4 cols each)
        ax_pr = fig.add_subplot(gs[2, 0:4])
        ax_er = fig.add_subplot(gs[2, 4:8])
        ax_en = fig.add_subplot(gs[2, 8:12])

        # Row 3 — decomp + gradnorm exponent trajectories (3 subplots, 4 cols each)
        ax_hd = fig.add_subplot(gs[3, 0:4])
        ax_gh = fig.add_subplot(gs[3, 4:8])
        ax_gn = fig.add_subplot(gs[3, 8:12])

        # ── Row 1: training dynamics ───────────────────────────────────────
        leg_A_handles = []
        for N_idx, N in enumerate(Ns):
            seeds = all_data[config][N]
            sd = seeds.get(42) or next(iter(seeds.values()))
            darkness = 0.3 + 0.6 * N_idx / max(1, len(Ns) - 1)
            color = rocket(darkness)

            tl = sd.get('train_loss') or []
            ta = sd.get('train_acc')  or []
            hl = sd.get('h_L_rms_diff') or []
            ps = sd.get('psi_norm')   or []

            if tl:
                ax_tl.semilogy(tl, color=color, linewidth=2)
            if ta:
                ax_ta.plot(ta, color=color, linewidth=2)
            if hl:
                ax_hl.plot(hl, color=color, linewidth=2)
            if ps:
                ax_ps.plot(ps, color=color, linewidth=2)

            leg_A_handles.append(
                mlines.Line2D([], [], color=color, linewidth=2, label=f'N={N}')
            )

        ax_tl.set_xlabel('Step'); ax_tl.set_ylabel('Training Loss')
        ax_ta.set_xlabel('Step'); ax_ta.set_ylabel('Train Acc')
        ax_hl.set_xlabel('Step'); ax_hl.set_ylabel(r'Feature Learning RMS($h^L_t - h^L_0$)')
        ax_ps.set_xlabel('Step'); ax_ps.set_ylabel(r'Router Logits RMS($\psi$)')
        for ax in [ax_tl, ax_ta, ax_hl, ax_ps]:
            ax.grid(True, alpha=0.3)

        # Legend A — N values
        ax_legA.legend(
            handles=leg_A_handles, loc='center',
            ncol=len(leg_A_handles), fontsize=9, frameon=True,
            title='Width', title_fontsize=11,
        )

        # ── Row 2: layer exponent trajectories ────────────────────────────
        _row2_specs = [
            (ax_pr, 'propagating_updates',               'raw',        _EXPTRAJ_YLABEL[('propagating_updates',               'raw')]),
            (ax_er, 'total_effective_updates_per_layer', 'raw',        _EXPTRAJ_YLABEL[('total_effective_updates_per_layer', 'raw')]),
            (ax_en, 'total_effective_updates_per_layer', 'normalized', _EXPTRAJ_YLABEL[('total_effective_updates_per_layer', 'normalized')]),
        ]
        # Regime detection (used in row 2 and gradnorm reference lines)
        _cfg_lower = config.lower()
        _is_bottleneck = 'bottleneck' in _cfg_lower
        _is_allscaling = 'allscaling' in _cfg_lower or 'allscale' in _cfg_lower
        _is_bottleneck_mssp = (_is_bottleneck and 'stdinit' not in _cfg_lower
                               and 'heuristic' not in _cfg_lower
                               and 'jiang' not in _cfg_lower
                               and 'globaleps' not in _cfg_lower)

        seen_layers = set()
        for ax, update_type, data_type, ylabel in _row2_specs:
            traj = _compute_layer_traj(all_data, config, update_type, data_type, timestep_stride)
            for layer, (ts_vals, exp_vals, std_vals) in traj.items():
                if not ts_vals:
                    continue
                color = _LAYER_COLORS.get(layer, 'black')
                ts_arr  = np.array(ts_vals)
                exp_arr = np.array(exp_vals)
                std_arr = np.array(std_vals)
                ax.plot(ts_arr, exp_arr, color=color, linewidth=1.5)
                ax.fill_between(ts_arr, exp_arr - 2 * std_arr, exp_arr + 2 * std_arr,
                                color=color, alpha=0.35)
                seen_layers.add(layer)
            ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.set_xlabel('Step')
            ax.set_ylabel(ylabel)
            ax.set_ylim(-1, 1)
            ax.yaxis.set_major_locator(MultipleLocator(0.5))
            ax.grid(True, alpha=0.3)

        # Propagating updates: W_exp2 expected = 0.5 for bottleneck MSSP only
        if _is_bottleneck_mssp:
            _color_exp2 = _LAYER_COLORS.get('W_exp2', 'black')
            ax_pr.axhline(y=0.5, color=_color_exp2, linestyle='--',
                          linewidth=1.0, alpha=0.5, zorder=1)

        # ── Row 3: decomp subplots with per-subplot legends ────────────────
        for decomp_type, ax in [('hagg_decomp', ax_hd), ('grad_h1_decomp', ax_gh)]:
            traj = _compute_decomp_traj(all_data, config, decomp_type, timestep_stride)
            term_labels_map = next(tl for dt, tl in _DECOMP_TRAJ_TYPES if dt == decomp_type)
            subplot_handles = []
            for term, (ts_vals, exp_vals, std_vals) in traj.items():
                if not ts_vals:
                    continue
                color = _DECOMP_TERM_COLORS.get(term, 'black')
                label = term_labels_map.get(term, term)
                ts_arr  = np.array(ts_vals)
                exp_arr = np.array(exp_vals)
                std_arr = np.array(std_vals)
                if term == 'base':
                    ax.scatter(ts_arr, exp_arr, marker='x', s=60,
                               color=color, linewidths=1.5, zorder=5)
                    subplot_handles.append(
                        mlines.Line2D([], [], color=color, marker='x', linestyle='none',
                                      markersize=6, markeredgewidth=1.5, label=label)
                    )
                else:
                    ax.plot(ts_arr, exp_arr, color=color, linewidth=1.5)
                    ax.fill_between(ts_arr, exp_arr - 2 * std_arr, exp_arr + 2 * std_arr,
                                    color=color, alpha=0.35)
                    subplot_handles.append(
                        mlines.Line2D([], [], color=color, linewidth=1.5, label=label)
                    )
            ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.set_xlabel('Step')
            ax.set_ylabel(_DECOMP_TRAJ_YLABEL.get(decomp_type, 'Exponent'))
            ax.set_ylim(-2, 1)
            ax.yaxis.set_major_locator(MultipleLocator(0.5))
            ax.grid(True, alpha=0.3)
            if subplot_handles:
                _leg_loc = 'upper center' if decomp_type == 'grad_h1_decomp' else 'lower center'
                ax.legend(handles=subplot_handles, fontsize=9, loc=_leg_loc,
                          frameon=True, title='Decomp. term', title_fontsize=11)

        # Gradnorm (raw) for ax_gn
        traj_gn = _compute_layer_traj(all_data, config, 'grad_norms', 'raw', timestep_stride)
        for layer, (ts_vals, exp_vals, std_vals) in traj_gn.items():
            if not ts_vals:
                continue
            color = _LAYER_COLORS.get(layer, 'black')
            ts_arr  = np.array(ts_vals)
            exp_arr = np.array(exp_vals)
            std_arr = np.array(std_vals)
            ax_gn.plot(ts_arr, exp_arr, color=color, linewidth=1.5)
            ax_gn.fill_between(ts_arr, exp_arr - 2 * std_arr, exp_arr + 2 * std_arr,
                                color=color, alpha=0.35)
            seen_layers.add(layer)
        # Expected exponent reference lines for regime II (bottleneck) and III (allscaling)
        if _is_bottleneck or _is_allscaling:
            _expected_gn = {
                'W_in':   -1,
                'Q':      -1,
                'W_exp1': -1 if _is_bottleneck else -2,
                'W_exp2': -2,
                'W_out':   0,
            }
            _plotted_layers = {l for l, (ts, _, __) in traj_gn.items() if ts}
            # Group layers by expected value, then stagger ±0.02 to avoid overlap
            from collections import defaultdict as _dd
            _groups = _dd(list)
            for _l in _LAYER_ORDER:
                if _l in _expected_gn and _l in _plotted_layers:
                    _groups[_expected_gn[_l]].append(_l)
            for _exp_val, _layers in _groups.items():
                _n = len(_layers)
                for _i, _layer in enumerate(_layers):
                    if _n == 1:
                        _offset = 0.0
                    elif _n == 2:
                        _offset = -0.02 if _i == 0 else 0.02
                    else:
                        _offset = -0.02 + 0.02 * _i
                    _color = _LAYER_COLORS.get(_layer, 'black')
                    ax_gn.axhline(y=_exp_val + _offset, color=_color, linestyle='--',
                                  linewidth=1.0, alpha=0.5, zorder=1)
        ax_gn.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        ax_gn.set_xlabel('Step')
        ax_gn.set_ylabel(_EXPTRAJ_YLABEL[('grad_norms', 'raw')])
        ax_gn.set_ylim(-3, 0.5)
        ax_gn.yaxis.set_major_locator(MultipleLocator(0.5))
        ax_gn.grid(True, alpha=0.3)

        # ── Legend B: layer lines only (right below A) ─────────────────────
        layer_handles = [
            mlines.Line2D([], [], color=_LAYER_COLORS[l], linewidth=1.5,
                          label=_LAYER_DISPLAY.get(l, l))
            for l in _LAYER_ORDER if l in seen_layers
        ]
        if layer_handles:
            ax_legB.legend(
                handles=layer_handles, loc='center',
                ncol=len(layer_handles), fontsize=9, frameon=True,
                title='Layer', title_fontsize=11,
            )

        # ── Global title & save ────────────────────────────────────────────
        # Parse variant flags from the directory name (shared experts, routing,
        # router init, last-layer init) — these are runtime flags absent from config.
        _d = os.path.basename(os.path.normpath(results_dir))
        _flags = []
        if 'sharedexp' in _d or 'bothshared' in _d:
            _flags.append('shared experts')
        elif any(x in _d for x in ['allscaling', 'allscale']):
            _flags.append('no shared experts')
        _topk_re = re.search(r'topk(\d+)', _d)
        if _topk_re:
            _flags.append(f'top-{_topk_re.group(1)} routing')
        elif '_topk_' in _d or _d.endswith('_topk'):
            # k not in dirname — read minimum topk from config JSONs
            _cfg_jsons = glob.glob(os.path.join(results_dir, '**', 'config_*.json'), recursive=True)
            _topk_vals = []
            for _cj in _cfg_jsons:
                try:
                    import json as _json
                    with open(_cj) as _f:
                        _topk_vals.append(_json.load(_f).get('topk', 0))
                except Exception:
                    pass
            _base_k = min((v for v in _topk_vals if v > 0), default=None)
            _flags.append(f'top-{_base_k} routing' if _base_k else 'top-k routing')
        elif 'soft' in _d:
            _flags.append('soft routing')
        if 'rinitzero' in _d or 'routerzeroinit' in _d:
            _flags.append('router init=0')
        elif 'routerinit1overN' in _d or 'routerinit1overN' in _d:
            _flags.append('router init=1/N')
        if 'll1overN' in _d or 'lln' in _d.lower():
            _flags.append('last-layer init=1/N')
        elif 'llzero' in _d:
            _flags.append('last-layer init=0')
        _subtitle = ', '.join(_flags)
        _base_title = get_figure_title(config, results_dir)
        _title = _base_title + (f'\n{_subtitle}' if _subtitle else '')
        fig.suptitle(_title, fontsize=18, y=0.99)

        suffix = f'_{config}' if len(all_configs) > 1 else ''
        save_path = os.path.join(results_dir, f'joint_rcc_{stem}{suffix}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved joint overview to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True, help="Results directory")
    parser.add_argument("--T_total", type=int, default=None, help="Override T_total for timestep selection (default: use from data)")
    parser.add_argument("--timesteps", type=int, nargs='+', default=None, help="Timesteps for exponent tables (e.g., 2 3 50 499)")
    parser.add_argument("--separate_routing_and_init", action='store_true',
                        help="Create separate virtual configs for different routing modes and router inits (for ablation studies)")
    args = parser.parse_args()

    print(f"Loading data from {args.results_dir}...")
    all_data = load_all_data(args.results_dir, separate_routing_and_init=args.separate_routing_and_init)
    
    print("Plotting effective updates...")
    plot_effective_updates(all_data, args.results_dir, T_total_override=args.T_total)

    print("Plotting total effective updates per layer...")
    plot_total_effective_updates_per_layer(all_data, args.results_dir, T_total_override=args.T_total)
    
    print("Plotting gradient norms...")
    plot_gradient_norms(all_data, args.results_dir, T_total_override=args.T_total)

    print("Plotting weight RMS norms...")
    plot_weight_rms_norms(all_data, args.results_dir, T_total_override=args.T_total)

    print("Plotting propagating updates...")
    plot_propagating_updates(all_data, args.results_dir, T_total_override=args.T_total)
    
    print("Computing exponent tables...")
    compute_exponent_table(all_data, args.results_dir, timesteps=args.timesteps)

    print("Computing decomposition exponent tables...")
    compute_decomp_exponent_tables(all_data, args.results_dir, timesteps=args.timesteps)

    print("Plotting exponent trajectories over time...")
    plot_exponent_trajectories(all_data, args.results_dir)

    print("Plotting decomposition exponent trajectories over time...")
    plot_decomp_exponent_trajectories(all_data, args.results_dir)
    
    print("Plotting h^L coordinate check...")
    plot_h_L_coordinate_check(all_data, args.results_dir)
    
    print("Plotting activation RMS...")
    plot_activation_rms(all_data, args.results_dir)
    
    print("Plotting h^L histograms...")
    plot_h_L_hists_coordinate(all_data, args.results_dir)
    
    print("Plotting g^phi RMS...")
    plot_g_phi_rms_coordinate(all_data, args.results_dir)
    
    print("Plotting output gradient RMS...")
    plot_output_gradient_coordinate(all_data, args.results_dir)

    print("Plotting h^agg decomposition RCC...")
    plot_hagg_decomp_rcc(all_data, args.results_dir, T_total_override=args.T_total)

    print("Plotting grad_h1 decomposition RCC...")
    plot_grad_h1_decomp_rcc(all_data, args.results_dir, T_total_override=args.T_total)

    print("Plotting loss gradient RMS...")
    plot_loss_grad_rms(all_data, args.results_dir, T_total_override=args.T_total)

    print("Plotting expert grad to h1...")
    plot_expert_grad_to_h1(all_data, args.results_dir, T_total_override=args.T_total)

    print("Generating exponent bar charts...")
    import subprocess
    for update_type in ['effective_updates', 'propagating_updates', 'grad_norms', 'total_effective_updates_per_layer']:
        for data_type in ['normalized', 'raw']:
            # Collect all timesteps from CSV files
            import glob
            csv_files = glob.glob(os.path.join(args.results_dir, f'exponent_table_{update_type}_t*.csv'))
            if not csv_files:
                continue
            
            # Extract timesteps from filenames
            timesteps = []
            for csv_file in csv_files:
                timestep = os.path.basename(csv_file).replace(f'exponent_table_{update_type}_', '').replace('.csv', '')
                timestep_num = timestep.replace('t', '')
                try:
                    timesteps.append(int(timestep_num))
                except ValueError:
                    pass
            
            if not timesteps:
                continue
            
            # Call plot_rcc_exponents.py once with all timesteps
            try:
                subprocess.run(['python', 'plotting/plot_rcc_exponents.py',
                              '--results_dir', args.results_dir,
                              '--update_type', update_type,
                              '--timesteps'] + [str(t) for t in sorted(timesteps)] +
                              ['--data_type', data_type], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to generate bar chart for {update_type} {data_type}: {e}")

    print("Generating decomposition exponent bar charts...")
    for decomp_type in ['hagg_decomp', 'grad_h1_decomp']:
        csv_files = glob.glob(os.path.join(args.results_dir, f'exponent_table_{decomp_type}_t*.csv'))
        if csv_files:
            try:
                subprocess.run(['python', 'plotting/plot_rcc_exponents.py',
                              '--results_dir', args.results_dir,
                              '--update_type', decomp_type], check=True)
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to generate decomp exponent chart for {decomp_type}: {e}")

    print("Plotting training dynamics...")
    try:
        subprocess.run(['python', 'plotting/plot_training_dynamics.py', '--results_dir', args.results_dir], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to generate training dynamics plots: {e}")

    print("Plotting joint overview figures...")
    plot_joint_overview(all_data, args.results_dir)

    print("Plotting complete!")
    print(f"✓ Plots and tables generated in {args.results_dir}/")
