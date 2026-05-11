#!/usr/bin/env python3
"""Plot training dynamics matching jax implementation style."""

import argparse
import numpy as np
import matplotlib.pyplot as plt
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

try:
    import seaborn as sns
    rocket = sns.color_palette("rocket_r", as_cmap=True)
except ImportError:
    rocket = plt.cm.viridis

from label_utils import (
    format_label, get_optimizer_from_config,
    get_scaling_regime, format_latex_caption,
    get_figure_title,
)

def get_plot_save_path(results_dir, config_name, filename, is_joint_plot=True):
    """Determine where to save a plot.

    Args:
        results_dir: Base results directory
        config_name: Name of the config (for single-config plots) or None
        filename: Name of the file to save
        is_joint_plot: True if this is a multi-config comparison plot, False for single-config

    Returns:
        Path object for where to save the plot
    """
    from pathlib import Path

    # All plots go directly to the root results directory
    plot_dir = Path(results_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    return plot_dir / filename

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
            # Convert to numpy array if needed and filter out inf/nan values
            ydata = np.asarray(ydata)
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

def group_configs_by_scaling(all_data):
    """Group configs into fixed_E, allscaling, and other, separated by optimizer"""
    groups = {
        'fixed_E_sgd': {}, 'fixed_E_adam': {},
        'allscaling_sgd': {}, 'allscaling_adam': {},
        'bottleneck_sgd': {}, 'bottleneck_adam': {},
        'other': {}
    }

    for config, data in all_data.items():
        optimizer = get_optimizer_from_config(config)

        if 'fixed_E' in config:
            key = f'fixed_E_{optimizer.lower()}'
            groups[key][config] = data
        elif 'bottleneck' in config:
            key = f'bottleneck_{optimizer.lower()}'
            groups[key][config] = data
        elif 'allscaling' in config:
            key = f'allscaling_{optimizer.lower()}'
            groups[key][config] = data
        else:
            groups['other'][config] = data

    return groups

def load_all_data(results_dir, preferred_seed=42):
    """Load all results from config directories, using consistent seed across widths.

    Args:
        results_dir: Base results directory
        preferred_seed: Seed to use consistently across all widths (default: 42)
    """
    results_path = Path(results_dir)
    all_data = {}

    # Reserved directory names that should not be treated as config directories
    RESERVED_DIRS = {'plots', 'rcc', 'figures', 'tables'}

    for config_dir in results_path.iterdir():
        if not config_dir.is_dir() or config_dir.name.startswith('.'):
            continue

        config_name = config_dir.name

        # Skip reserved directory names to prevent nested plots/plots/ structure
        if config_name in RESERVED_DIRS:
            continue

        stats_dir = config_dir / 'stats'
        if not stats_dir.exists():
            continue

        # Group files by N and seed
        files_by_N = {}
        for f in glob.glob(str(stats_dir / 'nn_N*.npz')):
            basename = os.path.basename(f)
            N = int(basename.split('_N')[1].split('_')[0])

            # Extract seed if present in filename
            seed = None
            if '_seed' in basename:
                try:
                    seed = int(basename.split('_seed')[1].split('_')[0].split('.')[0])
                except (ValueError, IndexError):
                    seed = None

            if N not in files_by_N:
                files_by_N[N] = []
            files_by_N[N].append((f, seed))

        # For each N, pick the file with preferred_seed if available, otherwise the first file
        all_data[config_name] = {}
        for N, file_list in files_by_N.items():
            # Try to find file with preferred seed
            selected_file = None
            for f, seed in file_list:
                if seed == preferred_seed:
                    selected_file = f
                    break

            # If preferred seed not found, use the first file (and warn if multiple seeds exist)
            if selected_file is None:
                if len(file_list) > 1:
                    seeds_available = [s for _, s in file_list if s is not None]
                    print(f"  Warning: {config_name} N={N} - preferred seed {preferred_seed} not found. "
                          f"Available seeds: {seeds_available}. Using first file.")
                selected_file = file_list[0][0]

            data = np.load(selected_file, allow_pickle=True)
            all_data[config_name][N] = data

    return all_data

def plot_losses(all_data, results_dir, group_name=''):
    """Plot training and validation losses"""
    configs = sorted(all_data.keys())
    n_configs = len(configs)
    
    # Plot train loss
    fig, axes = plt.subplots(1, n_configs, figsize=(n_configs*onefigsize[0], onefigsize[1]), sharey=True)
    if n_configs == 1:
        axes = [axes]
    
    for ax_idx, config in enumerate(configs):
        ax = axes[ax_idx]
        Ns = sorted(all_data[config].keys())
        
        for N_idx, N in enumerate(Ns):
            darkness = 0.3 + 0.6 * N_idx / max(1, len(Ns)-1)
            data = all_data[config][N]
            if 'train_loss' in data:
                ax.semilogy(data['train_loss'], color=rocket(darkness), label=f'N={N}', linewidth=2)
        
        ax.set_xlabel('Iteration')
        if ax_idx == 0:
            ax.set_ylabel('Training Loss')
        ax.set_title(get_figure_title(config, results_dir))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    set_shared_ylim(axes)
    plt.tight_layout()
    suffix = f'_{group_name}' if group_name else ''
    single_config_name = configs[0] if n_configs == 1 else None
    save_path = get_plot_save_path(results_dir, single_config_name,
                                    f'train_loss{suffix}.png',
                                    is_joint_plot=(n_configs > 1))
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot val loss
    fig, axes = plt.subplots(1, n_configs, figsize=(n_configs*onefigsize[0], onefigsize[1]), sharey=True)
    if n_configs == 1:
        axes = [axes]
    
    for ax_idx, config in enumerate(configs):
        ax = axes[ax_idx]
        Ns = sorted(all_data[config].keys())
        
        for N_idx, N in enumerate(Ns):
            darkness = 0.3 + 0.6 * N_idx / max(1, len(Ns)-1)
            data = all_data[config][N]
            if 'val_loss' in data:
                ax.semilogy(data['val_loss'], color=rocket(darkness), label=f'N={N}', linewidth=2)
        
        ax.set_xlabel('Iteration')
        if ax_idx == 0:
            ax.set_ylabel('Validation Loss')
        ax.set_title(get_figure_title(config, results_dir))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    set_shared_ylim(axes)
    plt.tight_layout()
    single_config_name = configs[0] if n_configs == 1 else None
    save_path = get_plot_save_path(results_dir, single_config_name,
                                    f'val_loss{suffix}.png',
                                    is_joint_plot=(n_configs > 1))
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_concentration(all_data, results_dir, group_name=''):
    """Plot max and min concentration on same plot with min as dotted"""
    configs = sorted(all_data.keys())
    n_configs = len(configs)
    
    fig, axes = plt.subplots(1, n_configs, figsize=(n_configs*onefigsize[0], onefigsize[1]), sharey=True)
    if n_configs == 1:
        axes = [axes]
    
    for ax_idx, config in enumerate(configs):
        ax = axes[ax_idx]
        Ns = sorted(all_data[config].keys())
        
        for N_idx, N in enumerate(Ns):
            darkness = 0.3 + 0.6 * N_idx / max(1, len(Ns)-1)
            color = rocket(darkness)
            data = all_data[config][N]
            
            if 'max_concentration' in data:
                ax.plot(data['max_concentration'], color=color, linestyle='-', linewidth=2, label=f'N={N}')
            if 'min_concentration' in data:
                ax.plot(data['min_concentration'], color=color, linestyle=':', linewidth=2)
        
        ax.axhline(0.5, color='red', ls='--', alpha=0.5, linewidth=1)
        ax.axhline(1.0, color='gray', ls=':', alpha=0.5)
        ax.set_xlabel('Iteration')
        if ax_idx == 0:
            ax.set_ylabel('Concentration')
        ax.set_title(get_figure_title(config, results_dir))
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    set_shared_ylim(axes)
    plt.tight_layout()
    suffix = f'_{group_name}' if group_name else ''
    single_config_name = configs[0] if n_configs == 1 else None
    save_path = get_plot_save_path(results_dir, single_config_name,
                                    f'concentration{suffix}.png',
                                    is_joint_plot=(n_configs > 1))
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_router_weights(all_data, results_dir, group_name=''):
    """Plot router max and min weights on same plot with min as dotted"""
    configs = sorted(all_data.keys())
    n_configs = len(configs)
    
    fig, axes = plt.subplots(1, n_configs, figsize=(n_configs*onefigsize[0], onefigsize[1]), sharey=True)
    if n_configs == 1:
        axes = [axes]
    
    for ax_idx, config in enumerate(configs):
        ax = axes[ax_idx]
        Ns = sorted(all_data[config].keys())
        
        for N_idx, N in enumerate(Ns):
            darkness = 0.3 + 0.6 * N_idx / max(1, len(Ns)-1)
            color = rocket(darkness)
            data = all_data[config][N]
            
            if 'router_max' in data:
                ax.plot(data['router_max'], color=color, linestyle='-', linewidth=2, label=f'N={N}')
            if 'router_min' in data:
                ax.plot(data['router_min'], color=color, linestyle=':', linewidth=2)
        
        ax.set_xlabel('Iteration')
        if ax_idx == 0:
            ax.set_ylabel('Router Weight')
        ax.set_title(get_figure_title(config, results_dir))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    set_shared_ylim(axes)
    plt.tight_layout()
    suffix = f'_{group_name}' if group_name else ''
    single_config_name = configs[0] if n_configs == 1 else None
    save_path = get_plot_save_path(results_dir, single_config_name,
                                    f'router_weights{suffix}.png',
                                    is_joint_plot=(n_configs > 1))
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_psi_norm(all_data, results_dir, group_name=''):
    """Plot RMS(ψ) (router logit RMS norm)"""
    configs = sorted(all_data.keys())
    n_configs = len(configs)
    
    fig, axes = plt.subplots(1, n_configs, figsize=(n_configs*onefigsize[0], onefigsize[1]), sharey=True)
    if n_configs == 1:
        axes = [axes]
    
    for ax_idx, config in enumerate(configs):
        ax = axes[ax_idx]
        Ns = sorted(all_data[config].keys())
        
        for N_idx, N in enumerate(Ns):
            darkness = 0.3 + 0.6 * N_idx / max(1, len(Ns)-1)
            data = all_data[config][N]
            
            if 'psi_norm' in data:
                ax.plot(data['psi_norm'], color=rocket(darkness), linewidth=2, label=f'N={N}')
        
        ax.set_ylim(bottom=0)
        ax.set_xlabel('Iteration')
        if ax_idx == 0:
            ax.set_ylabel(r'Router logits $\mathrm{RMS}(\psi)$')
        ax.set_title(get_figure_title(config, results_dir))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    set_shared_ylim(axes)
    plt.tight_layout()
    suffix = f'_{group_name}' if group_name else ''
    single_config_name = configs[0] if n_configs == 1 else None
    save_path = get_plot_save_path(results_dir, single_config_name,
                                    f'psi_norm{suffix}.png',
                                    is_joint_plot=(n_configs > 1))
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def generate_latex_figure(config_name, results_dir, group_name=''):
    """Generate LaTeX figure code for coordinate check results.

    Args:
        config_name: Configuration name for generating caption
        results_dir: Directory containing the plots
        group_name: Optional group name suffix for plot filenames
    """
    # Generate caption
    caption = format_latex_caption(config_name)

    # Generate label for cross-referencing
    regime = get_scaling_regime(config_name)
    optimizer = get_optimizer_from_config(config_name).lower()
    label = f"fig:coord_check_{regime.lower().replace(' ', '_')}_{optimizer}"

    # Determine plot filenames
    suffix = f'_{group_name}' if group_name else ''

    # Get scaling regime for naming (used in filenames)
    regime = get_scaling_regime(config_name)
    if regime == 'Fixed M':
        regime_abbr = 'fixed_E'
    elif regime == 'All-scaling':
        regime_abbr = 'allscaling'
    elif regime == 'Bottleneck':
        regime_abbr = 'bottleneck'
    else:
        regime_abbr = 'other'

    # Determine optimizer suffix for training dynamics plots
    optimizer_suffix = f'_{regime_abbr}_{optimizer}'

    # Get relative path - use plots/ prefix for LaTeX
    results_path = Path(results_dir)
    output_dir = f'plots/{results_path.name}/'

    # Generate LaTeX code
    latex_code = r"""\begin{figure*}[t]
    \centering
    \begin{subfigure}[t]{0.32\textwidth}
        \centering
        \includegraphics[width=\textwidth]{""" + output_dir + f"""train_loss{suffix}{optimizer_suffix}.png}}
    \\end{{subfigure}}
    \\hfill
    \\begin{{subfigure}}[t]{{0.32\\textwidth}}
        \\centering
        \\includegraphics[width=\\textwidth]{{""" + output_dir + f"""val_acc{suffix}{optimizer_suffix}.png}}
    \\end{{subfigure}}
    \\hfill
    \\begin{{subfigure}}[t]{{0.32\\textwidth}}
        \\centering
        \\includegraphics[width=\\textwidth]{{""" + output_dir + f"""h_L_rms_diff{suffix}{optimizer_suffix}.png}}
    \\end{{subfigure}}

    \\vspace{{0.3cm}}

    \\begin{{subfigure}}[t]{{0.32\\textwidth}}
        \\centering
        \\includegraphics[width=\\textwidth]{{""" + output_dir + f"""psi_norm{suffix}{optimizer_suffix}.png}}
    \\end{{subfigure}}
    \\hfill
    \\begin{{subfigure}}[t]{{0.32\\textwidth}}
        \\centering
        \\includegraphics[width=\\textwidth]{{""" + output_dir + f"""concentration{suffix}{optimizer_suffix}.png}}
    \\end{{subfigure}}
    \\hfill
    \\begin{{subfigure}}[t]{{0.32\\textwidth}}
        \\centering
        \\includegraphics[width=\\textwidth]{{""" + output_dir + f"""expert_hists{suffix}{optimizer_suffix}.png}}
    \\end{{subfigure}}

    \\vspace{{0.3cm}}

    \\begin{{subfigure}}[t]{{0.24\\textwidth}}
        \\centering
        \\includegraphics[width=\\textwidth]{{""" + output_dir + f"""rcc_exponents_total_effective_updates_per_layer_t2_3_50_499_raw{suffix}.png}}
    \\end{{subfigure}}
    \\hfill
    \\begin{{subfigure}}[t]{{0.24\\textwidth}}
        \\centering
        \\includegraphics[width=\\textwidth]{{""" + output_dir + f"""rcc_exponents_total_effective_updates_per_layer_t2_3_50_499_normalized{suffix}.png}}
    \\end{{subfigure}}
    \\hfill
    \\begin{{subfigure}}[t]{{0.24\\textwidth}}
        \\centering
        \\includegraphics[width=\\textwidth]{{""" + output_dir + f"""rcc_exponents_propagating_updates_t2_3_50_499_raw{suffix}.png}}
    \\end{{subfigure}}
    \\hfill
    \\begin{{subfigure}}[t]{{0.24\\textwidth}}
        \\centering
        \\includegraphics[width=\\textwidth]{{""" + output_dir + f"""rcc_exponents_grad_norms_t2_3_50_499_raw{suffix}.png}}
    \\end{{subfigure}}

    \\caption{{""" + caption + r"""}}
    \label{""" + label + r"""}
\end{figure*}
"""

    # Save to file
    output_file = results_path / f"figure_{config_name}.tex"
    with open(output_file, 'w') as f:
        f.write(latex_code)

    print(f"  LaTeX figure saved to: {output_file}")
    return latex_code

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--seed", type=int, default=42, help="Seed to use consistently across all widths (default: 42)")
    args = parser.parse_args()

    print(f"Loading data from {args.results_dir}...")
    print(f"Using seed {args.seed} for all widths...")
    all_data = load_all_data(args.results_dir, preferred_seed=args.seed)

    if not all_data:
        print("No data found!")
        return

    groups = group_configs_by_scaling(all_data)

    print("Plotting training dynamics...")
    for group_name, group_data in groups.items():
        if not group_data:
            continue
        plot_losses(group_data, args.results_dir, group_name)
        plot_accuracies(group_data, args.results_dir, group_name)
        plot_concentration(group_data, args.results_dir, group_name)
        plot_router_weights(group_data, args.results_dir, group_name)
        plot_psi_norm(group_data, args.results_dir, group_name)
        plot_entropy(group_data, args.results_dir, group_name)
        plot_expert_l2_diffs(group_data, args.results_dir, group_name)
        plot_h_L_rms_diff(group_data, args.results_dir, group_name)
        plot_expert_hists(group_data, args.results_dir, group_name)
        plot_h_L_hists(group_data, args.results_dir, group_name)
        plot_output_gradient(group_data, args.results_dir, group_name)
        plot_hagg_decomposition(group_data, args.results_dir, group_name)
        plot_expert_grad_h1_decomposition(group_data, args.results_dir, group_name)

    print(f"\n✓ Plots saved to {args.results_dir}/")

    # Generate LaTeX figures for each config
    print("\nGenerating LaTeX figures...")
    for config_name in all_data.keys():
        # Determine which group this config belongs to
        group_name = ''
        for gname, gdata in groups.items():
            if config_name in gdata:
                group_name = gname
                break
        generate_latex_figure(config_name, args.results_dir, group_name)

    print("\n✓ LaTeX figures generated!")

def plot_accuracies(all_data, results_dir, group_name=''):
    """Plot training and validation accuracy"""
    configs = sorted(all_data.keys())
    n_configs = len(configs)

    # Plot train acc
    fig, axes = plt.subplots(1, n_configs, figsize=(n_configs*onefigsize[0], onefigsize[1]), sharey=True)
    if n_configs == 1:
        axes = [axes]

    for ax_idx, config in enumerate(configs):
        ax = axes[ax_idx]
        Ns = sorted(all_data[config].keys())

        for N_idx, N in enumerate(Ns):
            darkness = 0.3 + 0.6 * N_idx / max(1, len(Ns)-1)
            data = all_data[config][N]
            if 'train_acc' in data:
                ax.plot(data['train_acc'], color=rocket(darkness), linewidth=2, label=f'N={N}')

        ax.set_xlabel('Iteration')
        if ax_idx == 0:
            ax.set_ylabel('Training Accuracy')
        ax.set_title(get_figure_title(config, results_dir))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Set adaptive ylim but clamp to [0, 1] for accuracy
    set_shared_ylim(axes)
    for ax in axes:
        ylim = ax.get_ylim()
        ax.set_ylim(max(0.0, ylim[0]), min(1.0, ylim[1]))
    plt.tight_layout()
    suffix = f'_{group_name}' if group_name else ''
    single_config_name = configs[0] if n_configs == 1 else None
    save_path = get_plot_save_path(results_dir, single_config_name,
                                    f'train_acc{suffix}.png',
                                    is_joint_plot=(n_configs > 1))
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Plot val acc
    fig, axes = plt.subplots(1, n_configs, figsize=(n_configs*onefigsize[0], onefigsize[1]), sharey=True)
    if n_configs == 1:
        axes = [axes]

    for ax_idx, config in enumerate(configs):
        ax = axes[ax_idx]
        Ns = sorted(all_data[config].keys())

        for N_idx, N in enumerate(Ns):
            darkness = 0.3 + 0.6 * N_idx / max(1, len(Ns)-1)
            data = all_data[config][N]
            if 'val_acc' in data:
                ax.plot(data['val_acc'], color=rocket(darkness), linewidth=2, label=f'N={N}')

        ax.set_xlabel('Iteration')
        if ax_idx == 0:
            ax.set_ylabel('Validation Accuracy')
        ax.set_title(get_figure_title(config, results_dir))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Set adaptive ylim but clamp to [0, 1] for accuracy
    set_shared_ylim(axes)
    for ax in axes:
        ylim = ax.get_ylim()
        ax.set_ylim(max(0.0, ylim[0]), min(1.0, ylim[1]))
    plt.tight_layout()
    single_config_name = configs[0] if n_configs == 1 else None
    save_path = get_plot_save_path(results_dir, single_config_name,
                                    f'val_acc{suffix}.png',
                                    is_joint_plot=(n_configs > 1))
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_entropy(all_data, results_dir, group_name=''):
    """Plot routing entropy"""
    configs = sorted(all_data.keys())
    n_configs = len(configs)
    
    fig, axes = plt.subplots(1, n_configs, figsize=(n_configs*onefigsize[0], onefigsize[1]), sharey=True)
    if n_configs == 1:
        axes = [axes]
    
    for ax_idx, config in enumerate(configs):
        ax = axes[ax_idx]
        Ns = sorted(all_data[config].keys())
        
        for N_idx, N in enumerate(Ns):
            darkness = 0.3 + 0.6 * N_idx / max(1, len(Ns)-1)
            data = all_data[config][N]
            if 'entropy' in data:
                ax.plot(data['entropy'], color=rocket(darkness), linewidth=2, label=f'N={N}')
        
        ax.set_xlabel('Iteration')
        if ax_idx == 0:
            ax.set_ylabel('Routing Entropy')
        ax.set_title(get_figure_title(config, results_dir))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    set_shared_ylim(axes)
    plt.tight_layout()
    suffix = f'_{group_name}' if group_name else ''
    single_config_name = configs[0] if n_configs == 1 else None
    save_path = get_plot_save_path(results_dir, single_config_name,
                                    f'entropy{suffix}.png',
                                    is_joint_plot=(n_configs > 1))
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_expert_l2_diffs(all_data, results_dir, group_name=''):
    """Plot L2 differences between experts"""
    configs = sorted(all_data.keys())
    n_configs = len(configs)
    
    fig, axes = plt.subplots(1, n_configs, figsize=(n_configs*onefigsize[0], onefigsize[1]), sharey=True)
    if n_configs == 1:
        axes = [axes]
    
    for ax_idx, config in enumerate(configs):
        ax = axes[ax_idx]
        Ns = sorted(all_data[config].keys())
        
        for N_idx, N in enumerate(Ns):
            darkness = 0.3 + 0.6 * N_idx / max(1, len(Ns)-1)
            data = all_data[config][N]
            if 'expert_l2_diffs' in data and len(data['expert_l2_diffs']) > 0:
                diffs = np.array(data['expert_l2_diffs'])
                if len(diffs.shape) > 1:
                    avg_diffs = np.mean(diffs, axis=1)
                else:
                    avg_diffs = diffs
                ax.plot(avg_diffs, color=rocket(darkness), linewidth=2, label=f'N={N}')
        
        ax.set_xlabel('Iteration')
        if ax_idx == 0:
            ax.set_ylabel('Avg RMS Diff Betw. Experts')
        ax.set_title(get_figure_title(config, results_dir))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    set_shared_ylim(axes)
    plt.tight_layout()
    suffix = f'_{group_name}' if group_name else ''
    single_config_name = configs[0] if n_configs == 1 else None
    save_path = get_plot_save_path(results_dir, single_config_name,
                                    f'expert_l2_diffs{suffix}.png',
                                    is_joint_plot=(n_configs > 1))
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_h_L_rms_diff(all_data, results_dir, group_name=''):
    """Plot RMS(h^L_t - h^L_0)"""
    configs = sorted(all_data.keys())
    n_configs = len(configs)
    
    fig, axes = plt.subplots(1, n_configs, figsize=(n_configs*onefigsize[0], onefigsize[1]), sharey=True)
    if n_configs == 1:
        axes = [axes]
    
    for ax_idx, config in enumerate(configs):
        ax = axes[ax_idx]
        Ns = sorted(all_data[config].keys())
        
        for N_idx, N in enumerate(Ns):
            darkness = 0.3 + 0.6 * N_idx / max(1, len(Ns)-1)
            data = all_data[config][N]
            if 'h_L_rms_diff' in data:
                ax.plot(data['h_L_rms_diff'], color=rocket(darkness), linewidth=2, label=f'N={N}')
        
        ax.set_xlabel('Iteration')
        if ax_idx == 0:
            ax.set_ylabel('Feature Learning RMS(h^L_t - h^L_0)')
        ax.set_title(get_figure_title(config, results_dir))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    set_shared_ylim(axes)
    plt.tight_layout()
    suffix = f'_{group_name}' if group_name else ''
    single_config_name = configs[0] if n_configs == 1 else None
    save_path = get_plot_save_path(results_dir, single_config_name,
                                    f'h_L_rms_diff{suffix}.png',
                                    is_joint_plot=(n_configs > 1))
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_expert_hists(all_data, results_dir, group_name=''):
    """Plot expert activation histograms for largest N at final iteration.
    Shows 2 most routed (top-1, top-2) and 2 least routed (bottom-2, bottom-1) experts."""
    configs = sorted(all_data.keys())
    n_configs = len(configs)

    fig, axes = plt.subplots(1, n_configs, figsize=(n_configs*onefigsize[0], onefigsize[1]), sharey=True)
    if n_configs == 1:
        axes = [axes]

    for ax_idx, config in enumerate(configs):
        ax = axes[ax_idx]
        Ns = sorted(all_data[config].keys())
        N_max = Ns[-1]
        data = all_data[config][N_max]

        if 'expert_hists' in data and len(data['expert_hists']) > 0 and 'router_weights' in data:
            final_hists = data['expert_hists'][-1]
            M = len(final_hists)

            # Compute mean routing weight for each expert at final iteration
            router_weights_final = data['router_weights'][-1]  # [batch, M]
            mean_routing_weights = router_weights_final.mean(axis=0)  # [M]

            # Sort experts by mean routing weight
            expert_rankings = [(m, mean_routing_weights[m]) for m in range(M)]
            expert_rankings.sort(key=lambda x: x[1], reverse=True)

            # Select top-2 and bottom-2
            if M >= 4:
                selected_experts = [expert_rankings[0], expert_rankings[1],    # top-1, top-2
                                   expert_rankings[-2], expert_rankings[-1]]   # bottom-2, bottom-1
                labels = ['top-1', 'top-2', 'bottom-2', 'bottom-1']
            else:
                # If fewer than 4 experts, plot all
                selected_experts = expert_rankings
                labels = [f'Expert {m}' for m, _ in selected_experts]

            colors = ['Reds', 'Blues', 'Greens', 'Purples']

            for idx, ((m, _), label) in enumerate(zip(selected_experts, labels)):
                counts, edges = final_hists[m]
                centers = 0.5 * (edges[:-1] + edges[1:])
                widths = edges[1:] - edges[:-1]
                density = counts / (counts.sum() * widths + 1e-10)
                cmap = plt.colormaps.get_cmap(colors[idx % len(colors)])
                ax.plot(centers, density, color=cmap(0.7), label=label, linewidth=2)

        ax.set_xlabel('Activation')
        if ax_idx == 0:
            ax.set_ylabel('Density')
        ax.set_title(get_figure_title(config, results_dir))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    set_shared_ylim(axes)
    plt.tight_layout()
    suffix = f'_{group_name}' if group_name else ''
    single_config_name = configs[0] if n_configs == 1 else None
    save_path = get_plot_save_path(results_dir, single_config_name,
                                    f'expert_hists{suffix}.png',
                                    is_joint_plot=(n_configs > 1))
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_h_L_hists(all_data, results_dir, group_name=''):
    """Plot h^L (combined) activation histograms for largest N at final iteration"""
    configs = sorted(all_data.keys())
    n_configs = len(configs)
    
    fig, axes = plt.subplots(1, n_configs, figsize=(n_configs*onefigsize[0], onefigsize[1]), sharey=True)
    if n_configs == 1:
        axes = [axes]
    
    for ax_idx, config in enumerate(configs):
        ax = axes[ax_idx]
        Ns = sorted(all_data[config].keys())
        N_max = Ns[-1]
        data = all_data[config][N_max]
        
        if 'h_L_hists' in data and len(data['h_L_hists']) > 0:
            final_hist = data['h_L_hists'][-1]
            counts, edges = final_hist
            centers = 0.5 * (edges[:-1] + edges[1:])
            widths = edges[1:] - edges[:-1]
            density = counts / (counts.sum() * widths + 1e-10)
            ax.plot(centers, density, color='darkblue', linewidth=2)
        
        ax.set_xlabel('h^L Activation')
        if ax_idx == 0:
            ax.set_ylabel('Density')
        ax.set_title(get_figure_title(config, results_dir))
        ax.grid(True, alpha=0.3)
    
    set_shared_ylim(axes)
    plt.tight_layout()
    suffix = f'_{group_name}' if group_name else ''
    single_config_name = configs[0] if n_configs == 1 else None
    save_path = get_plot_save_path(results_dir, single_config_name,
                                    f'h_L_hists{suffix}.png',
                                    is_joint_plot=(n_configs > 1))
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_output_gradient(all_data, results_dir, group_name=''):
    """Plot RMS(dL/d(output))"""
    configs = sorted(all_data.keys())
    n_configs = len(configs)
    
    fig, axes = plt.subplots(1, n_configs, figsize=(n_configs*onefigsize[0], onefigsize[1]), sharey=True)
    if n_configs == 1:
        axes = [axes]
    
    for ax_idx, config in enumerate(configs):
        ax = axes[ax_idx]
        Ns = sorted(all_data[config].keys())
        
        for N_idx, N in enumerate(Ns):
            darkness = 0.3 + 0.6 * N_idx / max(1, len(Ns)-1)
            data = all_data[config][N]
            if 'output_grad_rms' in data:
                ax.plot(data['output_grad_rms'], color=rocket(darkness), linewidth=2, label=f'N={N}')
        
        ax.set_xlabel('Iteration')
        if ax_idx == 0:
            ax.set_ylabel('RMS(dL/d(output))')
        ax.set_title(get_figure_title(config, results_dir))
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    set_shared_ylim(axes)
    plt.tight_layout()
    suffix = f'_{group_name}' if group_name else ''
    single_config_name = configs[0] if n_configs == 1 else None
    save_path = get_plot_save_path(results_dir, single_config_name,
                                    f'output_grad{suffix}.png',
                                    is_joint_plot=(n_configs > 1))
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_hagg_decomposition(all_data, results_dir, group_name=''):
    """Plot h^agg decomposition: 4 terms (base, propagating, effective, cross) over time"""
    configs = sorted(all_data.keys())
    n_configs = len(configs)

    # Check if any data has hagg_decomposition
    has_data = False
    for config in configs:
        for N in all_data[config]:
            if 'hagg_decomposition' in all_data[config][N] and len(all_data[config][N]['hagg_decomposition']) > 0:
                has_data = True
                break
        if has_data:
            break

    if not has_data:
        print("  Skipping h^agg decomposition plot (no data available)")
        return

    # Create 4 subplots for the 4 terms
    fig, axes = plt.subplots(2, 2, figsize=(2*onefigsize[0]*n_configs, 2*onefigsize[1]), sharex=True)
    axes = axes.flatten()

    terms = ['base', 'propagating', 'effective', 'cross']
    term_labels = [r'Init: $\sum_i \phi_i(t) W^{3,i}_0 h^{2,i}_0$',
                   r'Propagating: $\sum_i \phi_i(t) W^{3,i}_0 \Delta h^{2,i}$',
                   r'Eff.-init: $\sum_i \phi_i(t) \Delta W^{3,i} h^{2,i}_0$',
                   r'Eff.-prop.: $\sum_i \phi_i(t) \Delta W^{3,i} \Delta h^{2,i}$']

    for term_idx, (term, term_label) in enumerate(zip(terms, term_labels)):
        ax = axes[term_idx]

        for config in configs:
            Ns = sorted(all_data[config].keys())

            for N_idx, N in enumerate(Ns):
                darkness = 0.3 + 0.6 * N_idx / max(1, len(Ns)-1)
                data = all_data[config][N]

                if 'hagg_decomposition' in data and len(data['hagg_decomposition']) > 0:
                    # Extract term values over time
                    term_values = []
                    for decomp_dict in data['hagg_decomposition']:
                        if isinstance(decomp_dict, dict) and term in decomp_dict:
                            term_values.append(decomp_dict[term])
                        else:
                            term_values.append(np.nan)

                    if len(term_values) > 0 and not np.all(np.isnan(term_values)):
                        label = f'{format_label(config)}, N={N}' if n_configs > 1 else f'N={N}'
                        ax.plot(term_values, color=rocket(darkness), linewidth=2, label=label, alpha=0.8)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('RMS')
        ax.set_title(term_label, fontsize=10)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    plt.tight_layout()
    suffix = f'_{group_name}' if group_name else ''
    single_config_name = configs[0] if n_configs == 1 else None
    save_path = get_plot_save_path(results_dir, single_config_name,
                                    f'hagg_decomposition{suffix}.png',
                                    is_joint_plot=(n_configs > 1))
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved h^agg decomposition plot")

def plot_expert_grad_h1_decomposition(all_data, results_dir, group_name=''):
    """Plot expert gradient to h^1 decomposition: 4 terms (base, propagating, effective, cross) over time"""
    configs = sorted(all_data.keys())
    n_configs = len(configs)

    # Check if any data has expert_grad_h1_decomposition
    has_data = False
    for config in configs:
        for N in all_data[config]:
            if 'expert_grad_h1_decomposition' in all_data[config][N] and len(all_data[config][N]['expert_grad_h1_decomposition']) > 0:
                has_data = True
                break
        if has_data:
            break

    if not has_data:
        print("  Skipping expert grad→h^1 decomposition plot (no data available)")
        return

    # Create 4 subplots for the 4 terms
    fig, axes = plt.subplots(2, 2, figsize=(2*onefigsize[0]*n_configs, 2*onefigsize[1]), sharex=True)
    axes = axes.flatten()

    terms = ['base', 'propagating', 'effective', 'cross']
    term_labels = [r'Init: $\sum_i g_i(0) (W^{2,i}_0)^T$',
                   r'Propagating: $\sum_i \Delta g_i (W^{2,i}_0)^T$',
                   r'Eff.-init: $\sum_i g_i(0) (\Delta W^{2,i})^T$',
                   r'Eff.-prop.: $\sum_i \Delta g_i (\Delta W^{2,i})^T$']

    for term_idx, (term, term_label) in enumerate(zip(terms, term_labels)):
        ax = axes[term_idx]

        for config in configs:
            Ns = sorted(all_data[config].keys())

            for N_idx, N in enumerate(Ns):
                darkness = 0.3 + 0.6 * N_idx / max(1, len(Ns)-1)
                data = all_data[config][N]

                if 'expert_grad_h1_decomposition' in data and len(data['expert_grad_h1_decomposition']) > 0:
                    # Extract term values over time
                    term_values = []
                    for decomp_dict in data['expert_grad_h1_decomposition']:
                        if isinstance(decomp_dict, dict) and term in decomp_dict:
                            term_values.append(decomp_dict[term])
                        else:
                            term_values.append(np.nan)

                    if len(term_values) > 0 and not np.all(np.isnan(term_values)):
                        label = f'{format_label(config)}, N={N}' if n_configs > 1 else f'N={N}'
                        ax.plot(term_values, color=rocket(darkness), linewidth=2, label=label, alpha=0.8)

        ax.set_xlabel('Iteration')
        ax.set_ylabel('RMS')
        ax.set_title(term_label, fontsize=10)
        ax.legend(fontsize=6)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

    plt.tight_layout()
    suffix = f'_{group_name}' if group_name else ''
    single_config_name = configs[0] if n_configs == 1 else None
    save_path = get_plot_save_path(results_dir, single_config_name,
                                    f'expert_grad_h1_decomposition{suffix}.png',
                                    is_joint_plot=(n_configs > 1))
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved expert grad→h^1 decomposition plot")

if __name__ == "__main__":
    main()
