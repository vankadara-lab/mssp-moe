#!/usr/bin/env python3
"""Plot exponent scatter plots grouped by scaling limit."""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Apply plotting style
try:
    from tueplots import bundles
    plt.rcParams.update(bundles.icml2022())
    plt.rcParams.update({"figure.dpi": 300, "text.usetex": False})
    onefigsize = bundles.icml2022()['figure.figsize']
except ImportError:
    onefigsize = (3.25, 2.0)

from label_utils import format_subplot_title

# Layer name mapping
LAYER_NAMES = {
    'W_in': 'Input',
    'Q': 'Router',
    'W_exp1': 'Expert Input',
    'W_exp2': 'Expert Output',
    'W_out': 'Output'
}

# Config grouping by scaling limit
SCALING_LIMITS = {
    'N, N_e → ∞, E fixed': [
        'fixed_E_ntp', 'fixed_E_ntp_multfree', 'fixed_E_sp',
        'fixed_E_mup', 'fixed_E_mup_multfree', 'fixed_E_mup_old_multfree', 'fixed_E_mup_heuristic',
        'fixed_E_mup_largerouterlr', 'fixed_E_mup_largerouterlr_multfree',
        'fixed_E_mup_largerouterlr_nonasympQ',
        'fixed_E_mup_adam', 'fixed_E_mup_adam_smallrouter', 'fixed_E_mup_adam_globaleps'
    ],
    'N → ∞, E ~ N/N_e, N_e fixed': [
        'ntp', 'ntp_multfree',
        'mup_smallrouter', 'mup_smallrouter_multfree',
        'mup_largerouter', 'mup_largerouter_multfree',
        'mup_smallrouter_singlelr'
    ],
    'N, E, N_e → ∞ (all scale)': [
        'ntp_allscaling', 'ntp_allscaling_multfree',
        'mup_allscaling', 'mup_clt_smallinlr_allscaling_multfree', 'mup_fullclt_allscaling_multfree',
        'mup_allscaling_multfree', 'mup_stdinit_allscaling_multfree', 'mup_heuristic_allscaling_multfree',
        'mup_allscaling_nonasympQ', 'mup_allscaling_nonasympQ_multfree',
        'mup_smallinputlr_allscaling_multfree',
        'sp_allscaling_multfree', 'sp_adam_allscaling_multfree',
        'mup_adam_jiang_allscaling_multfree', 'mup_adam_smallrouter_allscaling_multfree',
        'mup_adam_globaleps_allscaling_multfree', 'mup_adam_allscaling_ours', 'mup_adam_allscaling_stdinit_ours'
    ],
    'N, E → ∞, N_e fixed (bottleneck)': [
        'mup_bottleneck_multfree', 'mup_bottleneck_ours', 'mup_bottleneck_stdinit',
        'mup_bottleneck_largeexpertin_allclt', 'mup_bottleneck_largeexpertin_sepagg',
        'mup_bottleneck_jiang_sgd', 'mup_bottleneck_heuristic_multfree',
        'mup_bottleneck_adam_multfree', 'mup_bottleneck_adam_stdinit_multfree',
        'mup_bottleneck_adam_globaleps_multfree', 'mup_adam_jiang_bottleneck_multfree'
    ]
}

# Config display names and colors (synchronized with canonical LABEL_MAP)
CONFIG_NAMES = {
    'fixed_E_ntp': 'NTP', 'fixed_E_ntp_multfree': 'NTP', 'fixed_E_sp': 'SP',
    'fixed_E_mup': 'μP', 'fixed_E_mup_multfree': 'μP (ours)', 'fixed_E_mup_old_multfree': 'μP old',
    'fixed_E_mup_heuristic': 'μP-heuristic',
    'fixed_E_mup_largerouterlr': 'μP large router LR', 'fixed_E_mup_largerouterlr_multfree': 'μP large router LR',
    'fixed_E_mup_largerouterlr_nonasympQ': 'μP large router LR nonasymp',
    'fixed_E_mup_adam': 'μP Adam (ours)', 'fixed_E_mup_adam_smallrouter': 'Jiang et al. (2026)',
    'fixed_E_mup_adam_globaleps': 'μP Adam global eps',
    'ntp': 'NTP', 'ntp_multfree': 'NTP',
    'mup_smallrouter': 'μP small router', 'mup_smallrouter_multfree': 'μP small router',
    'mup_largerouter': 'μP large router', 'mup_largerouter_multfree': 'μP large router',
    'mup_smallrouter_singlelr': 'μP small router single LR',
    'ntp_allscaling': 'NTP', 'ntp_allscaling_multfree': 'NTP',
    'mup_allscaling': 'μP', 'mup_clt_smallinlr_allscaling_multfree': 'μP',
    'mup_fullclt_allscaling_multfree': 'μP full CLT',
    'mup_allscaling_multfree': 'μP All-scaling (ours)',
    'mup_stdinit_allscaling_multfree': 'μP with std expert init',
    'mup_heuristic_allscaling_multfree': 'μP-heuristic',
    'mup_allscaling_nonasympQ': 'μP nonasymp', 'mup_allscaling_nonasympQ_multfree': 'μP nonasymp',
    'mup_smallinputlr_allscaling_multfree': r'μP, $\eta_{W_{\mathrm{in}}}\propto N^{-1}$',
    'sp_allscaling_multfree': 'SP', 'sp_adam_allscaling_multfree': 'SP Adam',
    'mup_adam_jiang_allscaling_multfree': 'Jiang et al. (2026)', 'mup_adam_smallrouter_allscaling_multfree': 'Jiang et al. small router',
    'mup_adam_globaleps_allscaling_multfree': 'μP Adam Global ε',
    'mup_adam_allscaling_ours': 'μP Adam (ours)',
    'mup_adam_allscaling_stdinit_ours': 'μP Adam with std expert init',
    'mup_bottleneck_multfree': 'old mup guess',
    'mup_bottleneck_ours': 'μP Bottleneck (ours)',
    'mup_bottleneck_stdinit': 'μP with std expert init',
    'mup_bottleneck_largeexpertin_allclt': 'μP Large Expert In All-CLT',
    'mup_bottleneck_largeexpertin_sepagg': 'μP Large Expert In Sep-Agg',
    'mup_bottleneck_jiang_sgd': 'Jiang et al. SGD Bottleneck',
    'mup_bottleneck_heuristic_multfree': 'μP-heuristic',
    'mup_bottleneck_adam_multfree': 'μP Adam (ours)',
    'mup_bottleneck_adam_stdinit_multfree': 'μP Adam with std expert init',
    'mup_bottleneck_adam_globaleps_multfree': 'μP Adam Global ε',
    'mup_adam_jiang_bottleneck_multfree': 'Jiang et al. (2026)'
}

CONFIG_COLORS = {
    'fixed_E_ntp': 'gray', 'fixed_E_ntp_multfree': 'gray', 'fixed_E_sp': 'gray',
    'fixed_E_mup': 'green', 'fixed_E_mup_multfree': 'green', 'fixed_E_mup_old_multfree': 'darkgreen',
    'fixed_E_mup_heuristic': 'lightgreen',
    'fixed_E_mup_largerouterlr': 'blue', 'fixed_E_mup_largerouterlr_multfree': 'blue',
    'fixed_E_mup_largerouterlr_nonasympQ': 'purple',
    'fixed_E_mup_adam': 'green', 'fixed_E_mup_adam_smallrouter': 'orange', 'fixed_E_mup_adam_globaleps': 'blue',
    'ntp': 'gray', 'ntp_multfree': 'gray',
    'mup_smallrouter': 'orange', 'mup_smallrouter_multfree': 'orange',
    'mup_largerouter': 'green', 'mup_largerouter_multfree': 'green',
    'mup_smallrouter_singlelr': 'brown',
    'ntp_allscaling': 'gray', 'ntp_allscaling_multfree': 'gray',
    'mup_allscaling': 'green', 'mup_clt_smallinlr_allscaling_multfree': 'green',
    'mup_fullclt_allscaling_multfree': 'darkgreen', 'mup_lln_allscaling_multfree': 'orange',
    'mup_largeexpert_allscaling_multfree': 'purple',
    'mup_heuristic_allscaling_multfree': 'lightgreen',
    'mup_allscaling_nonasympQ': 'blue', 'mup_allscaling_nonasympQ_multfree': 'blue',
    'mup_smallinputlr_allscaling_multfree': 'red',
    'sp_allscaling_multfree': 'gray', 'sp_adam_allscaling_multfree': 'gray',
    'mup_adam_jiang_allscaling_multfree': 'orange', 'mup_adam_smallrouter_allscaling_multfree': 'darkorange',
    'mup_adam_globaleps_allscaling_multfree': 'blue',
    'mup_adam_allscaling_ours': 'green',
    'mup_bottleneck_multfree': 'green', 'mup_bottleneck_largeexpertin': 'blue',
    'mup_bottleneck_largeexpert': 'navy',
    'mup_bottleneck_largeexpertin_allclt': 'teal',
    'mup_bottleneck_largeexpertin_sepagg': 'purple',
    'mup_bottleneck_jiang_sgd': 'orange',
    'mup_bottleneck_heuristic_multfree': 'blue',
    'mup_bottleneck_adam_multfree': 'purple',
    'mup_bottleneck_adam_globaleps_multfree': 'cyan',
    'mup_adam_jiang_bottleneck_multfree': 'orange'
}


def plot_exponent_bars(csv_path, output_dir, data_type='normalized', metadata=None, update_type='effective_updates', timesteps=None):
    """Create scatter plot of exponents with subplots for each scaling limit.

    Args:
        timesteps: List of integers (e.g., [2, 3, 50, -1] where -1 means T-1)
    """
    if timesteps is None:
        timesteps = [2, 50, 200]
    else:
        timesteps = list(timesteps)

    # Always append T-1 if not present
    if metadata and 'T' in metadata:
        T_minus_1 = metadata['T'] - 1
        if T_minus_1 not in timesteps:
            timesteps.append(T_minus_1)

    # Load CSVs
    dfs = {}
    for ts in timesteps:
        ts_csv = csv_path.parent / f'exponent_table_{update_type}_t{ts}.csv'
        if ts_csv.exists():
            dfs[ts] = pd.read_csv(ts_csv, index_col=0)

    if not dfs:
        return

    common_configs = set(dfs[timesteps[0]].index)
    for df in dfs.values():
        common_configs &= set(df.index)

    # Extract layer names from first df
    layer_order = ['W_in', 'Q', 'W_exp1', 'W_exp2', 'W_out']
    first_df = list(dfs.values())[0]
    layer_cols = [f"{l}_{data_type}" for l in layer_order if f"{l}_{data_type}" in first_df.columns]

    # Count how many configs have data (one subplot per config)
    # Use predefined order from SCALING_LIMITS to ensure consistent ordering
    config_order = []
    for limit_name, config_list in SCALING_LIMITS.items():
        for c in config_list:
            if c in common_configs and c not in config_order:
                config_order.append(c)
    # Add any remaining configs not in SCALING_LIMITS
    for c in sorted(common_configs):
        if c not in config_order:
            config_order.append(c)

    configs_with_data = config_order

    if not configs_with_data:
        return

    # Create subplots: one per config
    n_configs = len(configs_with_data)
    fig, axes = plt.subplots(1, n_configs, figsize=(n_configs*onefigsize[0]*1.2, onefigsize[1]*1.5), sharey=True)
    if n_configs == 1:
        axes = [axes]

    # Define elegant color scheme for timesteps
    # Use a perceptually uniform colormap that works well for showing progression
    n_timesteps = len(timesteps)
    cmap = plt.cm.viridis
    timestep_colors = [cmap(i / max(1, n_timesteps - 1)) for i in range(n_timesteps)]

    for ax_idx, config in enumerate(configs_with_data):
        ax = axes[ax_idx]

        # Prepare data for plotting
        n_layers = len(layer_cols)
        x = np.arange(n_layers)

        # Plot scatter for each timestep
        for t_idx, ts in enumerate(timesteps):
            if ts not in dfs:
                continue
            df_ts = dfs[ts]

            values = []
            errors = []
            for col in layer_cols:
                if col not in df_ts.columns:
                    values.append(np.nan)
                    errors.append(0.0)
                    continue
                val_str = df_ts.loc[config, col]
                try:
                    if '±' in str(val_str):
                        mean_str, std_str = str(val_str).split('±')
                        values.append(float(mean_str))
                        errors.append(float(std_str))
                    else:
                        values.append(float(val_str))
                        errors.append(0.0)
                except:
                    values.append(np.nan)
                    errors.append(0.0)

            # Add x-jitter to separate different timesteps visually
            # Jitter spreads timesteps evenly around each layer position
            jitter_width = 0.15  # Total jitter range per layer
            x_offset = (t_idx - (n_timesteps - 1) / 2) * (jitter_width / max(1, n_timesteps - 1))
            x_jittered = x + x_offset

            ax.errorbar(x_jittered, values, yerr=errors, fmt='o', markersize=6, capsize=3,
                       alpha=0.9, linewidth=1.5, elinewidth=1.5, color=timestep_colors[t_idx])
        
        # Formatting
        ax.set_xlabel('Layer')
        if ax_idx == 0:
            if update_type == 'effective_updates' or update_type == 'total_effective_updates_per_layer':
                if data_type == 'normalized':
                    ax.set_ylabel(r'$||\Delta W_t (x_t/||x_t||_{\mathrm{RMS}})||_{\mathrm{RMS}}$ Exponent')
                else:
                    ax.set_ylabel(r'$||\Delta W_t x_t||_{\mathrm{RMS}}$ Exponent')
            elif update_type == 'propagating_updates':
                if data_type == 'normalized':
                    ax.set_ylabel(r'$||W_0 \Delta x_t / ||\Delta x_t||_{\mathrm{RMS}}||_{\mathrm{RMS}}$ Exponent')
                else:
                    ax.set_ylabel(r'$||W_0 \Delta x_t||_{\mathrm{RMS}}$ Exponent')
            elif update_type == 'grad_norms':
                ax.set_ylabel(r'$||\nabla_{W_t} L||_{\mathrm{RMS}}$ Exponent')
            elif update_type == 'weight_rms_norms':
                ax.set_ylabel(r'$||W_t||_{\mathrm{RMS}}$ Exponent')
            else:
                ax.set_ylabel('Exponent')
        
        ax.set_title(format_subplot_title(config))
        ax.set_xticks(x)
        ax.set_xticklabels([LAYER_NAMES.get(l.replace(f'_{data_type}', ''), l) for l in layer_cols], rotation=-30, ha='left')
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        if update_type == 'grad_norms':
            ax.set_ylim(-3, 0.5)
        else:
            ax.set_ylim(-2, 1)
        from matplotlib.ticker import MultipleLocator
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.grid(True, alpha=0.3, axis='y')
    
    # Single timestep legend centered below all subplots
    from matplotlib.lines import Line2D
    timestep_handles = [Line2D([0], [0], marker='o', color=timestep_colors[i], markersize=6,
                               alpha=0.9, linestyle='', label=f't={ts}')
                       for i, ts in enumerate(timesteps)]
    fig.legend(handles=timestep_handles, fontsize=7, loc='lower center',
              bbox_to_anchor=(0.5, -0.15), ncol=len(timesteps), title='Timestep')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    # Save combined plot
    ts_slug = '_'.join(map(str, timesteps))
    output_path = Path(output_dir) / f'rcc_exponents_{update_type}_t{ts_slug}_{data_type}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")


DECOMP_TERM_LABELS = {
    'hagg_decomp': {
        'base':        r'Init: $\sum_i \phi_i W^{3,i}_0 h^{2,i}_0$',
        'propagating': r'Prop.: $\sum_i \phi_i W^{3,i}_0 \Delta h^{2,i}$',
        'effective':   r'Eff.-init: $\sum_i \phi_i \Delta W^{3,i} h^{2,i}_0$',
        'cross':       r'Eff.-prop.: $\sum_i \phi_i \Delta W^{3,i} \Delta h^{2,i}$',
    },
    'grad_h1_decomp': {
        'base':        r'Init: $\sum_i (W^{2,i}_0)^\top g^i_0$',
        'propagating': r'Prop.: $\sum_i (W^{2,i}_0)^\top \Delta g^i$',
        'effective':   r'Eff.-init: $\sum_i (\Delta W^{2,i})^\top g^i_0$',
        'cross':       r'Eff.-prop.: $\sum_i (\Delta W^{2,i})^\top \Delta g^i$',
    },
}


def plot_decomp_exponent_bars(results_dir, decomp_type='hagg_decomp'):
    """Plot exponent scatter for decomposition terms: one subplot per scaling config.

    Each subplot has the 4 decomp terms on the x-axis (base, prop, eff(ΔΔ), cross(Δ0))
    and the scaling exponent on the y-axis, with different timesteps as viridis markers.
    Display naming: data key 'cross' (ΔΔ) → "Eff.-prop.", data key 'effective' (Δ0) → "Eff.-init".
    Saved to rcc/{decomp_type}_exponents_t{ts_slug}.png.
    """
    import glob as _glob

    results_dir = Path(results_dir)

    # Display order: base, prop, Eff.-prop.(ΔΔ)=data:'cross', Eff.-init(Δ0)=data:'effective'
    display_terms = [
        ('base',        r'Base'),
        ('propagating', r'Prop.'),
        ('cross',       r'Eff.-prop.'),
        ('effective',   r'Eff.-init'),
    ]
    term_keys = [t[0] for t in display_terms]
    term_xlabels = [t[1] for t in display_terms]

    # Auto-detect timesteps from CSV files
    csv_files = _glob.glob(str(results_dir / f'exponent_table_{decomp_type}_t*.csv'))
    if not csv_files:
        print(f"No CSV files found for {decomp_type}, skipping decomp exponent plot")
        return
    timesteps = []
    for f in csv_files:
        ts_str = (Path(f).name
                  .replace(f'exponent_table_{decomp_type}_', '')
                  .replace('.csv', '')
                  .lstrip('t'))
        try:
            timesteps.append(int(ts_str))
        except ValueError:
            pass
    timesteps = sorted(timesteps)

    if not timesteps:
        return

    # Load CSVs
    dfs = {}
    for ts in timesteps:
        csv_path = results_dir / f'exponent_table_{decomp_type}_t{ts}.csv'
        if csv_path.exists():
            dfs[ts] = pd.read_csv(csv_path, index_col=0)

    if not dfs:
        return

    # Get ordered config list
    common_configs = set(list(dfs.values())[0].index)
    for df in dfs.values():
        common_configs &= set(df.index)
    if not common_configs:
        return

    config_order = []
    for config_list in SCALING_LIMITS.values():
        for c in config_list:
            if c in common_configs and c not in config_order:
                config_order.append(c)
    for c in sorted(common_configs):
        if c not in config_order:
            config_order.append(c)
    configs = config_order
    n_configs = len(configs)

    n_timesteps = len(timesteps)
    cmap = plt.cm.viridis
    timestep_colors = [cmap(i / max(1, n_timesteps - 1)) for i in range(n_timesteps)]

    x = np.arange(len(term_keys))
    fig, axes = plt.subplots(1, n_configs,
                             figsize=(n_configs * onefigsize[0] * 1.2, onefigsize[1] * 1.5),
                             sharey=True)
    if n_configs == 1:
        axes = [axes]

    from matplotlib.ticker import MultipleLocator

    for cfg_idx, config in enumerate(configs):
        ax = axes[cfg_idx]

        for t_idx, ts in enumerate(timesteps):
            if ts not in dfs:
                continue
            df_ts = dfs[ts]

            values, errors = [], []
            for term_key in term_keys:
                if config not in df_ts.index or term_key not in df_ts.columns:
                    values.append(np.nan)
                    errors.append(0.0)
                    continue
                val_str = df_ts.loc[config, term_key]
                try:
                    if '±' in str(val_str):
                        mean_str, std_str = str(val_str).split('±')
                        values.append(float(mean_str))
                        errors.append(float(std_str))
                    else:
                        values.append(float(val_str))
                        errors.append(0.0)
                except Exception:
                    values.append(np.nan)
                    errors.append(0.0)

            jitter_width = 0.15
            x_offset = (t_idx - (n_timesteps - 1) / 2) * (jitter_width / max(1, n_timesteps - 1))
            ax.errorbar(x + x_offset, values, yerr=errors, fmt='o', markersize=6, capsize=3,
                        alpha=0.9, linewidth=1.5, elinewidth=1.5, color=timestep_colors[t_idx])

        if cfg_idx == 0:
            ax.set_ylabel('Exponent')
        ax.set_title(format_subplot_title(config))
        ax.set_xticks(x)
        ax.set_xticklabels(term_xlabels, rotation=-20, ha='left', fontsize=8)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_ylim(-2, 1)
        ax.yaxis.set_major_locator(MultipleLocator(0.5))
        ax.grid(True, alpha=0.3, axis='y')

    from matplotlib.lines import Line2D
    timestep_handles = [
        Line2D([0], [0], marker='o', color=timestep_colors[i], markersize=6,
               alpha=0.9, linestyle='', label=f't={ts}')
        for i, ts in enumerate(timesteps)
    ]
    fig.legend(handles=timestep_handles, fontsize=7, loc='lower center',
               bbox_to_anchor=(0.5, -0.15), ncol=len(timesteps), title='Timestep')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)

    rcc_dir = results_dir / 'rcc'
    rcc_dir.mkdir(exist_ok=True)
    ts_slug = '_'.join(map(str, timesteps))
    output_path = rcc_dir / f'{decomp_type}_exponents_t{ts_slug}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', required=True, help='Results directory')
    parser.add_argument('--update_type', default='effective_updates',
                       choices=['effective_updates', 'propagating_updates',
                                'total_effective_updates_per_layer', 'grad_norms', 'weight_rms_norms',
                                'hagg_decomp', 'grad_h1_decomp'])
    parser.add_argument('--timesteps', type=int, nargs='+', default=[2, 50, 200],
                       help='Timesteps to plot (e.g., 2 50 200). T-1 is always included.')
    parser.add_argument('--data_type', default='normalized', choices=['normalized', 'raw'])
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Route decomposition types to the new function
    if args.update_type in ('hagg_decomp', 'grad_h1_decomp'):
        plot_decomp_exponent_bars(results_dir, decomp_type=args.update_type)
        return

    # Skip normalized grad_norms plots
    if args.update_type == 'grad_norms' and args.data_type == 'normalized':
        print(f"Skipping normalized grad_norms plot")
        return

    # Try to load metadata from first config
    metadata = {}
    for config_dir in results_dir.iterdir():
        if config_dir.is_dir() and not config_dir.name.startswith('.'):
            stats_dir = config_dir / 'stats'
            if stats_dir.exists():
                import glob
                npz_files = glob.glob(str(stats_dir / 'nn_N*.npz'))
                if npz_files:
                    data = np.load(npz_files[0], allow_pickle=True)
                    if 'metadata' in data:
                        meta = data['metadata'].item()
                        if isinstance(meta, dict) and 'args' in meta:
                            metadata = meta['args']
                    break

    # Generate plots
    plot_exponent_bars(results_dir / 'dummy.csv', results_dir, args.data_type, metadata, args.update_type, args.timesteps)


if __name__ == '__main__':
    main()
