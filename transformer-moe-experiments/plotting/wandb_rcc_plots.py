#!/usr/bin/env python
# coding: utf-8

import argparse
import pickle
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from wandb_helper import get_runs_and_widths, get_runs_by_width_and_seed, get_metric_history
import matplotlib.pyplot as plt
import numpy as np

# Cache directory
CACHE_DIR = Path(__file__).resolve().parent.parent / 'results' / 'wandb_cache'
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# RCC type registry with key patterns and labels
RCC_TYPES = {
    "RCC (W_t-W_0)x_t": {"key": "{metric_path}/l2norm", "ylabel": r'$||\Delta W_t x_t||_{\mathrm{RMS}}$', "prefix": "effective"},
    "RCC W_0(x_t-x_0)": {"key": "{metric_path}/l2norm", "ylabel": r'$||W_0 \Delta x_t||_{\mathrm{RMS}}$', "prefix": "propagating"},
    "RCC (x_t-x_0)": {"key": "{metric_path}/l2norm", "ylabel": r'$||\Delta x^{in}_t||_{\mathrm{RMS}}$', "prefix": "in_activation_delta"},
    "activation_difference": {"key": "activation_difference/{metric_path}/L2norm", "ylabel": r"$||x_t-x_0||_{\mathrm{RMS}}$", "prefix": "activation_delta"},
    "activation": {"key": "activation/{metric_path}/L2norm", "ylabel": r"$||x_t||_{\mathrm{RMS}}$", "prefix": "activation"},
    "gradient": {"key": "gradient/{metric_path}/L2norm", "ylabel": r"$||\nabla_W||_{\mathrm{RMS}}$", "prefix": "gradient"},
    "parameter_difference": {"key": "parameter_difference/{metric_path}/L2norm", "ylabel": r"$||W_t-W_0||_{\mathrm{RMS}}$", "prefix": "param_diff"},
}

def get_input_dim(metric_path, width, config_type):
    """Get input dimension for a layer."""
    VOCAB_SIZE = 50304
    expert_hidden = 32 if config_type == 'bottleneck' else width // 2

    if 'q_norm' in metric_path or 'k_norm' in metric_path:
        return 64  # head dimension (constant across all widths)
    if 'wte' in metric_path:
        return VOCAB_SIZE
    if 'lm_head' in metric_path or 'router' in metric_path or 'w_g' in metric_path or 'w_noise' in metric_path:
        return width
    if 'expert' in metric_path:
        return width if 'c_fc' in metric_path else expert_hidden
    if 'mlp' in metric_path:
        return width if 'c_fc' in metric_path else 2 * width
    if 'ln_' in metric_path or 'norm' in metric_path or 'attn' in metric_path:
        return width
    return width


def get_output_dim(metric_path, width, config_type):
    """Get output dimension for a layer."""
    VOCAB_SIZE = 50304

    if config_type == 'bottleneck':
        n_exp = width // 8
    elif config_type == 'fixedE':
        n_exp = 8
    elif config_type in ['allscale']:
        n_exp = width // 32
    else:
        raise ValueError(f"Undefined scaling config_type {config_type}")
    
    expert_hidden = 32 if config_type == 'bottleneck' else width // 2

    if 'wte' in metric_path:
        return width
    if 'lm_head' in metric_path:
        return VOCAB_SIZE
    if 'router' in metric_path or 'w_g' in metric_path or 'w_noise' in metric_path:
        return n_exp
    if 'expert' in metric_path:
        return expert_hidden if 'c_fc' in metric_path else width
    if 'mlp' in metric_path:
        return 2 * width if 'c_fc' in metric_path else width
    if 'attn' in metric_path:
        if 'c_attn' in metric_path:
            return 3 * width
        if 'q_norm' in metric_path or 'k_norm' in metric_path:
            return 64  # head dimension (constant across all widths)
        return width
    if 'ln_' in metric_path or 'norm' in metric_path:
        return width
    return width



def get_layer_dim(metric_path, rcc_type, width, config_type='bottleneck'):
    """Return the dimension to normalize by for RMS calculation.

    Args:
        metric_path: Path to the metric (e.g., 'transformer.h.3.expert_module.router.w_g.weight')
        rcc_type: Type of RCC metric
        width: Model width (n_embd)
        config_type: 'bottleneck', 'allscale', or 'fixedE'

    Returns:
        Dimension to use in 1/sqrt(dim) normalization, or None if not applicable
    """
    # LayerNorm parameters: just width (one parameter per dimension)
    if ('ln_' in metric_path or 'norm' in metric_path) and rcc_type in ["gradient", "parameter_difference"]:
        return width

    # For gradient and parameter_difference: normalize by sqrt(num_entries) = sqrt(in_dim × out_dim)
    if rcc_type in ["gradient", "parameter_difference"]:
        in_dim = get_input_dim(metric_path, width, config_type)
        out_dim = get_output_dim(metric_path, width, config_type)
        return in_dim * out_dim

    # RCC (x_t-x_0): input activation dimension
    # Note: wte doesn't have input activation difference (discrete vocab indices)
    if rcc_type == "RCC (x_t-x_0)":
        if 'wte' in metric_path:
            return None
        return get_input_dim(metric_path, width, config_type)

    # Output activation metrics: normalize by output dimension
    if rcc_type in ["RCC (W_t-W_0)x_t", "RCC W_0(x_t-x_0)", "activation", "activation_difference"]:
        return get_output_dim(metric_path, width, config_type)

    # Fallback
    return width

def get_cached_metric(project, key, cache_suffix='', force_refresh=False):
    """Get metric with caching, supporting multiple seeds.

    Args:
        project: WandB project name
        key: Metric key
        cache_suffix: Optional suffix for cache filename
        force_refresh: If True, ignore cache and fetch from WandB

    Returns:
        steps: Array of timesteps
        run_results: Dict mapping (width, seed) -> array of metric values
        widths: Sorted list of unique widths
        seeds: Sorted list of unique seeds
    """
    cache_file = CACHE_DIR / f"{project.replace('/', '_')}_{key.replace('/', '_')}_{cache_suffix}.pkl"

    if cache_file.exists() and not force_refresh:
        print(f"Loading from cache: {cache_file.name}")
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
            # Handle old cache format (without seeds)
            if len(data) == 3:
                print("  Old cache format detected → forcing refresh")
                return get_cached_metric(project, key, cache_suffix, force_refresh=True)
                # steps, run_results, widths = data
                # run_results = {k: (steps, v) for k, v in run_results.items()}
                # seeds = [42]  # Assume single seed for old cache
                # print("  Warning: Old cache format detected (no seed info)")
            else:
                steps, run_results, widths, seeds = data
            return steps, run_results, widths, seeds

    if force_refresh and cache_file.exists():
        print(f"Force refresh: ignoring cache {cache_file.name}")

    print(f"Fetching from WandB: {key}")
    width_seed_runs, widths, seeds = get_runs_by_width_and_seed(project)

    run_results = {}
    steps = None

    for (width, seed), run in width_seed_runs.items():
        s, values = get_metric_history(run, key)
        if steps is None:
            steps = s
        run_results[(width, seed)] = (s, values) #run_results[(width, seed)] = values

    with open(cache_file, 'wb') as f:
        pickle.dump((steps, run_results, widths, seeds), f)

    return steps, run_results, widths, seeds

def compute_exponent(widths, values):
    """Compute scaling exponent using robust linear regression."""
    from scipy import stats
    if len(widths) < 2:
        return np.nan
    widths_arr = np.array(widths)
    values_arr = np.array(values)
    valid_mask = np.isfinite(values_arr) & np.isfinite(widths_arr) & (values_arr > 1e-20) & (widths_arr > 0)
    
    if np.sum(valid_mask) < 2:
        return np.nan
    
    widths_valid = widths_arr[valid_mask]
    values_valid = values_arr[valid_mask]
    log_widths = np.log10(widths_valid)
    log_values = np.log10(values_valid)
    
    try:
        slope, _, _, _, _ = stats.linregress(log_widths, log_values)
        return slope if np.isfinite(slope) else np.nan
    except:
        return np.nan

def infer_config_type(project):
    """Infer config type from project name."""
    if 'bottleneck' in project.lower():
        return 'bottleneck'
    elif 'allscale' in project.lower():
        return 'allscale'
    elif 'fixede' in project.lower() or 'fixed_e' in project.lower():
        return 'fixedE'
    return 'allscale'  # default


def plot_rcc_metric(project, metric_path, save_name=None, rcc_type="RCC (x_t-x_0)", force_refresh=False, widths_filter=None):
    """Generic function to plot any RCC metric as RMS norm with mean and std across seeds.

    Args:
        project: WandB project name
        metric_path: Path to the metric (e.g., 'transformer.h.3.ln_1')
        save_name: Custom save name (default: derived from metric_path and rcc_type)
        rcc_type: Type of RCC metric (default: 'RCC (x_t-x_0)')
        force_refresh: If True, ignore cache and fetch from WandB
        widths_filter: List of widths to include (e.g., [256, 512, 1024]). If None, all widths are used.
    """
    if rcc_type not in RCC_TYPES:
        raise ValueError(f"Unknown rcc_type: {rcc_type}. Available: {list(RCC_TYPES.keys())}")

    rcc_config = RCC_TYPES[rcc_type]
    key = f"{rcc_type}/{rcc_config['key'].format(metric_path=metric_path)}" if "{metric_path}" in rcc_config['key'] else rcc_config['key'].format(metric_path=metric_path)

    if save_name is None:
        base_name = metric_path.replace('.', '_').replace('/', '_')
        save_name = f"{rcc_config['prefix']}_{base_name}"

    steps, run_results, widths, seeds = get_cached_metric(project, key, save_name, force_refresh=force_refresh)

    # Filter widths if specified
    if widths_filter is not None:
        widths = [w for w in widths if w in widths_filter]
        if not widths:
            print(f"  Warning: No widths remaining after filtering for {metric_path}")
            return
        # Add widths to filename
        widths_str = "_".join(map(str, sorted(widths)))
        save_name = f"{save_name}_w{widths_str}"

    output_dir = Path(f"../../results/wandb/rcc/{project}")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))

    # Select specific timesteps: 2, 4, 10, 50, 100, and last available (should be ~500)
    desired_steps = [2, 4, 10, 50, 100, 300]
    if len(steps) > 0:
        desired_steps.append(int(steps[-1]))  # Add last available step

    # Filter to only steps that exist in the data (convert to list for comparison)
    # steps_list = steps.tolist() if hasattr(steps, 'tolist') else list(steps)
    # available_steps = [s for s in desired_steps if s in steps_list]
    all_steps = set()
    for (w, s), (run_steps, _) in run_results.items():
        all_steps.update(run_steps)

    available_steps = [s for s in desired_steps if s in all_steps]
    n_steps = len(available_steps)

    if n_steps == 0:
        plt.close()
        return

    blues = plt.cm.Blues(np.linspace(0.4, 0.95, n_steps))

    config_type = infer_config_type(project)

    # Check if this metric is applicable for this layer
    test_dim = get_layer_dim(metric_path, rcc_type, widths[0], config_type)
    if test_dim is None:
        plt.close()
        print(f"  Skipping {rcc_type} for {metric_path}: metric not applicable")
        return

    has_data = False
    exponents = []
    for i, gradient_step in enumerate(available_steps):
        # Compute RMS by dividing L2 norm by sqrt(dim) for each (width, seed)
        y_means = []
        y_stds = []
        valid_widths = []

        for width in widths:
            # Collect values across all seeds for this width
            seed_values = []

            for seed in seeds:
                if (width, seed) not in run_results:
                    continue

                # values = run_results[(width, seed)]

                # # Handle different indexing schemes
                # if len(values) == len(steps):
                #     # Indexed by position in steps array
                #     if gradient_step in steps_list:
                #         step_idx = steps_list.index(gradient_step)
                #         l2_norm = values[step_idx]
                #     else:
                #         continue
                # else:
                #     # Indexed by actual step value
                #     if gradient_step < len(values):
                #         l2_norm = values[gradient_step]
                #     else:
                #         continue

                run_steps, values = run_results[(width, seed)]
                run_steps_list = run_steps.tolist() if hasattr(run_steps, 'tolist') else list(run_steps)

                if gradient_step not in run_steps_list:
                    continue

                step_idx = run_steps_list.index(gradient_step)
                l2_norm = values[step_idx]

                dim = get_layer_dim(metric_path, rcc_type, width, config_type)
                if dim is None:
                    continue
                rms = l2_norm / np.sqrt(dim)
                seed_values.append(rms)

            if len(seed_values) > 0:
                y_means.append(np.mean(seed_values))
                y_stds.append(np.std(seed_values))
                valid_widths.append(width)

        if len(y_means) == 0:
            continue

        # Convert to arrays for easier processing
        y_means = np.array(y_means)
        y_stds = np.array(y_stds)
        valid_widths = np.array(valid_widths)

        # Check if data is valid for log-log plotting
        # Skip if all values are zero, negative, or NaN
        valid_data_mask = (y_means > 1e-30) & np.isfinite(y_means)
        if not valid_data_mask.any():
            # All values invalid, skip this timestep
            continue

        # Filter to only valid data points
        y_means = y_means[valid_data_mask]
        y_stds = y_stds[valid_data_mask]
        valid_widths = valid_widths[valid_data_mask]

        # Compute exponent from mean values
        exp = compute_exponent(valid_widths, y_means)
        exponents.append(exp)

        # compute per-width seed counts at this timestep
        seed_counts = []
        for width in valid_widths:
            count = 0
            for seed in seeds:
                if (width, seed) not in run_results:
                    continue
                run_steps, values = run_results[(width, seed)]
                run_steps_list = run_steps.tolist() if hasattr(run_steps, 'tolist') else list(run_steps)
                if gradient_step in run_steps_list:
                    count += 1
            if count > 0:
                seed_counts.append(count)

        n_eff = min(seed_counts) if seed_counts else 0

        # Plot mean line
        plt.loglog(valid_widths, y_means, 'o-', color=blues[i],
                   label=f"t={gradient_step} (α={exp:.2f}, n={n_eff})")#    label=f"t={gradient_step} (α={exp:.2f}, n={len(seeds)})")

        # Add shaded std band (ensure positive bounds for log scale)
        if len(seeds) > 1:
            lower_bound = np.maximum(y_means - 2 * y_stds, 1e-30)  # Ensure positive
            upper_bound = y_means + 2 * y_stds
            plt.fill_between(valid_widths, lower_bound, upper_bound,
                            color=blues[i], alpha=0.2)

        # Add exponent text annotation at x-center
        # Use geometric mean for x-position in log space
        x_center = np.exp(np.mean(np.log(valid_widths)))
        # Interpolate y-value at x_center
        y_center = np.exp(np.interp(np.log(x_center), np.log(valid_widths), np.log(y_means)))
        plt.text(x_center, y_center, f'α={exp:.2f}',
                color=blues[i], fontsize=12, fontweight='bold',
                ha='center', va='bottom')

        has_data = True

    if not has_data:
        plt.close()
        return

    plt.xlabel("Width N")
    plt.ylabel(rcc_config['ylabel'])

    # Add exponent info to title
    avg_exp = np.nanmean(exponents) if exponents else np.nan
    title = f"{metric_path}\nAvg α={avg_exp:.3f} (mean ± std over {len(seeds)} seeds)"
    plt.title(title, fontsize=10)

    plt.legend(fontsize=8, loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f"{save_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved {rcc_type} plot for {metric_path}: avg exponent = {avg_exp:.3f} ({len(seeds)} seeds)")

# Projects and metric paths to plot
projects = [
    "allscale_soft_coordinate_check_swc_baseline",
    # "allscale_soft_coordinate_check_swc",
    "fullmoe_bottleneck_coordinate_check",
    # "bottleneck_coordinate_check",
    # "allscale_coordinate_check",
    # "adamw_bottleneck_coordinate_check",
    # "adamw_allscale_coordinate_check",
    # "adamw_jiang_bottleneck_coordinate_check",
    # "adamw_jiang_allscale_coordinate_check",
    # "adamw_fixedE_coordinate_check",
    # "adamw_fixedE_heuristic_coordinate_check",
]

# Representative layer from each type
metric_paths = {
    "router": "transformer.h.3.expert_module.router.w_g.weight",
    "expert_in": "transformer.h.3.expert_module.experts.expert0.c_fc.weight",
    "expert_out": "transformer.h.3.expert_module.experts.expert0.c_proj.weight",
    "attn_in": "transformer.h.2.attn.c_attn.weight",
    "attn_out": "transformer.h.2.attn.c_proj.weight",
    "ln_final": "transformer.ln_f.weight",
    "q_norm": "transformer.h.2.attn.q_norm",
    "k_norm": "transformer.h.2.attn.k_norm",
    "mlp_in": "transformer.h.2.mlp.c_fc.weight",
    "mlp_out": "transformer.h.2.mlp.c_proj.weight",
    "embedding": "transformer.wte.weight",
    "output_head": "lm_head.weight",
}

# RCC types to plot (only the main ones user cares about)
main_rcc_types = [
    "RCC (W_t-W_0)x_t",
    "RCC W_0(x_t-x_0)",
    "RCC (x_t-x_0)",
    "activation_difference",
    "activation",
]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot RCC metrics from WandB projects')
    parser.add_argument('--force-refresh', action='store_true',
                        help='Force refresh cache: ignore existing cache and fetch fresh data from WandB')
    parser.add_argument('--widths', type=int, nargs='+', default=None,
                        help='Widths to include in plots (e.g., --widths 256 512 1024). If not specified, all widths are used.')
    args = parser.parse_args()

    # Plot all registered RCC types for all projects and metric paths
    print("="*80)
    print("PLOTTING RCC METRICS")
    if args.force_refresh:
        print("Mode: FORCE REFRESH (ignoring cache)")
    else:
        print("Mode: Using cache (use --force-refresh to fetch fresh data)")
    if args.widths:
        print(f"Filtering widths: {args.widths}")
    print("="*80)

    for project in projects:
        print(f"\n{'='*80}")
        print(f"PROJECT: {project}")
        print(f"{'='*80}")

        for layer_type, metric_path in metric_paths.items():
            print(f"\n  Layer type: {layer_type}")
            print(f"  Metric path: {metric_path}")

            for rcc_type in main_rcc_types:
                try:
                    plot_rcc_metric(project, metric_path, None, rcc_type,
                                   force_refresh=args.force_refresh,
                                   widths_filter=args.widths)
                except Exception as e:
                    print(f"    Skipping {rcc_type}: {e}")
