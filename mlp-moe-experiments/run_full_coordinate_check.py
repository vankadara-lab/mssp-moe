#!/usr/bin/env python3
"""
Run full coordinate check with torch implementation: train models and generate plots.
"""
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from scaling_configs import CONFIGS

# Try to import optimal values (optional)
try:
    from tuned_multipliers import TUNED_MULTIPLIERS, get_optimal_values
except ImportError:
    TUNED_MULTIPLIERS = None
    get_optimal_values = None


def _run_gpu_bucket(bucket):
    """Run a list of jobs sequentially on one GPU (one worker process per GPU)."""
    return [_run_single_job(j) for j in bucket]


def _run_single_job(args_tuple):
    """Run a single training job (must be at module level for pickling)"""
    config_name, N, seed, config_router_fn, P, P_val, T, eta, results_dir, nonlin, router_fn_scale_alpha, optimizer, adam_eps, router_init, last_layer_init, sigmoid_norm, input_lr_zero, separate_aggregation, post_agg_nonlin, use_residual, expert_nonlin, init_std_mult, lr_mult_in, lr_mult_out, lr_mult_router, lr_mult_expert1, lr_mult_expert2, dataset, topk, extensive_logging, share_expert_weights_in, share_expert_weights_out, topk_mini_batch_size, eval_chunk_size, gpu_id = args_tuple

    print(f"\nRunning {config_name} N={N} seed={seed} topk={topk} on GPU {gpu_id}")

    cmd = [
        'python', 'moe_training.py',
        '--N', str(N),
        '--P', str(P),
        '--P_val', str(P_val),
        '--T', str(T),
        '--eta', str(eta),
        '--gamma', '1.0',
        '--seed', str(seed),
        '--scaling_config', config_name,
        '--results_dir', str(results_dir),
        '--nonlin', nonlin,
        '--router_fn', config_router_fn,
        '--optimizer', optimizer,
        '--adam_eps', str(adam_eps),
        '--dataset', dataset,
    ]

    if init_std_mult is not None:
        cmd.extend(['--init_std_mult', str(init_std_mult)])
    if lr_mult_in is not None:
        cmd.extend(['--lr_mult_in', str(lr_mult_in)])
    if lr_mult_out is not None:
        cmd.extend(['--lr_mult_out', str(lr_mult_out)])
    if lr_mult_router is not None:
        cmd.extend(['--lr_mult_router', str(lr_mult_router)])
    if lr_mult_expert1 is not None:
        cmd.extend(['--lr_mult_expert1', str(lr_mult_expert1)])
    if lr_mult_expert2 is not None:
        cmd.extend(['--lr_mult_expert2', str(lr_mult_expert2)])

    if router_fn_scale_alpha is not None:
        cmd.extend(['--router_fn_scale_alpha', str(router_fn_scale_alpha)])

    if router_init is not None:
        cmd.extend(['--router_init', router_init])
    if last_layer_init is not None:
        cmd.extend(['--last_layer_init', last_layer_init])
    if topk is not None:
        cmd.extend(['--topk', str(topk)])
    if sigmoid_norm:
        cmd.append('--sigmoid_norm')
    if input_lr_zero:
        cmd.append('--input_lr_zero')
    if separate_aggregation:
        cmd.append('--separate_aggregation')
    if post_agg_nonlin:
        cmd.extend(['--post_agg_nonlin', post_agg_nonlin])
    if use_residual:
        cmd.append('--use_residual')
    if expert_nonlin:
        cmd.extend(['--expert_nonlin', expert_nonlin])
    if extensive_logging:
        cmd.append('--extensive_logging')
    if share_expert_weights_in:
        cmd.append('--share_expert_weights_in')
    if share_expert_weights_out:
        cmd.append('--share_expert_weights_out')
    if topk_mini_batch_size is not None:
        cmd.extend(['--topk_mini_batch_size', str(topk_mini_batch_size)])
    if eval_chunk_size is not None:
        cmd.extend(['--eval_chunk_size', str(eval_chunk_size)])

    import os
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    try:
        subprocess.run(cmd, check=True, env=env)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {config_name} N={N} seed={seed}: {e}")
        return False


def run_coordinate_check(configs, widths, widths_allscaling, P, P_val, T, eta, results_dir, seeds=[42], nonlin='tanh', router_fn='softmax', sigmoid_norm=False, router_init=None, last_layer_init=None, input_lr_zero=False, router_fn_scale_alpha=None, optimizer='sgd', adam_eps=1e-8, num_gpus=0, separate_aggregation=False, post_agg_nonlin=None, use_residual=False, expert_nonlin=None, init_std_mult=None, lr_mult_in=None, lr_mult_out=None, lr_mult_router=None, lr_mult_expert1=None, lr_mult_expert2=None, dataset='tinyimagenet', topk=None, extensive_logging=True, share_expert_weights_in=False, share_expert_weights_out=False, topk_mini_batch_size=None, eval_chunk_size=None):
    """Run coordinate check experiments"""

    print(f"Starting coordinate check...")
    print(f"Configs: {configs}")
    print(f"Widths (standard): {widths}")
    print(f"Widths (allscaling): {widths_allscaling}")
    print(f"Seeds: {seeds}")
    print(f"Nonlin: {nonlin}, Router: {router_fn}, Sigmoid norm: {sigmoid_norm}")
    print(f"Router fn scale alpha: {router_fn_scale_alpha}")
    print(f"Input LR zero: {input_lr_zero}")
    print(f"Optimizer: {optimizer}")
    print(f"Results: {results_dir}")
    print(f"GPUs: {num_gpus}")

    # Build list of all jobs
    jobs = []
    for config_name in configs:
        if config_name not in CONFIGS:
            print(f"Warning: Unknown config {config_name}, skipping")
            continue

        if 'allscaling' in config_name:
            widths_to_use = widths_allscaling
        else:
            widths_to_use = widths

        config_router_fn = 'sigmoid' if config_name.startswith('fixed_E') else router_fn

        is_fixed_e = 'fixed_E' in config_name
        is_scaling = 'allscaling' in config_name or 'bottleneck' in config_name

        for N in widths_to_use:
            if topk is not None and topk > 0:
                if is_fixed_e:
                    topk_for_width = topk
                elif is_scaling:
                    base_N = 128
                    topk_for_width = int(topk * (N / base_N))
                    print(f"  {config_name} N={N}: scaling topk from base {topk} to {topk_for_width} (M={N//16})")
                else:
                    topk_for_width = topk
            else:
                topk_for_width = topk

            for seed in seeds:
                jobs.append((config_name, N, seed, config_router_fn, topk_for_width))

    # Build job args (shared between parallel and sequential paths)
    job_args = []
    for i, (config_name, N, seed, config_router_fn, topk_for_width) in enumerate(jobs):
        gpu_id = i % num_gpus if num_gpus > 0 else 0
        job_args.append((config_name, N, seed, config_router_fn, P, P_val, T, eta, results_dir, nonlin, router_fn_scale_alpha, optimizer, adam_eps, router_init, last_layer_init, sigmoid_norm, input_lr_zero, separate_aggregation, post_agg_nonlin, use_residual, expert_nonlin, init_std_mult, lr_mult_in, lr_mult_out, lr_mult_router, lr_mult_expert1, lr_mult_expert2, dataset, topk_for_width, extensive_logging, share_expert_weights_in, share_expert_weights_out, topk_mini_batch_size, eval_chunk_size, gpu_id))

    if num_gpus > 0:
        import multiprocessing as mp

        gpu_buckets = [[] for _ in range(num_gpus)]
        for job in job_args:
            gpu_buckets[job[-1]].append(job)

        with mp.Pool(processes=num_gpus) as pool:
            bucket_results = pool.map(_run_gpu_bucket, gpu_buckets)

        results = [r for bucket in bucket_results for r in bucket]
    else:
        results = [_run_single_job(j) for j in job_args]

    print(f"\nCompleted {sum(results)}/{len(results)} jobs successfully")
    print(f"\n✓ Coordinate check complete! Results in {results_dir}/")


def generate_plots(results_dir):
    """Generate plots and exponent tables"""
    print("\nGenerating plots and tables...")

    cmd = ['python', 'plotting/plot_coordinate_check.py', '--results_dir', str(results_dir)]

    try:
        subprocess.run(cmd, check=True)
        print(f"✓ Plots and tables generated in {results_dir}/")
    except subprocess.CalledProcessError as e:
        print(f"Error generating plots: {e}")


def main():
    parser = argparse.ArgumentParser(description='Run coordinate check with torch implementation')
    parser.add_argument('--configs', nargs='+', required=True,
                       help='Configs to test')
    parser.add_argument('--widths', type=int, nargs='+', default=[256, 512, 1024, 2048],
                       help='Widths for standard configs')
    parser.add_argument('--widths_allscaling', type=int, nargs='+', default=[128, 256, 512, 1024],
                       help='Widths for allscaling configs')
    parser.add_argument('--P', type=int, default=50000, help='Training samples')
    parser.add_argument('--P_val', type=int, default=1000, help='Validation samples')
    parser.add_argument('--T', type=int, default=1000, help='Training iterations')
    parser.add_argument('--eta', type=float, default=None, help='Base learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (or first seed if --seeds is used)')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45], help='Multiple seeds to run (overrides --seed)')
    parser.add_argument('--nonlin', type=str, default='gelu', help='Nonlinearity function')
    parser.add_argument('--expert_nonlin', type=str, default=None, help='Expert nonlinearity (default: use --nonlin)')
    parser.add_argument('--router_fn', type=str, default='sigmoid', help='Router activation function')
    parser.add_argument('--router_init', type=str, default=None, help='Router initialization (overrides config)')
    parser.add_argument('--last_layer_init', type=str, default='zero', help='Last layer initialization (overrides config)')
    parser.add_argument('--sigmoid_norm', action='store_true', help='Enable sigmoid normalization')
    parser.add_argument('--router_fn_scale_alpha', type=float, default=None, help='Scale sigmoid router weights by 1/M^alpha (None=use config default)')
    parser.add_argument('--input_lr_zero', action='store_true', help='Set input layer learning rate to 0')
    parser.add_argument('--separate_aggregation', action='store_true', help='Separate scaling for h^L_0 (by 1/M^0.5) and Delta h^L (by 1/M)')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'], help='Optimizer to use')
    parser.add_argument('--adam_eps', type=float, default=1e-8, help='Base epsilon for Adam optimizer (default: 1e-8)')
    parser.add_argument('--num_gpus', type=int, default=-1, help='Number of GPUs to use for parallel execution (-1=auto-detect all available, 0=CPU only)')

    parser.add_argument('--init_std_mult', type=float, default=None, help='Init std multiplier for all layers')
    parser.add_argument('--lr_mult_in', type=float, default=None, help='LR multiplier for w_in layer')
    parser.add_argument('--lr_mult_out', type=float, default=None, help='LR multiplier for w_out layer')
    parser.add_argument('--lr_mult_router', type=float, default=None, help='LR multiplier for router layer')
    parser.add_argument('--lr_mult_expert1', type=float, default=None, help='LR multiplier for w_expert1 layer')
    parser.add_argument('--lr_mult_expert2', type=float, default=None, help='LR multiplier for w_expert2 layer')

    parser.add_argument('--post_agg_nonlin', type=str, default=None, choices=[None, 'rmsnorm', 'sigmoid'], help='Post-aggregation nonlinearity before output layer')
    parser.add_argument('--use_residual', action='store_true', help='Add residual connection from input to output (skip experts and router)')
    parser.add_argument('--results_dir', type=str, default=None, help='Results directory')
    parser.add_argument('--skip_training', action='store_true', help='Skip training, only generate plots')
    parser.add_argument('--skip_plots', action='store_true', help='Skip plot generation')
    parser.add_argument('--no_extensive_logging', action='store_false', dest='extensive_logging', default=True, help='Disable extensive logging (decomposition analyses enabled by default)')

    parser.add_argument('--use_optimal', action='store_true', help='Load optimal hyperparameters from tuned_multipliers.py (requires single config). Use --eta to override the optimal base LR.')
    parser.add_argument('--routing_mode', type=str, default='soft', choices=['soft', 'topk'], help='Routing mode for loading optimal values (used with --use_optimal)')
    parser.add_argument('--topk', type=int, default=None, help='Top-k routing: base value of k at N=128 (0=soft routing, 2=top-2, etc.). For allscaling/bottleneck configs, k scales proportionally with M: k=topk*(N/128). For fixed-E configs, k stays constant.')
    parser.add_argument('--dataset', type=str, default='tinyimagenet', help='Dataset to use (mnist, cifar10, tinyimagenet)')
    parser.add_argument('--share_expert_weights_in', action='store_true', help='Share input layer weights across all experts at initialization')
    parser.add_argument('--share_expert_weights_out', action='store_true', help='Share output layer weights across all experts at initialization')
    parser.add_argument('--topk_mini_batch_size', type=int, default=None, help='Mini-batch size for memory-efficient topk routing (processes this many samples at once per expert). Useful for large N to avoid OOM.')
    parser.add_argument('--eval_chunk_size', type=int, default=None, help='Chunk size for topk evaluation to avoid OOM (processes batches in chunks during eval). Useful for large N to avoid OOM.')

    args = parser.parse_args()

    # Load optimal hyperparameters if requested
    if args.use_optimal:
        if TUNED_MULTIPLIERS is None:
            print("ERROR: --use_optimal requires tuned_multipliers.py to be present")
            return
        if len(args.configs) != 1:
            print("ERROR: --use_optimal requires exactly one config")
            return

        config_name = args.configs[0]
        if config_name not in TUNED_MULTIPLIERS:
            print(f"ERROR: No optimal values found for config '{config_name}'")
            print(f"Available configs: {list(TUNED_MULTIPLIERS.keys())}")
            return

        shared_experts = args.share_expert_weights_in or args.share_expert_weights_out
        optimal = get_optimal_values(config_name, args.routing_mode, shared_experts=shared_experts, last_layer_init=args.last_layer_init)
        if optimal is None:
            print(f"ERROR: No optimal values for routing mode '{args.routing_mode}' in config '{config_name}'")
            return

        eta_was_provided = args.eta != parser.get_default('eta')

        if not eta_was_provided:
            args.eta = optimal['base_lr']

        args.init_std_mult = optimal['init_std_mult']
        args.lr_mult_in = optimal['lr_mult_in']
        args.lr_mult_out = optimal['lr_mult_out']
        args.lr_mult_router = optimal['lr_mult_router']
        args.lr_mult_expert1 = optimal['lr_mult_expert1']
        args.lr_mult_expert2 = optimal['lr_mult_expert2']

        variant = "shared experts" if shared_experts else "no weight sharing"
        print(f"\n✓ Loaded optimal hyperparameters for {config_name} ({args.routing_mode} routing, {variant}):")
        if eta_was_provided:
            print(f"  Base LR: {args.eta} (user-provided, optimal={optimal['base_lr']})")
        else:
            print(f"  Base LR: {args.eta} (optimal)")
        print(f"  Init std mult: {args.init_std_mult}")
        print(f"  LR multipliers: in={args.lr_mult_in}, out={args.lr_mult_out}, router={args.lr_mult_router}")
        print(f"  Expert LR mults: exp1={args.lr_mult_expert1}, exp2={args.lr_mult_expert2}")
        print(f"  Best accuracy: {optimal['best_acc']:.4f}\n")

    if args.topk is None:
        if args.routing_mode == 'soft':
            args.topk = 0
        elif args.routing_mode == 'topk':
            args.topk = 2
        print(f"Auto-setting --topk={args.topk} based on --routing_mode={args.routing_mode}")

    if args.separate_aggregation and args.router_fn_scale_alpha is None:
        args.router_fn_scale_alpha = 0.0
        print("Setting router_fn_scale_alpha=0.0 for separate_aggregation")

    if args.num_gpus == -1:
        import torch
        args.num_gpus = torch.cuda.device_count()
        print(f"Auto-detected {args.num_gpus} GPUs")

    if args.eta is None:
        args.eta = 0.001

    if args.results_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')
        config_str = '_'.join(args.configs[:2]) if len(args.configs) <= 2 else f"{args.configs[0]}_and_{len(args.configs)-1}more"
        width_str = f"w{min(args.widths)}-{max(args.widths)}"
        features = []
        if args.post_agg_nonlin:
            features.append(args.post_agg_nonlin)
        if args.separate_aggregation:
            features.append('sepagg')
        if args.input_lr_zero:
            features.append('noinlr')
        feature_str = '_'.join(features) if features else 'default'
        args.results_dir = f'results/coord_{config_str}_{width_str}_{feature_str}_{timestamp}'
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_training:
        seeds = args.seeds if args.seeds else [args.seed]
        run_coordinate_check(
            args.configs,
            args.widths,
            args.widths_allscaling,
            args.P,
            args.P_val,
            args.T,
            args.eta,
            results_dir,
            seeds,
            args.nonlin,
            args.router_fn,
            args.sigmoid_norm,
            args.router_init,
            args.last_layer_init,
            args.input_lr_zero,
            args.router_fn_scale_alpha,
            args.optimizer,
            args.adam_eps,
            args.num_gpus,
            args.separate_aggregation,
            args.post_agg_nonlin,
            args.use_residual,
            args.expert_nonlin,
            args.init_std_mult,
            args.lr_mult_in,
            args.lr_mult_out,
            args.lr_mult_router,
            args.lr_mult_expert1,
            args.lr_mult_expert2,
            args.dataset,
            args.topk,
            args.extensive_logging,
            args.share_expert_weights_in,
            args.share_expert_weights_out,
            args.topk_mini_batch_size,
            args.eval_chunk_size
        )

    if not args.skip_plots:
        generate_plots(results_dir)

    print(f"\n✓ Done! Results in {results_dir}/")


if __name__ == '__main__':
    main()
