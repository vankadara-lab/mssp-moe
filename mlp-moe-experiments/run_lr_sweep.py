#!/usr/bin/env python3
"""
Learning rate sweep across widths
"""
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import os
import torch

try:
    from tuned_multipliers import TUNED_MULTIPLIERS, get_optimal_values
except ImportError:
    TUNED_MULTIPLIERS = None
    get_optimal_values = None

def run_training(args_tuple):
    """Run a single training job"""
    config, N, optimizer, lr, results_dir, seed, gpu_id, topk, P, T, use_ce, dataset, optimal, router_init, router_fn, post_agg_nonlin, last_layer_init, share_exp_in, share_exp_out, eval_chunk_size = args_tuple
    
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Create logs directory
    log_dir = Path(results_dir) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    routing = 'soft' if topk == 0 else f'top{topk}'
    log_file = log_dir / f'{config}_N{N}_{routing}_lr{lr}_seed{seed}.log'
    
    cmd = [
        'python', '-u', 'moe_training.py',  # -u for unbuffered output
        '--N', str(N),
        '--P', str(P),
        '--T', str(T),
        '--optimizer', optimizer,
        '--scaling_config', config,
        '--results_dir', results_dir,
        '--seed', str(seed),
        '--eta', str(lr),
        '--topk', str(topk),
        '--dataset', dataset,
        '--minimal_stats',
    ]

    # Add optimal multipliers if provided
    if optimal is not None:
        cmd.extend([
            '--init_std_mult', str(optimal['init_std_mult']),
            '--lr_mult_in', str(optimal['lr_mult_in']),
            '--lr_mult_out', str(optimal['lr_mult_out']),
            '--lr_mult_router', str(optimal['lr_mult_router']),
            '--lr_mult_expert1', str(optimal['lr_mult_expert1']),
            '--lr_mult_expert2', str(optimal['lr_mult_expert2']),
        ])

    if use_ce:
        cmd.append('--use_cross_entropy')

    if router_init is not None:
        cmd.extend(['--router_init', router_init])

    if router_fn is not None:
        cmd.extend(['--router_fn', router_fn])

    if post_agg_nonlin is not None:
        cmd.extend(['--post_agg_nonlin', post_agg_nonlin])

    if last_layer_init is not None:
        cmd.extend(['--last_layer_init', last_layer_init])

    if share_exp_in:
        cmd.append('--share_expert_weights_in')

    if share_exp_out:
        cmd.append('--share_expert_weights_out')

    if eval_chunk_size is not None:
        cmd.extend(['--eval_chunk_size', str(eval_chunk_size)])

    print(f"GPU {gpu_id}: config={config}, N={N}, {routing}, optimizer={optimizer}, lr={lr:.6f}, P={P}, T={T}, seed={seed}", flush=True)
    if optimal is not None:
        print(f"  Using optimal mults: init_std={optimal['init_std_mult']}, lr_in={optimal['lr_mult_in']}, lr_out={optimal['lr_mult_out']}", flush=True)
    print(f"  Logging to: {log_file}", flush=True)
    
    with open(log_file, 'w') as f:
        process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            f.write(line)
            f.flush()
            print(line, end='', flush=True)  # Print everything to console
        process.wait()
        if process.returncode != 0:
            print(f"  ERROR: Job failed with return code {process.returncode}", flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Config to test')
    parser.add_argument('--optimizer', type=str, required=True, help='Optimizer (sgd or adam)')
    parser.add_argument('--widths', type=int, nargs='+', required=True, help='Widths to test')
    parser.add_argument('--lr_values', type=float, nargs='+', default=None, help='Custom LR values (overrides default grid)')
    parser.add_argument('--results_dir', type=str, required=True, help='Results directory')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44, 45], help='Seeds')
    parser.add_argument('--num_gpus', type=int, default=8, help='Number of GPUs')
    parser.add_argument('--fixed_E', action='store_true', help='Fixed-E variant (k=2), else allscaling (k scales)')
    parser.add_argument('--P', type=int, default=50000, help='Number of samples')
    parser.add_argument('--T', type=int, default=1000, help='Number of steps')
    parser.add_argument('--use_cross_entropy', action='store_true', help='Use cross-entropy loss')
    parser.add_argument('--dataset', type=str, default='tinyimagenet', choices=['cifar10', 'tinyimagenet'], help='Dataset to use')
    parser.add_argument('--routing_types', type=str, nargs='+', default=['soft'],
                        choices=['soft', 'topk'], help='Routing types to test (default: soft)')
    parser.add_argument('--no_use_optimal', dest='use_optimal', action='store_false', default=True,
                        help='Disable loading optimal hyperparameters from tuned_multipliers.py')
    parser.add_argument('--use_optimal', dest='use_optimal', action='store_true',
                        help='Enable loading optimal hyperparameters (default, explicit alias for clarity)')
    parser.add_argument('--router_init', type=str, default=None, choices=['zero', 'mup', 'ntp', None],
                        help='Router initialization scheme (default: None, uses config default)')
    parser.add_argument('--router_fn', type=str, default=None, choices=['sigmoid', 'softmax', 'linear', None],
                        help='Router activation function (default: None, uses config default)')
    parser.add_argument('--post_agg_nonlin', type=str, default=None, choices=['rmsnorm', 'sigmoid', None],
                        help='Post-aggregation nonlinearity before output layer')
    parser.add_argument('--last_layer_init', type=str, default=None, choices=['zero', 'mup', 'ntp', None],
                        help='Last layer initialization scheme')
    parser.add_argument('--share_expert_weights_in', action='store_true',
                        help='Share expert input weights')
    parser.add_argument('--share_expert_weights_out', action='store_true',
                        help='Share expert output weights')
    parser.add_argument('--eval_chunk_size', type=int, default=None,
                        help='Chunk size for evaluation (reduces memory usage for large N)')
    parser.add_argument('--lr_mult_in', type=float, default=None, help='Override lr_mult_in (overrides use_optimal)')
    parser.add_argument('--lr_mult_out', type=float, default=None, help='Override lr_mult_out (overrides use_optimal)')
    parser.add_argument('--lr_mult_router', type=float, default=None, help='Override lr_mult_router (overrides use_optimal)')
    parser.add_argument('--lr_mult_expert1', type=float, default=None, help='Override lr_mult_expert1 (overrides use_optimal)')
    parser.add_argument('--lr_mult_expert2', type=float, default=None, help='Override lr_mult_expert2 (overrides use_optimal)')
    parser.add_argument('--init_std_mult', type=float, default=None, help='Override init_std_mult (overrides use_optimal)')
    args = parser.parse_args()

    # Load optimal hyperparameters (enabled by default)
    # We'll load optimal values for each routing type separately
    optimal_values = {}
    if args.use_optimal:
        if TUNED_MULTIPLIERS is None:
            print("WARNING: Optimal HP loading enabled but tuned_multipliers.py not found")
            print("Continuing without optimal hyperparameters\n")
        elif args.config not in TUNED_MULTIPLIERS:
            print(f"WARNING: No optimal values found for config '{args.config}'")
            print(f"Available configs: {list(TUNED_MULTIPLIERS.keys())}")
            print("Continuing without optimal hyperparameters\n")
        else:
            # Load optimal values for each requested routing type
            shared_experts = args.share_expert_weights_in or args.share_expert_weights_out
            variant = "shared experts" if shared_experts else "no weight sharing"
            for routing_type in args.routing_types:
                optimal = get_optimal_values(args.config, routing_type, shared_experts=shared_experts, last_layer_init=args.last_layer_init)
                if optimal is not None:
                    optimal_values[routing_type] = optimal
                    print(f"\n✓ Loaded optimal hyperparameters for {args.config} ({routing_type} routing, {variant}):")
                    print(f"  Base LR (will be overridden by sweep): {optimal['base_lr']}")
                    print(f"  Init std mult: {optimal['init_std_mult']}")
                    print(f"  LR multipliers: in={optimal['lr_mult_in']}, out={optimal['lr_mult_out']}, router={optimal['lr_mult_router']}")
                    print(f"  Expert LR mults: exp1={optimal['lr_mult_expert1']}, exp2={optimal['lr_mult_expert2']}")
                    print(f"  Best accuracy: {optimal['best_acc']:.4f}")
                else:
                    print(f"WARNING: No optimal values for routing mode '{routing_type}' in config '{args.config}'")
            print()

    # Apply explicit mult overrides (take precedence over use_optimal)
    override_keys = ['init_std_mult', 'lr_mult_in', 'lr_mult_out', 'lr_mult_router', 'lr_mult_expert1', 'lr_mult_expert2']
    overrides = {k: getattr(args, k) for k in override_keys if getattr(args, k) is not None}
    if overrides:
        print(f"Applying explicit mult overrides: {overrides}")
        # Apply to all routing types (create entry if missing)
        for routing_type in args.routing_types:
            if routing_type not in optimal_values:
                optimal_values[routing_type] = {'base_lr': 0.1, 'init_std_mult': 1.0,
                    'lr_mult_in': 1.0, 'lr_mult_out': 1.0, 'lr_mult_router': 1.0,
                    'lr_mult_expert1': 1.0, 'lr_mult_expert2': 1.0, 'best_acc': 0.0}
            optimal_values[routing_type].update(overrides)

    # Generate LR grid based on optimizer and loss function (or use custom values)
    if args.lr_values is not None:
        lr_grid = args.lr_values
        print(f"Using custom LR values: {args.lr_values}")
    elif args.optimizer == 'sgd':
        if args.use_cross_entropy:
            # SGD with CE: custom grid for stage 3
            lr_grid = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64, 1.28]
            print("Using default SGD + CE grid")
        else:
            # SGD with MSE: custom grid
            lr_grid = [0.00078125, 0.0015625, 0.003125, 0.00625, 0.0125, 0.025, 0.05, 0.1]
            print("Using default SGD + MSE grid")
    else:
        # Adam: custom grid with log-spaced values
        lr_grid = [3.16227766e-06, 1e-05, 3.16227766e-05, 0.0001,
                   0.000316227766, 0.001, 0.00316227766, 0.01]
        print("Using default Adam grid")

    # Auto-detect GPUs if -1 was specified
    if args.num_gpus == -1:
        args.num_gpus = torch.cuda.device_count()
        print(f"Auto-detected {args.num_gpus} GPUs\n")

    print(f"\nExperiment Configuration:")
    print(f"  Config: {args.config}")
    print(f"  Optimizer: {args.optimizer}")
    print(f"  Widths: {args.widths}")
    print(f"  LR grid ({len(lr_grid)} values): {lr_grid}")
    print(f"  Seeds: {args.seeds}")
    print(f"  Routing types: {args.routing_types}")
    print(f"  P={args.P}, T={args.T}")
    print(f"  Dataset: {args.dataset}")
    print(f"  Use CE: {args.use_cross_entropy}")

    # Build all jobs
    jobs = []
    for N in args.widths:
        # Determine topk values based on routing_types requested
        topk_list = []

        if 'soft' in args.routing_types:
            topk_list.append(0)

        if 'topk' in args.routing_types:
            if args.fixed_E:
                k = 2  # k=2 for fixed-E
            else:
                # Allscaling/Bottleneck: k scales proportionally with M
                M = 8 * (N // 128)  # M scales with N from base 128
                k = max(2, int(2 * (M / 8)))  # k scales proportionally
            topk_list.append(k)

        for topk in topk_list:
            # Determine routing type for this topk value
            routing_type = 'soft' if topk == 0 else 'topk'
            # Get optimal values for this routing type (None if not available)
            optimal = optimal_values.get(routing_type, None)

            for lr in lr_grid:
                for seed in args.seeds:
                    gpu_id = 0 if args.num_gpus == 0 else len(jobs) % args.num_gpus
                    jobs.append((args.config, N, args.optimizer, lr, args.results_dir, seed,
                                gpu_id, topk, args.P, args.T, args.use_cross_entropy, args.dataset, optimal, args.router_init, args.router_fn, args.post_agg_nonlin, args.last_layer_init, args.share_expert_weights_in, args.share_expert_weights_out, args.eval_chunk_size))

    print(f"\nTotal jobs to run: {len(jobs)}")
    print(f"  = {len(args.widths)} widths × {len(topk_list)} routing types × {len(lr_grid)} LRs × {len(args.seeds)} seeds")
    if args.num_gpus == 0:
        print(f"  Running on CPU (sequential)\n")
    else:
        print(f"  Distributed across {args.num_gpus} GPUs\n")

    # Use ProcessPoolExecutor with one worker per GPU (or 1 for CPU)
    # Each worker will process jobs sequentially on its assigned GPU
    max_workers = 1 if args.num_gpus == 0 else args.num_gpus
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(run_training, jobs)
    
    print("\nLR sweep complete!")
    
    # Generate LR sweep plots
    print("\nGenerating LR sweep plots...")
    subprocess.run(["python", "plotting/plot_lr_sweep.py", "--results_dir", args.results_dir])

if __name__ == '__main__':
    main()
