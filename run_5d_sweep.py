import subprocess
import itertools
from concurrent.futures import ProcessPoolExecutor
import os
import argparse
import torch

def center_first_order(values):
    """Reorder values to visit center first, then expand outward.

    For a list like [0.0039, 0.0625, 1.0, 16.0, 256.0], returns:
    [1.0, 0.0625, 16.0, 0.0039, 256.0]

    Strategy: Start at center index, alternate between smaller and larger indices.
    """
    if not values:
        return []

    n = len(values)
    center_idx = n // 2
    result = [values[center_idx]]

    # Alternate between left and right of center
    left = center_idx - 1
    right = center_idx + 1

    while left >= 0 or right < n:
        if left >= 0:
            result.append(values[left])
            left -= 1
        if right < n:
            result.append(values[right])
            right += 1

    return result

def run_job(args):
    (init_std, lr_in, lr_out, lr_router, lr_exp1, lr_exp2, gpu_id, config, N, P, T, optimizer,
     results_dir, seed, base_lr, nonlin, dataset,
     early_stop_threshold, early_stop_check_step, last_layer_init, share_exp_in, share_exp_out, topk) = args
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    cmd = [
        "python", "moe_training.py",
        "--N", str(N),
        "--P", str(P),
        "--T", str(T),
        "--eta", str(base_lr),
        "--router_fn", "sigmoid",
        "--nonlin", nonlin,
        "--seed", str(seed),
        "--optimizer", optimizer,
        "--scaling_config", config,
        "--results_dir", results_dir,
        "--minimal_stats",
        "--dataset", dataset,
        "--init_std_mult", str(init_std),
        "--lr_mult_in", str(lr_in),
        "--lr_mult_out", str(lr_out),
        "--lr_mult_router", str(lr_router),
        "--lr_mult_expert1", str(lr_exp1),
        "--lr_mult_expert2", str(lr_exp2)
    ]
    if early_stop_threshold:
        cmd.extend(["--early_stop_threshold", str(early_stop_threshold)])
        cmd.extend(["--early_stop_check_step", str(early_stop_check_step)])
    if last_layer_init:
        cmd.extend(["--last_layer_init", last_layer_init])
    if share_exp_in:
        cmd.append("--share_expert_weights_in")
    if share_exp_out:
        cmd.append("--share_expert_weights_out")
    if topk is not None:
        cmd.extend(["--topk", str(topk)])
    print(f"GPU {gpu_id}: init_std={init_std}, lr_in={lr_in}, lr_out={lr_out}, lr_router={lr_router}, lr_exp1={lr_exp1}, lr_exp2={lr_exp2}")
    subprocess.run(cmd, env=env)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--N', type=int, default=128)
    parser.add_argument('--P', type=int, default=5000)
    parser.add_argument('--T', type=int, default=500)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--base_lr', type=float, required=True)
    parser.add_argument('--nonlin', type=str, default='gelu')
    parser.add_argument('--init_std_values', type=float, nargs='+', required=True)
    parser.add_argument('--lr_in_values', type=float, nargs='+', required=True)
    parser.add_argument('--lr_out_values', type=float, nargs='+', required=True)
    parser.add_argument('--lr_router_values', type=float, nargs='+', required=True)
    parser.add_argument('--lr_expert1_values', type=float, nargs='+', required=True)
    parser.add_argument('--lr_expert2_values', type=float, nargs='+', required=True)
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'tinyimagenet'])
    parser.add_argument('--early_stop_threshold', type=float, default=None, help='Abort if val_acc < best_val_acc / threshold (threshold=2 means keep runs with acc >= 0.5 * best)')
    parser.add_argument('--early_stop_check_step', type=int, default=250)
    parser.add_argument('--seed', type=int, nargs='+', default=[52])
    parser.add_argument('--results_dir', type=str, default=None)
    parser.add_argument('--num_gpus', type=int, default=8)
    parser.add_argument('--last_layer_init', type=str, default=None, choices=['mup', 'ntp', 'zero', None],
                        help='Last layer initialization scheme')
    parser.add_argument('--share_expert_weights_in', action='store_true',
                        help='Share expert input weights')
    parser.add_argument('--share_expert_weights_out', action='store_true',
                        help='Share expert output weights')
    parser.add_argument('--topk', type=int, default=None,
                        help='Top-k routing (0=soft, 2=top-2, etc.)')
    args = parser.parse_args()

    # Auto-detect GPUs if -1 was specified
    if args.num_gpus == -1:
        args.num_gpus = torch.cuda.device_count()
        print(f"Auto-detected {args.num_gpus} GPUs")

    # Auto-generate results_dir if not provided
    if args.results_dir is None:
        from datetime import datetime
        date_str = datetime.now().strftime("%d%m")
        args.results_dir = f"results/5d_sweep_lr_init_{args.config}_{date_str}"

    seeds = args.seed

    # Apply center-first ordering to all parameter value lists
    # This explores central values first, then expands outward
    init_std_ordered = center_first_order(args.init_std_values)
    lr_in_ordered = center_first_order(args.lr_in_values)
    lr_out_ordered = center_first_order(args.lr_out_values)
    lr_router_ordered = center_first_order(args.lr_router_values)
    lr_exp1_ordered = center_first_order(args.lr_expert1_values)
    lr_exp2_ordered = center_first_order(args.lr_expert2_values)

    # Generate all jobs with init_std in outer loop (after seeds)
    # This allows early stopping to rule out bad initialization values quickly
    jobs = [(init_std, lr_in, lr_out, lr_router, lr_exp1, lr_exp2, 0 if args.num_gpus == 0 else i % args.num_gpus,
             args.config, args.N, args.P, args.T, args.optimizer, args.results_dir, seed,
             args.base_lr, args.nonlin,
             args.dataset, args.early_stop_threshold, args.early_stop_check_step, args.last_layer_init,
             args.share_expert_weights_in, args.share_expert_weights_out, args.topk)
            for i, (seed, init_std, lr_in, lr_out, lr_router, lr_exp1, lr_exp2) in
            enumerate(itertools.product(seeds, init_std_ordered, lr_in_ordered, lr_out_ordered,
                                       lr_router_ordered, lr_exp1_ordered, lr_exp2_ordered))]

    print(f"Total jobs: {len(jobs)}")
    print(f"Seeds: {seeds}")
    print(f"Init std values (center-first): {init_std_ordered} (original: {args.init_std_values})")
    print(f"LR in values (center-first): {lr_in_ordered} (original: {args.lr_in_values})")
    print(f"LR out values (center-first): {lr_out_ordered} (original: {args.lr_out_values})")
    print(f"LR router values (center-first): {lr_router_ordered} (original: {args.lr_router_values})")
    print(f"LR expert1 values (center-first): {lr_exp1_ordered} (original: {args.lr_expert1_values})")
    print(f"LR expert2 values (center-first): {lr_exp2_ordered} (original: {args.lr_expert2_values})")
    
    with ProcessPoolExecutor(max_workers=max(1, args.num_gpus)) as executor:
        executor.map(run_job, jobs)
    
    print("\nGenerating 5D sweep plots...")
    subprocess.run(["python", "plotting/plot_5d_sweep.py", "--results_dir", args.results_dir, "--config", args.config])

if __name__ == '__main__':
    main()
