import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np
from pathlib import Path
import json
import os
from moe_logging import compute_stats, compute_effective_updates, compute_total_effective_updates_per_layer, compute_propagating_updates, compute_gradient_norms, compute_h_L_rms_diff, compute_h_L, compute_h_L_rms, compute_activation_rms, compute_output_gradient, compute_weight_rms_norms, compute_hagg_decomposition, compute_expert_grad_h1_decomposition
from scaling_configs import get_config
try:
    from tuned_multipliers import get_optimal_values as _get_optimal_values
except ImportError:
    _get_optimal_values = None
from utils.tiny_imagenet import TinyImageNet

# Set PyTorch memory allocator to avoid fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

class MoEModel(nn.Module):
    def __init__(self, args, scaling_config=None, input_dim=3072):
        super().__init__()
        self.args = args
        self.N, self.M, self.N_expert, self.input_dim = args.N, args.M, args.N_expert, input_dim
        self.scaling_config = scaling_config
        self.post_agg_nonlin = args.post_agg_nonlin if hasattr(args, 'post_agg_nonlin') else None
        
        # Initialize all weights as N(0,1) then apply scaling
        self.w_in = nn.Parameter(torch.randn(input_dim, self.N))
        self.w_router = nn.Parameter(torch.randn(self.N, self.M))
        self.w_expert1 = nn.Parameter(torch.randn(self.M, self.N, self.N_expert))
        self.w_expert2 = nn.Parameter(torch.randn(self.M, self.N_expert, self.N))
        # Binary MSE: scalar output, Multi-class MSE or CE: vector output
        n_out = 1 if (not args.use_cross_entropy and args.num_classes == 2) else args.num_classes
        self.w_out = nn.Parameter(torch.randn(self.N, n_out))
        
        if scaling_config:
            # Apply init scaling with optional std multiplier
            init_std_mult = args.init_std_mult if args.init_std_mult is not None else 1.0
            self.w_in.data *= init_std_mult * scaling_config.get_init_scale('w_in', self.N, self.M, self.N_expert, input_dim, self.N)
            self.w_router.data *= init_std_mult * scaling_config.get_init_scale('w_router', self.N, self.M, self.N_expert, self.N, self.M)
            self.w_expert1.data *= init_std_mult * scaling_config.get_init_scale('w_expert1', self.N, self.M, self.N_expert, self.N, self.N_expert)
            self.w_expert2.data *= init_std_mult * scaling_config.get_init_scale('w_expert2', self.N, self.M, self.N_expert, self.N_expert, self.N)
            self.w_out.data *= init_std_mult * scaling_config.get_init_scale('w_out', self.N, self.M, self.N_expert, self.N, n_out)
            
            # Zero init if needed (can be overridden by args)
            # Router initialization
            if args.router_init is not None:
                # Flag overrides config
                if args.router_init == 'zero':
                    self.w_router.data.zero_()
                # else: keep the scaled init applied above
            else:
                # Use config's default
                if scaling_config.should_init_zero('w_router'):
                    self.w_router.data.zero_()

            # Last layer initialization
            if args.last_layer_init is not None:
                # Flag overrides config
                if args.last_layer_init == 'zero':
                    self.w_out.data.zero_()
                # else: keep the scaled init applied above
            else:
                # Use config's default
                if scaling_config.should_init_zero('w_out'):
                    self.w_out.data.zero_()

            # Share expert weights at initialization if requested
            if args.share_expert_weights_in:
                # Copy first expert's input weights to all other experts
                for m in range(1, self.M):
                    self.w_expert1.data[m] = self.w_expert1.data[0].clone()

            if args.share_expert_weights_out:
                # Copy first expert's output weights to all other experts
                for m in range(1, self.M):
                    self.w_expert2.data[m] = self.w_expert2.data[0].clone()

            # Forward pass multipliers from config
            self.input_scale = scaling_config.get_forward_scale('w_in', self.N, self.M, self.N_expert, input_dim, self.N)
            self.router_scale = scaling_config.get_forward_scale('w_router', self.N, self.M, self.N_expert, self.N, self.M)
            self.expert1_scale = scaling_config.get_forward_scale('w_expert1', self.N, self.M, self.N_expert, self.N, self.N_expert)
            self.expert2_scale = scaling_config.get_forward_scale('w_expert2', self.N, self.M, self.N_expert, self.N_expert, self.N)
            self.out_scale = scaling_config.get_forward_scale('w_out', self.N, self.M, self.N_expert, self.N, n_out)
        else:
            # Legacy: use args-based scaling
            if args.router_init == 'zero':
                self.w_router.data.zero_()
            elif args.router_init == 'nonasymp':
                scale = self.N / (self.N / np.sqrt(self.M) + np.sqrt(self.N))
                self.w_router.data *= scale
            elif args.router_init == 'nonasympsqrt':
                scale = np.sqrt(self.N) / (self.N / np.sqrt(self.M) + np.sqrt(self.N))
                self.w_router.data *= scale

            if args.last_layer_init == 'zero':
                self.w_out.data.zero_()

            # Share expert weights at initialization if requested (legacy path)
            if args.share_expert_weights_in:
                # Copy first expert's input weights to all other experts
                for m in range(1, self.M):
                    self.w_expert1.data[m] = self.w_expert1.data[0].clone()

            if args.share_expert_weights_out:
                # Copy first expert's output weights to all other experts
                for m in range(1, self.M):
                    self.w_expert2.data[m] = self.w_expert2.data[0].clone()

            self.input_scale = 1 / np.sqrt(input_dim)
            self.router_scale = 1 / self.N if args.router_init in ['mup', 'zero', 'nonasymp'] else 1 / np.sqrt(self.N)
            self.expert1_scale = 1 / self.N if args.first_expert_layer_init == 'mup' else 1 / np.sqrt(self.N)
            self.expert2_scale = 1 / np.sqrt(self.N_expert)
            self.out_scale = 1 / self.N if args.last_layer_init in ['mup', 'zero'] else 1 / np.sqrt(self.N)

    def _forward_topk_batch(self, h, topk_vals, topk_idx, model_init=None):
        """
        Memory-efficient topk routing that loops over experts instead of materializing all weights.
        Processes mini-batches (default 64) and loops over k selected experts.
        Memory scales with mini_batch_size * N * N_expert (not batch_size * k * N * N_expert).

        Args:
            h: Hidden states [batch_size, N]
            topk_vals: Top-k router logits [batch_size, k]
            topk_idx: Top-k expert indices [batch_size, k]
            model_init: Initial model for separate aggregation (optional)

        Returns:
            combined: Expert outputs [batch_size, N]
        """
        # Apply router function to top-k logits
        if self.args.router_fn == 'softmax':
            router_weights_topk = F.softmax(topk_vals, dim=-1)
        elif self.args.router_fn == 'sigmoid':
            M_base = self.args.M_base if hasattr(self.args, 'M_base') else 8
            router_weights_topk = torch.sigmoid(topk_vals) / ((self.M / M_base) ** self.args.router_fn_scale_alpha)
            if self.args.sigmoid_norm:
                router_weights_topk = router_weights_topk / (router_weights_topk.sum(dim=-1, keepdim=True) + 1e-10)
        elif self.args.router_fn == 'linear':
            M_base = self.args.M_base if hasattr(self.args, 'M_base') else 8
            router_weights_topk = topk_vals / ((self.M / M_base) ** self.args.router_fn_scale_alpha)
        else:
            raise ValueError(f"Unknown router_fn: {self.args.router_fn}")

        # Memory-efficient implementation: loop over experts, process in mini-batches
        batch_size = h.shape[0]
        k = self.args.topk
        scale1 = self.expert1_scale
        scale2 = self.expert2_scale

        # Use mini-batch size for memory efficiency (default 1024)
        mini_batch_size = getattr(self.args, 'topk_mini_batch_size', 1024)

        combined = torch.zeros(batch_size, self.N, device=h.device, dtype=h.dtype)

        # Process in mini-batches
        num_mini_batches = (batch_size + mini_batch_size - 1) // mini_batch_size

        for mb_idx in range(num_mini_batches):
            mb_start = mb_idx * mini_batch_size
            mb_end = min(mb_start + mini_batch_size, batch_size)

            h_mb = h[mb_start:mb_end]  # [mini_batch, N]
            topk_idx_mb = topk_idx[mb_start:mb_end]  # [mini_batch, k]
            router_weights_mb = router_weights_topk[mb_start:mb_end]  # [mini_batch, k]

            # Loop over k selected experts
            for j in range(k):
                expert_indices = topk_idx_mb[:, j]  # [mini_batch]

                # Find unique experts in this mini-batch
                unique_experts = torch.unique(expert_indices)

                for expert_idx in unique_experts:
                    # Mask for samples that selected this expert at position j
                    mask = (expert_indices == expert_idx)  # [mini_batch]
                    if not mask.any():
                        continue

                    # Get samples that use this expert
                    h_expert = h_mb[mask]  # [n_samples, N]
                    router_weight = router_weights_mb[mask, j]  # [n_samples]

                    # Get expert weights (only one expert, not all k)
                    w1 = self.w_expert1[expert_idx] * scale1  # [N, N_expert]
                    w2 = self.w_expert2[expert_idx] * scale2  # [N_expert, N]

                    # Forward through expert
                    e = h_expert @ w1  # [n_samples, N_expert]
                    e = self.apply_expert_nonlin(e)

                    # Apply expert2 layer with separate aggregation if enabled
                    if self.args.separate_aggregation and model_init is not None:
                        w2_init = model_init.w_expert2[expert_idx] * scale2
                        w2_update = w2 - w2_init
                        expert_out = (e @ w2_init) / np.sqrt(self.M) + (e @ w2_update) / self.M
                    elif self.args.separate_aggregation:
                        expert_out = (e @ w2) / np.sqrt(self.M)
                    else:
                        expert_out = e @ w2

                    # Accumulate weighted expert outputs
                    combined[mb_start:mb_end][mask] += router_weight.unsqueeze(-1) * expert_out

        return combined

    def _forward_topk_chunked(self, h, topk_vals, topk_idx, chunk_size, model_init=None):
        """
        Process topk routing in chunks to avoid OOM.
        Memory scales with chunk_size * k instead of batch_size * k.

        Args:
            h: Hidden states [batch_size, N]
            topk_vals: Top-k router logits [batch_size, k]
            topk_idx: Top-k expert indices [batch_size, k]
            chunk_size: Number of samples to process at once
            model_init: Initial model for separate aggregation (optional)

        Returns:
            combined: Expert outputs [batch_size, N]
        """
        batch_size = h.shape[0]
        num_chunks = (batch_size + chunk_size - 1) // chunk_size

        combined_chunks = []
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, batch_size)

            # Process this chunk
            h_chunk = h[start_idx:end_idx]
            topk_vals_chunk = topk_vals[start_idx:end_idx]
            topk_idx_chunk = topk_idx[start_idx:end_idx]

            combined_chunk = self._forward_topk_batch(h_chunk, topk_vals_chunk, topk_idx_chunk, model_init)
            combined_chunks.append(combined_chunk)

        # Concatenate all chunks
        combined = torch.cat(combined_chunks, dim=0)
        return combined

    def forward(self, x, model_init=None, return_combined=False):
        # Input layer
        h = x @ (self.input_scale * self.w_in)
        h = self.apply_nonlin(h)
        
        # Router
        router_logits = h @ (self.router_scale * self.w_router)
        
        # Top-k routing: only compute selected experts
        if self.args.topk > 0:
            topk_vals, topk_idx = torch.topk(router_logits, self.args.topk, dim=-1)  # [batch, k]

            # Use chunked processing for large batches in eval mode
            batch_size = h.shape[0]
            chunk_size = self.args.eval_chunk_size
            if not self.training and batch_size > chunk_size:
                # Memory-efficient chunked processing
                combined = self._forward_topk_chunked(h, topk_vals, topk_idx, chunk_size, model_init)
            else:
                # Original vectorized implementation for small batches or training
                combined = self._forward_topk_batch(h, topk_vals, topk_idx, model_init)
        else:
            # Soft routing: use all experts (original implementation)
            if self.args.router_fn == 'softmax':
                router_weights = F.softmax(router_logits, dim=-1)
            elif self.args.router_fn == 'sigmoid':
                M_base = self.args.M_base if hasattr(self.args, 'M_base') else 8
                router_weights = torch.sigmoid(router_logits) / ((self.M / M_base) ** self.args.router_fn_scale_alpha)
                if self.args.sigmoid_norm:
                    router_weights = router_weights / (router_weights.sum(dim=-1, keepdim=True) + 1e-10)
            elif self.args.router_fn == 'linear':
                M_base = self.args.M_base if hasattr(self.args, 'M_base') else 8
                router_weights = router_logits / ((self.M / M_base) ** self.args.router_fn_scale_alpha)
            else:
                raise ValueError(f"Unknown router_fn: {self.args.router_fn}")

            # Experts - simple loop with separate aggregation
            scale1 = self.expert1_scale
            scale2 = self.expert2_scale

            M = self.M
            outs = []
            for m in range(M):
                e = self.apply_expert_nonlin(h @ (scale1 * self.w_expert1[m]))

                # Apply separate aggregation: W_0 x_t / sqrt(M) + (W_t - W_0) x_t / M
                if self.args.separate_aggregation:
                    if model_init is not None:
                        # After first iteration: separate W_0 and delta W
                        w2_init = model_init.w_expert2[m]
                        w2_update = self.w_expert2[m] - w2_init
                        expert_out = (e @ (scale2 * w2_init)) / np.sqrt(M) + (e @ (scale2 * w2_update)) / M
                    else:
                        # First iteration: just scale by 1/sqrt(M)
                        expert_out = (e @ (scale2 * self.w_expert2[m])) / np.sqrt(M)
                else:
                    # No separate aggregation
                    expert_out = e @ (scale2 * self.w_expert2[m])

                outs.append(expert_out)

            expert_outputs = torch.stack(outs, dim=0)  # [M, B, N]
            expert_outputs = expert_outputs.transpose(0, 1).transpose(1, 2)  # [M, B, N] -> [B, N, M]

            # Combine experts
            combined = (expert_outputs * router_weights.unsqueeze(1)).sum(dim=-1)  # [batch, N]
        
        # Return combined if requested (after separate aggregation if enabled)
        if return_combined:
            return combined
        
        # Apply post-aggregation nonlinearity if enabled
        if self.post_agg_nonlin == 'rmsnorm':
            combined = self.apply_rmsnorm(combined)
        elif self.post_agg_nonlin == 'sigmoid':
            combined = torch.sigmoid(self.args.post_agg_mult * combined)
        
        # Add residual connection if enabled
        if hasattr(self.args, 'use_residual') and self.args.use_residual:
            combined = combined + h
        
        # Output layer
        out = combined @ (self.args.gamma * self.out_scale * self.w_out)
        return out
    
    def apply_rmsnorm(self, x, eps=1e-6):
        """Apply RMSNorm: x / RMS(x) where RMS = sqrt(mean(x^2))"""
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    
    def apply_nonlin(self, x):
        if self.args.nonlin == 'tanh':
            return torch.tanh(self.args.b * x) / self.args.b
        elif self.args.nonlin == 'relu':
            return F.relu(x)
        elif self.args.nonlin == 'sigmoid':
            return torch.sigmoid(x)
        elif self.args.nonlin == 'gelu':
            return F.gelu(x)
        elif self.args.nonlin in ['identity', 'linear']:
            return x
    
    def apply_expert_nonlin(self, x):
        """Apply expert-specific nonlinearity (defaults to apply_nonlin if not specified)"""
        if not hasattr(self.args, 'expert_nonlin') or self.args.expert_nonlin is None:
            return self.apply_nonlin(x)
        
        nonlin = self.args.expert_nonlin
        if nonlin == 'tanh':
            return torch.tanh(self.args.b * x) / self.args.b
        elif nonlin == 'relu':
            return F.relu(x)
        elif nonlin == 'sigmoid':
            return torch.sigmoid(x)
        elif nonlin == 'gelu':
            return F.gelu(x)
        elif nonlin in ['identity', 'linear']:
            return x
        return x


def main():
    parser = argparse.ArgumentParser()
    # dataset parameters
    parser.add_argument('--P', type=int, default=1000, help='Number of training samples')
    parser.add_argument('--P_val', type=int, default=1000, help='Number of validation samples')
    parser.add_argument('--T', type=int, default=100, help='Number of training iterations')
    parser.add_argument('--early_stop_threshold', type=float, default=None, help='Abort if val_acc < best_val_acc / threshold')
    parser.add_argument('--early_stop_check_step', type=int, default=250, help='Step to check early stopping criterion')
    parser.add_argument('--multipass', action='store_true', help='Reuse same batch each iteration (default: use fresh batch each iteration)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--num_classes', type=int, default=None, help='Number of classes for classification (2 for binary, 10 for CIFAR-10, None=auto based on P)')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'tinyimagenet'], help='Dataset to use')

    # width parameters 
    parser.add_argument('--N', type=int, default=128, help='Hidden layer width')
    parser.add_argument('--M', type=int, default=8, help='Number of experts')
    parser.add_argument('--N_expert', type=int, default=64, help='Fixed expert width')

    # MoE parameters
    parser.add_argument('--nonlin', type=str, default='gelu', choices=['tanh', 'identity', 'linear', 'sigmoid', 'relu', 'gelu'], help='Hidden layer nonlinearity')
    parser.add_argument('--expert_nonlin', type=str, default=None, choices=[None, 'tanh', 'identity', 'linear', 'sigmoid', 'relu', 'gelu'], help='Expert nonlinearity (default: use --nonlin)')
    parser.add_argument('--topk', type=int, default=0, help='Top-k routing: select top k experts (0=soft routing with all experts)')
    parser.add_argument('--topk_mini_batch_size', type=int, default=1024, help='Mini-batch size for memory-efficient topk routing (processes this many samples at once per expert)')
    parser.add_argument('--eval_chunk_size', type=int, default=1024, help='Chunk size for topk evaluation to avoid OOM (processes batches in chunks during eval)')
    parser.add_argument('--router_fn', type=str, default='sigmoid', choices=['softmax', 'sigmoid', 'linear'], help='Router activation function')
    parser.add_argument('--sigmoid_norm', action='store_true', help='Normalize sigmoid routing (else keep raw sigmoid outputs)')
    parser.add_argument('--router_fn_scale_alpha', type=float, default=None, help='Scale sigmoid router weights by (M_base/M)^alpha (default None = use config default)')
    parser.add_argument('--M_base', type=int, default=8, help='Base number of experts for alpha scaling')
    parser.add_argument('--separate_aggregation', action='store_true', help='Separate scaling for h^L_0 (by 1/M^0.5) and Delta h^L (by 1/M)')
    parser.add_argument('--post_agg_nonlin', type=str, default=None, choices=[None, 'rmsnorm', 'sigmoid'], help='Post-aggregation nonlinearity before output layer')
    parser.add_argument('--post_agg_mult', type=float, default=0.001, help='Scalar multiplier applied before sigmoid post-aggregation nonlinearity')
    parser.add_argument('--use_residual', action='store_true', help='Add residual connection from input to output (skip experts and router)')

    # training parameters
    parser.add_argument('--use_cross_entropy', action='store_true', help='Use cross-entropy loss instead of MSE')
    parser.add_argument('--eta', type=float, default=None, help='Base learning rate (default: 0.003125 for bottleneck, 0.000781 otherwise)')
    parser.add_argument('--gamma', type=float, default=1.0, help='Output scale parameter')
    parser.add_argument('--b', type=float, default=0.8, help='Nonlinearity scaling parameter for tanh(b*x)/b')
    parser.add_argument('--optimizer', type=str, default='sgd', choices=['sgd', 'adam'], help='Optimizer to use')
    parser.add_argument('--adam_eps', type=float, default=1e-8, help='Base epsilon for Adam optimizer')
    
    # tunable scalar multipliers
    parser.add_argument('--lr_mult_in', type=float, default=None, help='LR multiplier for w_in (default: 1.0)')
    parser.add_argument('--lr_mult_router', type=float, default=None, help='LR multiplier for w_router (default: 1.0)')
    parser.add_argument('--lr_mult_expert1', type=float, default=None, help='LR multiplier for w_expert1 (default: 1.0)')
    parser.add_argument('--lr_mult_expert2', type=float, default=None, help='LR multiplier for w_expert2 (default: 1.0)')
    parser.add_argument('--lr_mult_out', type=float, default=None, help='LR multiplier for w_out (default: 1.0)')
    parser.add_argument('--init_std_mult', type=float, default=None, help='Init std multiplier for all layers (default: 1.0)')
    
    # layerwise initialization, weight multipliers and learning rates
    parser.add_argument('--router_init', type=str, default=None, choices=['zero', 'mup', 'ntp', 'nonasymp', 'nonasympsqrt', None],
                        help='Router init: zero (Q=0), mup (forward scale=1/N), ntp (forward scale=1/sqrt(N)), nonasymp (std: (fanin / (sqrt(fanout) + sqrt(fanin)))^(-1) * N, fanin=N, fanout=M, scale=1/N), nonasympsqrt (std: (fanin / (sqrt(fanout) + sqrt(fanin)))^(-1) * sqrt(N), scale=1/sqrt(N)), None (use scaling config)')
    parser.add_argument('--last_layer_init', type=str, default=None, choices=['mup', 'ntp', 'zero', None],
                        help='Last layer init: mup (forward scale=1/N), ntp (forward scale=1/sqrt(N)), zero (w_out=0), None (use scaling config)')
    parser.add_argument('--first_expert_layer_init', type=str, default='mup', choices=['mup', 'ntp'],
                        help='First expert layer init: mup (forward scale=1/N), ntp (forward scale=1/sqrt(N))')
    parser.add_argument('--expert_lr_expon', type=float, default=0.0, help='Scale expert learning rate by *=N**(expert_lr_expon)')
    parser.add_argument('--router_lr_expon', type=float, default=0.0, help='Scale router learning rate by *= N**(router_lr_expon)')
    parser.add_argument('--input_lr_zero', action='store_true', help='Set input layer learning rate to 0 (freeze input layer)')
    parser.add_argument('--share_expert_weights_in', action='store_true', help='Share input layer weights across all experts at initialization (all experts start with same w_expert1)')
    parser.add_argument('--share_expert_weights_out', action='store_true', help='Share output layer weights across all experts at initialization (all experts start with same w_expert2)')

    # saving
    parser.add_argument('--results_dir', type=str, default=None, help='Shared results directory for parallel runs')
    parser.add_argument('--minimal_stats', action='store_true', help='Only track train/val losses and accuracies, skip activation stats and other tracking')
    parser.add_argument('--extensive_logging', action='store_true', help='Enable extensive logging including h^agg decomposition and expert gradient decomposition (compute-intensive)')
    
    # scaling configuration
    parser.add_argument('--scaling_config', type=str, default=None, help='Use predefined scaling config (overrides individual init/lr args)')

    # optimal hyperparameters
    parser.add_argument('--use_optimal', action='store_true', help='Load optimal hyperparameters from tuned_multipliers.py. Automatically selects no-shared-expert variant when --share_expert_weights_in/out are not set.')
    parser.add_argument('--routing_mode', type=str, default='soft', choices=['soft', 'topk'], help='Routing mode for loading optimal values (used with --use_optimal)')

    args = parser.parse_args()

    # Apply optimal hyperparameters if requested
    if args.use_optimal:
        if _get_optimal_values is None:
            print("ERROR: --use_optimal requires tuned_multipliers.py to be present")
            return
        if not args.scaling_config:
            print("ERROR: --use_optimal requires --scaling_config to be set")
            return
        shared_experts = args.share_expert_weights_in or args.share_expert_weights_out
        optimal = _get_optimal_values(args.scaling_config, args.routing_mode, shared_experts=shared_experts, last_layer_init=args.last_layer_init)
        if optimal is None:
            print(f"ERROR: No optimal values found for config '{args.scaling_config}' / routing_mode '{args.routing_mode}'")
            return
        # Only override values the user did not explicitly pass (None means "not set")
        if args.eta is None:
            args.eta = optimal['base_lr']
        if args.init_std_mult is None:
            args.init_std_mult = optimal['init_std_mult']
        if args.lr_mult_in is None:
            args.lr_mult_in = optimal['lr_mult_in']
        if args.lr_mult_out is None:
            args.lr_mult_out = optimal['lr_mult_out']
        if args.lr_mult_router is None:
            args.lr_mult_router = optimal['lr_mult_router']
        if args.lr_mult_expert1 is None:
            args.lr_mult_expert1 = optimal['lr_mult_expert1']
        if args.lr_mult_expert2 is None:
            args.lr_mult_expert2 = optimal['lr_mult_expert2']
        variant = "shared experts" if shared_experts else "no weight sharing"
        print(f"\n✓ Loaded optimal hyperparameters for {args.scaling_config} ({args.routing_mode} routing, {variant}):")
        print(f"  Base LR: {args.eta}, init_std_mult: {args.init_std_mult}")
        print(f"  LR multipliers: in={args.lr_mult_in}, out={args.lr_mult_out}, router={args.lr_mult_router}")
        print(f"  Expert LR mults: exp1={args.lr_mult_expert1}, exp2={args.lr_mult_expert2}")
        print(f"  Best accuracy: {optimal['best_acc']:.4f}\n")

    # Set defaults based on scaling config
    is_bottleneck = args.scaling_config and 'bottleneck' in args.scaling_config

    # Auto-set num_classes based on dataset if not specified
    if args.num_classes is None:
        if args.dataset == 'tinyimagenet':
            args.num_classes = 200 if args.P > 50000 else 100
        elif args.dataset == 'cifar10':
            args.num_classes = 10 if args.P > 10000 else 2
        else:
            args.num_classes = 10 if args.P > 10000 else 2

    # Auto-reduce batch sizes for N=2048 with topk in allscaling configs (still hits OOM at 44GB)
    if args.N >= 2048 and args.topk > 0 and args.scaling_config and 'allscaling' in args.scaling_config:
        args.topk_mini_batch_size = 64
        args.eval_chunk_size = 64
        print(f"Auto-reduced batch sizes for N={args.N} with topk={args.topk} in allscaling: topk_mini_batch_size=64, eval_chunk_size=64")

    # Auto-set cross-entropy for TinyImageNet (standard for multi-class classification)
    if args.dataset == 'tinyimagenet' and not args.use_cross_entropy:
        args.use_cross_entropy = True
        print("Auto-enabled cross-entropy loss for TinyImageNet (standard for 200-class classification)")

    # Apply scaling config dimension adjustments
    if args.scaling_config:
        scaling_config = get_config(args.scaling_config)
        args.M, args.N_expert = scaling_config.adjust_dimensions(args.N, args.M, args.N_expert)

        # Auto-set optimizer to 'adam' if config name contains 'adam'
        if 'adam' in args.scaling_config.lower() and args.optimizer == 'sgd':
            args.optimizer = 'adam'
            print(f"Auto-set optimizer to 'adam' based on scaling_config name: {args.scaling_config}")

        # Use config's default alpha if not explicitly set
        if args.router_fn_scale_alpha is None:
            args.router_fn_scale_alpha = scaling_config.get_default_alpha()

        print(f"Scaling config '{args.scaling_config}': N={args.N}, M={args.M}, N_expert={args.N_expert}, alpha={args.router_fn_scale_alpha}")
    else:
        # No scaling config: default alpha to 0.0
        if args.router_fn_scale_alpha is None:
            args.router_fn_scale_alpha = 0.0

    # Force alpha=0 when using separate aggregation (applies to both with and without scaling config)
    if args.separate_aggregation and args.router_fn_scale_alpha != 0.0:
        print(f"Forcing router_fn_scale_alpha=0.0 for separate_aggregation (was {args.router_fn_scale_alpha})")
        args.router_fn_scale_alpha = 0.0

    if args.eta is None:
        args.eta = 0.003125

    # Set default last_layer_init to 'zero' for all MuP variants
    if args.last_layer_init is None and args.scaling_config and 'mup' in args.scaling_config.lower():
        args.last_layer_init = 'zero'
        print(f"Auto-set last_layer_init='zero' for MuP variant")

    # Inform user about shared expert weights
    if args.share_expert_weights_in or args.share_expert_weights_out:
        shared_layers = []
        if args.share_expert_weights_in:
            shared_layers.append("w_expert1 (input)")
        if args.share_expert_weights_out:
            shared_layers.append("w_expert2 (output)")
        print(f"Expert weight sharing enabled at initialization: {', '.join(shared_layers)}")

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Early exit check: skip if output file already exists
    if args.results_dir:
        config_str = f"N{args.N}_M{args.M}_Ne{args.N_expert}_P{args.P}_T{args.T}_eta{args.eta}_gamma{args.gamma}_seed{args.seed}"
        if args.scaling_config:
            config_str += f"_{args.scaling_config}"
        routing_str = f"k{args.topk}" if args.topk > 0 else "soft"
        config_str += f"_{routing_str}"
        if args.init_std_mult is not None:
            config_str += f"_istd{args.init_std_mult}"
        if args.lr_mult_in is not None:
            config_str += f"_lrin{args.lr_mult_in}"
        if args.lr_mult_out is not None:
            config_str += f"_lrout{args.lr_mult_out}"
        if args.lr_mult_router is not None:
            config_str += f"_lrrouter{args.lr_mult_router}"
        if args.lr_mult_expert1 is not None:
            config_str += f"_lrexp1_{args.lr_mult_expert1}"
        if args.lr_mult_expert2 is not None:
            config_str += f"_lrexp2_{args.lr_mult_expert2}"
        if args.share_expert_weights_in:
            config_str += "_sharedexp1"
        if args.share_expert_weights_out:
            config_str += "_sharedexp2"

        config_dir = Path(args.results_dir) / (args.scaling_config if args.scaling_config else 'default')
        stats_dir = config_dir / 'stats'
        output_file = stats_dir / f"nn_{config_str}.npz"

        if output_file.exists():
            print(f"Output file already exists, skipping: {output_file}")
            return

    # Load dataset
    if args.dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            transforms.Lambda(lambda x: x.view(-1))
        ])
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        val_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        input_dim = 3072
    elif args.dataset == 'tinyimagenet':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            transforms.Lambda(lambda x: x.view(-1))
        ])
        train_dataset = TinyImageNet(root='./data', train=True, download=True, transform=transform)
        val_dataset = TinyImageNet(root='./data', train=False, download=True, transform=transform)
        input_dim = 12288  # 64x64x3
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Filter to only classes 0 to num_classes-1
    train_mask = torch.tensor([train_dataset[i][1] < args.num_classes for i in range(len(train_dataset))])
    val_mask = torch.tensor([val_dataset[i][1] < args.num_classes for i in range(len(val_dataset))])
    train_filtered_indices = torch.where(train_mask)[0]
    val_filtered_indices = torch.where(val_mask)[0]
    
    # Sample subsets with balanced classes
    if args.num_classes == 2:
        # Binary: sample equal from each class
        train_class_0 = [i.item() for i in train_filtered_indices if train_dataset[i][1] == 0]
        train_class_1 = [i.item() for i in train_filtered_indices if train_dataset[i][1] == 1]
        samples_per_class = args.P // 2
        train_class_0_sample = torch.tensor(train_class_0)[torch.randperm(len(train_class_0))[:samples_per_class]]
        train_class_1_sample = torch.tensor(train_class_1)[torch.randperm(len(train_class_1))[:samples_per_class]]
        train_indices = torch.cat([train_class_0_sample, train_class_1_sample])[torch.randperm(args.P)]
        
        val_class_0 = [i.item() for i in val_filtered_indices if val_dataset[i][1] == 0]
        val_class_1 = [i.item() for i in val_filtered_indices if val_dataset[i][1] == 1]
        val_samples_per_class = args.P_val // 2
        val_class_0_sample = torch.tensor(val_class_0)[torch.randperm(len(val_class_0))[:val_samples_per_class]]
        val_class_1_sample = torch.tensor(val_class_1)[torch.randperm(len(val_class_1))[:val_samples_per_class]]
        val_indices = torch.cat([val_class_0_sample, val_class_1_sample])[torch.randperm(args.P_val)]
    else:
        train_indices = train_filtered_indices[torch.randperm(len(train_filtered_indices))[:args.P]]
        val_indices = val_filtered_indices[torch.randperm(len(val_filtered_indices))[:args.P_val]]
    
    X_train = torch.stack([train_dataset[i][0] for i in train_indices]).to(device)
    y_train = torch.tensor([train_dataset[i][1] for i in train_indices]).to(device)
    X_val = torch.stack([val_dataset[i][0] for i in val_indices]).to(device)
    y_val = torch.tensor([val_dataset[i][1] for i in val_indices]).to(device)
    
    # Prepare labels
    if not args.use_cross_entropy:
        if args.num_classes == 2:
            # Binary MSE: scalar labels {-1, 1}
            y_train_targets = y_train.float() * 2 - 1
            y_val_targets = y_val.float() * 2 - 1
        else:
            # Multi-class MSE: one-hot vector labels {0, 1}
            y_train_targets = F.one_hot(y_train, args.num_classes).float()
            y_val_targets = F.one_hot(y_val, args.num_classes).float()
    else:
        y_train_targets = y_train
        y_val_targets = y_val
    
    # Initialize model with scaling config
    scaling_config = get_config(args.scaling_config) if args.scaling_config else None
    model = MoEModel(args, scaling_config, input_dim=input_dim).to(device)
    
    # Setup learning rates
    if scaling_config:
        base_lr = args.eta * args.gamma**2
        lr_in = 0.0 if args.input_lr_zero else scaling_config.get_lr_scale('w_in', args.N, args.M, args.N_expert, base_lr, input_dim)
        lr_router = scaling_config.get_lr_scale('w_router', args.N, args.M, args.N_expert, base_lr, args.N)
        lr_expert1 = scaling_config.get_lr_scale('w_expert1', args.N, args.M, args.N_expert, base_lr, args.N)
        lr_expert2 = scaling_config.get_lr_scale('w_expert2', args.N, args.M, args.N_expert, base_lr, args.N_expert)
        lr_out = scaling_config.get_lr_scale('w_out', args.N, args.M, args.N_expert, base_lr, args.N)
        
        # Apply optional LR multipliers
        if args.lr_mult_in is not None:
            lr_in *= args.lr_mult_in
        if args.lr_mult_router is not None:
            lr_router *= args.lr_mult_router
        if args.lr_mult_expert1 is not None:
            lr_expert1 *= args.lr_mult_expert1
        if args.lr_mult_expert2 is not None:
            lr_expert2 *= args.lr_mult_expert2
        if args.lr_mult_out is not None:
            lr_out *= args.lr_mult_out
        
        if args.optimizer == 'sgd':
            param_groups = [
                {'params': [model.w_in], 'lr': lr_in},
                {'params': [model.w_router], 'lr': lr_router},
                {'params': [model.w_expert1], 'lr': lr_expert1},
                {'params': [model.w_expert2], 'lr': lr_expert2},
                {'params': [model.w_out], 'lr': lr_out}
            ]
        elif args.optimizer == 'adam':
            # Get epsilon scales for Adam
            eps_in = args.adam_eps * scaling_config.get_adam_eps_scale('w_in', args.N, args.M, args.N_expert)
            eps_router = args.adam_eps * scaling_config.get_adam_eps_scale('w_router', args.N, args.M, args.N_expert)
            eps_expert1 = args.adam_eps * scaling_config.get_adam_eps_scale('w_expert1', args.N, args.M, args.N_expert)
            eps_expert2 = args.adam_eps * scaling_config.get_adam_eps_scale('w_expert2', args.N, args.M, args.N_expert)
            eps_out = args.adam_eps * scaling_config.get_adam_eps_scale('w_out', args.N, args.M, args.N_expert)
        
            param_groups = [
                {'params': [model.w_in], 'lr': lr_in, 'eps': eps_in},
                {'params': [model.w_router], 'lr': lr_router, 'eps': eps_router},
                {'params': [model.w_expert1], 'lr': lr_expert1, 'eps': eps_expert1},
                {'params': [model.w_expert2], 'lr': lr_expert2, 'eps': eps_expert2},
                {'params': [model.w_out], 'lr': lr_out, 'eps': eps_out}
            ]
    else:
        # Legacy: use args-based LR scaling
        base_lr = args.eta * args.gamma**2 * args.N if args.last_layer_init in ['mup', 'zero'] else args.eta * args.gamma**2
        lr_in = 0.0 if args.input_lr_zero else base_lr
        router_lr = base_lr * (args.N ** args.router_lr_expon)
        expert_lr = base_lr * (args.N ** args.expert_lr_expon)
        
        param_groups = [
            {'params': [model.w_in], 'lr': lr_in},
            {'params': [model.w_out], 'lr': base_lr},
            {'params': [model.w_router], 'lr': router_lr},
            {'params': [model.w_expert1, model.w_expert2], 'lr': expert_lr}
        ]
    
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(param_groups)
    else:
        optimizer = torch.optim.SGD(param_groups)
    
    # Save initial model for propagating updates (before any training)
    model_init = MoEModel(args, scaling_config, input_dim=input_dim).to(device)
    model_init.load_state_dict(model.state_dict())
    model_init.eval()
    # Ensure model_init is not trainable
    for param in model_init.parameters():
        param.requires_grad = False
    
    # Training loop
    results = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
        'train_top5_acc': [], 'val_top5_acc': [],
        'psi_norm': [], 'router_max': [], 'router_min': [],
        'max_concentration': [], 'min_concentration': [], 'entropy': [],
        'sparsity': [], 'routing_consistency': [], 'expert_l2_diffs': [],
        'layer_stats_by_class': [], 'effective_updates': [], 'total_effective_updates_per_layer': [], 'propagating_updates': [],
        'grad_norms': [], 'h_L_rms_diff': [], 'h_L_rms': [], 'expert_hists': [],
        'h_L_hists': [],
        'activation_rms': [],  # Track RMS of all activations
        'output_grad_rms': [],  # Track dL/d(output)
        'weight_rms_norms': [],  # Track RMS norm of each weight matrix
        'hagg_decomposition': [],  # Track h^agg decomposition (4 terms: base, propagating, effective, cross)
        'expert_grad_h1_decomposition': []  # Track expert gradient to h1 decomposition (4 terms)
    }
    model_prev = None
    h_L_init = None
    # Use smaller batch for coordinate checks to avoid OOM
    # Note: compute_stats uses sequential expert computation for memory efficiency
    X_coord_check = X_train[:min(1000, args.P)].clone()
    prev_stats = None
    
    # Early stopping setup - will create timestep-specific files later
    if args.early_stop_threshold:
        early_stop_dir = Path(args.results_dir)
        early_stop_dir.mkdir(parents=True, exist_ok=True)

    # Batched evaluation function to avoid OOM on large datasets
    def batched_eval(model, X, y, model_init=None, batch_size=1000):
        """Evaluate model on dataset in batches to avoid OOM"""
        model.eval()
        n_samples = X.shape[0]
        all_logits = []

        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            X_batch = X[i:end_idx]

            with torch.no_grad():
                logits_batch = model(X_batch, model_init=model_init)
                all_logits.append(logits_batch)

        # Concatenate all batches
        return torch.cat(all_logits, dim=0)

    # Determine evaluation batch size based on dataset size and width
    # For large N (>=2048), use smaller batches to avoid OOM
    if args.N >= 2048:
        eval_batch_size = 500  # Very conservative for N>=2048
    elif args.N >= 1024:
        eval_batch_size = 1000
    elif args.N >= 512:
        eval_batch_size = 2000
    else:
        eval_batch_size = args.P  # Small N, load all at once

    print(f"Using evaluation batch size: {eval_batch_size}")

    for t in range(args.T):
        # Evaluate before training step
        with torch.no_grad():
            model.eval()
            # For separate_aggregation: pass model_init only after first step
            model_init_for_eval = model_init if (args.separate_aggregation and t > 0) else None

            # Use batched evaluation to avoid OOM on large datasets
            train_logits = batched_eval(model, X_train, y_train, model_init=model_init_for_eval, batch_size=eval_batch_size)
            val_logits = batched_eval(model, X_val, y_val, model_init=model_init_for_eval, batch_size=eval_batch_size)
            
            if args.use_cross_entropy:
                train_loss = F.cross_entropy(train_logits, y_train).item()
                val_loss = F.cross_entropy(val_logits, y_val).item()
                train_acc = (train_logits.argmax(1) == y_train).float().mean().item()
                val_acc = (val_logits.argmax(1) == y_val).float().mean().item()

                # Compute top-5 accuracy for multi-class (>5 classes)
                if args.num_classes > 5:
                    _, train_top5_pred = train_logits.topk(5, dim=1)
                    train_top5_acc = (train_top5_pred == y_train.unsqueeze(1)).any(dim=1).float().mean().item()
                    _, val_top5_pred = val_logits.topk(5, dim=1)
                    val_top5_acc = (val_top5_pred == y_val.unsqueeze(1)).any(dim=1).float().mean().item()
                else:
                    train_top5_acc = None
                    val_top5_acc = None
            else:
                if args.num_classes == 2:
                    train_loss = F.mse_loss(train_logits.squeeze(-1), y_train_targets).item()
                    val_loss = F.mse_loss(val_logits.squeeze(-1), y_val_targets).item()
                    train_acc = ((train_logits.squeeze() > 0) == (y_train == 1)).float().mean().item()
                    val_acc = ((val_logits.squeeze() > 0) == (y_val == 1)).float().mean().item()
                    train_top5_acc = None
                    val_top5_acc = None
                else:
                    train_loss = F.mse_loss(train_logits, y_train_targets).item()
                    val_loss = F.mse_loss(val_logits, y_val_targets).item()
                    train_acc = (train_logits.argmax(1) == y_train).float().mean().item()
                    val_acc = (val_logits.argmax(1) == y_val).float().mean().item()

                    # Compute top-5 accuracy for multi-class (>5 classes)
                    if args.num_classes > 5:
                        _, train_top5_pred = train_logits.topk(5, dim=1)
                        train_top5_acc = (train_top5_pred == y_train.unsqueeze(1)).any(dim=1).float().mean().item()
                        _, val_top5_pred = val_logits.topk(5, dim=1)
                        val_top5_acc = (val_top5_pred == y_val.unsqueeze(1)).any(dim=1).float().mean().item()
                    else:
                        train_top5_acc = None
                        val_top5_acc = None

            results['train_loss'].append(train_loss)
            results['train_acc'].append(train_acc)
            results['val_loss'].append(val_loss)
            results['val_acc'].append(val_acc)

            # Track top-5 accuracy if available
            if train_top5_acc is not None:
                results['train_top5_acc'].append(train_top5_acc)
                results['val_top5_acc'].append(val_top5_acc)
            
            # Early stopping on NaN
            if np.isnan(train_loss) or np.isnan(val_loss):
                print(f"\n⚠ NaN detected at iteration {t}. Aborting training.")
                break
            
            # Early stopping check (accuracy-based only, never use loss)
            if args.early_stop_threshold and t == args.early_stop_check_step:
                # Prefer val_top5 if available (for multi-class problems), otherwise use val_acc
                use_val_top5 = len(results['val_top5_acc']) > 0 and results['val_top5_acc'][-1] is not None

                if use_val_top5:
                    # Use val_top5 accuracy (higher is better)
                    # Compute mean val_top5 over last 50 steps
                    if len(results['val_top5_acc']) >= 50:
                        mean_acc = np.mean([x for x in results['val_top5_acc'][-50:] if x is not None])
                    else:
                        mean_acc = val_top5_acc
                    metric_name = "val_top5"
                else:
                    # Fall back to val_acc
                    if len(results['val_acc']) >= 50:
                        mean_acc = np.mean(results['val_acc'][-50:])
                    else:
                        mean_acc = val_acc
                    metric_name = "val_acc"

                # Create timestep-specific best accuracy file
                best_acc_filename = f"best_acc_{args.scaling_config}_N{args.N}_{args.dataset}_t{t}.txt"
                best_acc_path = early_stop_dir / best_acc_filename

                # Read current best accuracy at this timestep
                if best_acc_path.exists():
                    with open(best_acc_path, 'r') as f:
                        best_acc = float(f.read().strip())
                else:
                    best_acc = 0.0  # Start from 0 for accuracy

                # Check if current run is too bad (accuracy should be >= best / threshold)
                # Threshold of 2 means: keep if acc >= (1/2) * best_acc = 0.5 * best_acc
                min_acceptable = best_acc / args.early_stop_threshold
                if mean_acc < min_acceptable and best_acc > 0:
                    print(f"\n⚠ Early stop at t={t}: {metric_name} {mean_acc:.4f} < {1/args.early_stop_threshold:.2f}x best at t={t} {best_acc:.4f}")
                    break

                # Update best accuracy at this timestep if improved
                if mean_acc > best_acc:
                    with open(best_acc_path, 'w') as f:
                        f.write(str(mean_acc))
            
            # Compute additional statistics
            if not args.minimal_stats:
                stats = compute_stats(model, X_train, y_train, args, t, prev_stats, model_init)
                results['psi_norm'].append(stats['psi_norm'])
                results['router_max'].append(stats['router_max'])
                results['router_min'].append(stats['router_min'])
                results['max_concentration'].append(stats['max_concentration'])
                results['min_concentration'].append(stats['min_concentration'])
                results['entropy'].append(stats['entropy'])
                results['sparsity'].append(stats['sparsity'])
                results['routing_consistency'].append(stats['routing_consistency'])
                results['expert_l2_diffs'].append(stats['expert_l2_diffs'])
                results['layer_stats_by_class'].append(stats['layer_stats_by_class'])
                results['expert_hists'].append(stats['expert_hists'])
                results['h_L_hists'].append(stats['h_L_hist'])
                
                # Effective updates (on fixed batch)
                if model_prev is not None:
                    eff_updates = compute_effective_updates(model, model_prev, X_coord_check, args, model_init)
                    results['effective_updates'].append(eff_updates)
                else:
                    results['effective_updates'].append({'raw': {}, 'normalized': {}})
                
                
                # Propagating updates (on fixed batch with initial weights)
                if t == 0:
                    # Compute and save h^L_0 on first iteration
                    with torch.no_grad():
                        h_L_init = compute_h_L(model_init, X_coord_check, args, None)
                    results['propagating_updates'].append({'raw': {}, 'normalized': {}})
                else:
                    prop_updates = compute_propagating_updates(model_init, model, X_coord_check, args)
                    results['propagating_updates'].append(prop_updates)
                

                # Total effective updates per layer (on fixed batch with initial weights)
                if t == 0:
                    results['total_effective_updates_per_layer'].append({'raw': {}, 'normalized': {}})
                else:
                    # Total effective updates per layer
                    total_effective_updates_per_layer = compute_total_effective_updates_per_layer(model_init, model, X_coord_check, args)
                    results['total_effective_updates_per_layer'].append(total_effective_updates_per_layer)
                

                # Track RMS(h^L_t - h^L_0) and RMS(h^L_t)
                if h_L_init is not None:
                    h_L_rms_diff = compute_h_L_rms_diff(model, X_coord_check, h_L_init, args, model_init)
                    results['h_L_rms_diff'].append(h_L_rms_diff)
                else:
                    results['h_L_rms_diff'].append(0.0)
                
                h_L_rms = compute_h_L_rms(model, X_coord_check, args, model_init)
                results['h_L_rms'].append(h_L_rms)
                
                # Track RMS of all activations
                activation_rms = compute_activation_rms(model, X_coord_check, args, model_init)
                results['activation_rms'].append(activation_rms)
                
                # Track dL/d(output)
                output_grad_rms = compute_output_gradient(model, X_coord_check, y_train[:min(1000, args.P)], args, model_init)
                results['output_grad_rms'].append(output_grad_rms)

                # Track weight RMS norms
                weight_rms = compute_weight_rms_norms(model)
                results['weight_rms_norms'].append(weight_rms)

                # Extensive logging: decomposition analyses (compute-intensive)
                if args.extensive_logging:
                    if t == 0:
                        # No decomposition at t=0 (no change from init yet)
                        results['hagg_decomposition'].append({})
                        results['expert_grad_h1_decomposition'].append({})
                    else:
                        # h^agg decomposition: 4-term decomposition of aggregated expert output
                        hagg_decomp = compute_hagg_decomposition(model_init, model, X_coord_check, args)
                        results['hagg_decomposition'].append(hagg_decomp)

                        # Expert gradient to h^1 decomposition: 4-term decomposition of expert gradients
                        expert_grad_decomp = compute_expert_grad_h1_decomposition(
                            model_init, model, X_coord_check, y_train[:min(1000, args.P)], args
                        )
                        results['expert_grad_h1_decomposition'].append(expert_grad_decomp)

            if t % 10 == 0:
                if not args.minimal_stats:
                    if args.num_classes > 5:
                        print(f"Iter {t}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, train_top5={train_top5_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_top5={val_top5_acc:.4f}, max_conc={stats['max_concentration']:.3f}, min_conc={stats['min_concentration']:.3f}")
                    else:
                        print(f"Iter {t}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, max_conc={stats['max_concentration']:.3f}, min_conc={stats['min_concentration']:.3f}")
                else:
                    if args.num_classes > 5:
                        print(f"Iter {t}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, train_top5={train_top5_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_top5={val_top5_acc:.4f}")
                    else:
                        print(f"Iter {t}: train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        
        # Clone model_prev AFTER evaluation but BEFORE training step
        if not args.minimal_stats:
            model_prev = MoEModel(args, scaling_config, input_dim=input_dim).to(device)
            model_prev.load_state_dict(model.state_dict())
            model_prev.eval()
            for param in model_prev.parameters():
                param.requires_grad = False
        
        model.train()
        
        # Get batch
        if not args.multipass:
            batch_size = args.P // args.T
            start_idx = t * batch_size
            end_idx = min(start_idx + batch_size, args.P)
            X_batch = X_train[start_idx:end_idx]
            y_batch = y_train_targets[start_idx:end_idx] if not args.use_cross_entropy else y_train[start_idx:end_idx]
        else:
            X_batch = X_train
            y_batch = y_train_targets if not args.use_cross_entropy else y_train
        
        # Forward
        optimizer.zero_grad()
        # For separate_aggregation: pass model_init only after first step
        model_init_for_forward = model_init if (args.separate_aggregation and t > 0) else None
        logits = model(X_batch, model_init=model_init_for_forward)
        
        # Loss
        if args.use_cross_entropy:
            loss = F.cross_entropy(logits, y_batch)
        else:
            if args.num_classes == 2:
                loss = F.mse_loss(logits.squeeze(-1), y_batch)
            else:
                loss = F.mse_loss(logits, y_batch)
        
        # Backward
        loss.backward()
        
        # Track gradient norms before optimizer step
        if not args.minimal_stats:
            grad_norms = compute_gradient_norms(model)
            results['grad_norms'].append(grad_norms)
        
        optimizer.step()
        
        # Clone model for next iteration (after step)
        if not args.minimal_stats:
            prev_stats = stats if 'stats' in locals() else None
        
        # Clear cache periodically to avoid fragmentation
        if t % 10 == 0:
            torch.cuda.empty_cache()
    
    # Save results
    if args.results_dir:
        Path(args.results_dir).mkdir(parents=True, exist_ok=True)
        # Create config subdirectory
        config_dir = Path(args.results_dir) / (args.scaling_config if args.scaling_config else 'default')
        stats_dir = config_dir / 'stats'
        stats_dir.mkdir(parents=True, exist_ok=True)
        
        config_str = f"N{args.N}_M{args.M}_Ne{args.N_expert}_P{args.P}_T{args.T}_eta{args.eta}_gamma{args.gamma}_seed{args.seed}"
        if args.scaling_config:
            config_str += f"_{args.scaling_config}"
        # Add topk to filename
        routing_str = f"k{args.topk}" if args.topk > 0 else "soft"
        config_str += f"_{routing_str}"
        # Add router_init to filename (critical for distinguishing initialization ablations)
        if args.router_init is not None:
            config_str += f"_rinit{args.router_init}"
        # Add layer-wise LR and init multipliers (for 6D sweeps) - always include if set (even if 1.0)
        # This is critical to avoid overwriting files in hyperparameter sweeps
        if args.init_std_mult is not None:
            config_str += f"_istd{args.init_std_mult}"
        if args.lr_mult_in is not None:
            config_str += f"_lrin{args.lr_mult_in}"
        if args.lr_mult_out is not None:
            config_str += f"_lrout{args.lr_mult_out}"
        if args.lr_mult_router is not None:
            config_str += f"_lrrouter{args.lr_mult_router}"
        if args.lr_mult_expert1 is not None:
            config_str += f"_lrexp1_{args.lr_mult_expert1}"
        if args.lr_mult_expert2 is not None:
            config_str += f"_lrexp2_{args.lr_mult_expert2}"
        if args.share_expert_weights_in:
            config_str += "_sharedexp1"
        if args.share_expert_weights_out:
            config_str += "_sharedexp2"
        
        # Convert lists to numpy arrays for npz
        results_np = {}
        for k, v in results.items():
            if k in ['expert_hists', 'h_L_hists']:
                # Store as object array to handle inhomogeneous shapes
                results_np[k] = np.array(v, dtype=object)
            elif isinstance(v, list) and len(v) > 0:
                # Check if list contains tensors and convert to CPU
                if isinstance(v[0], torch.Tensor):
                    results_np[k] = np.array([x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x for x in v])
                else:
                    results_np[k] = np.array(v)
            elif isinstance(v, list):
                results_np[k] = np.array(v)
            else:
                results_np[k] = v
        metadata = {'args': vars(args)}
        np.savez(stats_dir / f"nn_{config_str}.npz", **results_np, metadata=metadata)
        
        # Save human-readable config
        config_file = stats_dir / f"config_{config_str}.json"
        with open(config_file, 'w') as f:
            json.dump(vars(args), f, indent=2, default=str)
    
    if args.num_classes > 5 and len(results['train_top5_acc']) > 0:
        print(f"\nFinal: train_acc={results['train_acc'][-1]:.4f}, train_top5={results['train_top5_acc'][-1]:.4f}, val_acc={results['val_acc'][-1]:.4f}, val_top5={results['val_top5_acc'][-1]:.4f}")
    else:
        print(f"\nFinal: train_acc={results['train_acc'][-1]:.4f}, val_acc={results['val_acc'][-1]:.4f}")


if __name__ == '__main__':
    main()
