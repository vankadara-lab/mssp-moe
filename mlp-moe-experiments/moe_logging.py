import torch
import torch.nn.functional as F
import numpy as np

def compute_expert_outputs_batched(h, w_expert1, w_expert2, scale1, scale2, apply_nonlin_fn, model_init=None, args=None):
    """Compute expert outputs using simple loop with memory-efficient batching.

    Args:
        h: [batch, N] input activations
        w_expert1: [M, N, N_expert] first expert weights
        w_expert2: [M, N_expert, N] second expert weights
        scale1, scale2: scaling factors
        apply_nonlin_fn: nonlinearity function
        model_init: Initial model for separate aggregation (optional)
        args: Arguments object for separate_aggregation flag and eval_chunk_size (optional)

    Returns:
        expert_outputs: [M, batch, N]
    """
    M = w_expert1.shape[0]
    total_batch = h.shape[0]
    N = h.shape[1]

    # Use eval_chunk_size from args if available, otherwise default to 1024
    batch_size = args.eval_chunk_size if args is not None and hasattr(args, 'eval_chunk_size') else 1024

    # Process in smaller batches to avoid OOM
    all_outputs = []
    for i in range(0, total_batch, batch_size):
        end_idx = min(i + batch_size, total_batch)
        h_batch = h[i:end_idx]

        outs = []
        for m in range(M):
            e = apply_nonlin_fn(h_batch @ (scale1 * w_expert1[m]))

            # Apply separate aggregation: W_0 x_t / sqrt(M) + (W_t - W_0) x_t / M
            if args is not None and hasattr(args, 'separate_aggregation') and args.separate_aggregation:
                if model_init is not None:
                    # After first iteration: separate W_0 and delta W
                    w2_init = model_init.w_expert2[m]
                    w2_update = w_expert2[m] - w2_init
                    expert_out = (e @ (scale2 * w2_init)) / np.sqrt(M) + (e @ (scale2 * w2_update)) / M
                else:
                    # First iteration: just scale by 1/sqrt(M)
                    expert_out = (e @ (scale2 * w_expert2[m])) / np.sqrt(M)
            else:
                # No separate aggregation
                expert_out = e @ (scale2 * w_expert2[m])

            outs.append(expert_out)

        # Stack for this batch: [M, batch_chunk, N]
        batch_outputs = torch.stack(outs, dim=0)
        all_outputs.append(batch_outputs)

    # Concatenate along batch dimension: [M, total_batch, N]
    return torch.cat(all_outputs, dim=1)


def _compute_router_weights(router_logits, M, args):
    """Compute router weights from pre-activation logits (top-k or soft routing)."""
    M_base = args.M_base if hasattr(args, 'M_base') else 8
    if args.topk > 0:
        topk_vals, topk_idx = torch.topk(router_logits, args.topk, dim=-1)
        if args.router_fn == 'softmax':
            router_weights_topk = F.softmax(topk_vals, dim=-1)
        elif args.router_fn == 'sigmoid':
            router_weights_topk = torch.sigmoid(topk_vals) / (M / M_base) ** args.router_fn_scale_alpha
            if args.sigmoid_norm:
                router_weights_topk = router_weights_topk / (router_weights_topk.sum(dim=-1, keepdim=True) + 1e-10)
        elif args.router_fn == 'linear':
            router_weights_topk = topk_vals / (M / M_base) ** args.router_fn_scale_alpha
        router_weights = torch.zeros_like(router_logits)
        router_weights.scatter_(-1, topk_idx, router_weights_topk)
    else:
        if args.router_fn == 'softmax':
            router_weights = F.softmax(router_logits, dim=-1)
        elif args.router_fn == 'sigmoid':
            router_weights = torch.sigmoid(router_logits) / (M / M_base) ** args.router_fn_scale_alpha
            if args.sigmoid_norm:
                router_weights = router_weights / (router_weights.sum(dim=-1, keepdim=True) + 1e-10)
        elif args.router_fn == 'linear':
            router_weights = router_logits / (M / M_base) ** args.router_fn_scale_alpha
    return router_weights


def compute_stats(model, X, y, args, t, prev_stats=None, model_init=None):
    """Compute training statistics logged at each checkpoint.

    Returns a dict with router concentration (Gini, entropy), per-expert activation
    norms, L2 distances between expert weights, and expert selection histograms.
    """
    stats = {}
    
    with torch.no_grad():
        # Forward pass with intermediate activations - reuse for all computations
        h = X @ (model.input_scale * model.w_in)
        h1 = model.apply_nonlin(h)
        
        router_logits = h1 @ (model.router_scale * model.w_router)
        
        router_weights = _compute_router_weights(router_logits, model.M, args)

        # Router statistics
        stats['router_logits'] = router_logits.detach().cpu().numpy()
        stats['router_weights'] = router_weights.detach().cpu().numpy()
        stats['psi_norm'] = float(torch.sqrt(torch.mean(router_logits**2)).cpu())  # RMS norm
        
        # Router concentration
        M_active = args.topk if args.topk > 0 else model.M
        if args.topk > 0:
            # For topk, only consider the selected experts
            max_vals = []
            min_vals = []
            for i in range(router_weights.shape[0]):
                # Get top-k values for this sample
                topk_weights = torch.topk(router_weights[i], args.topk)[0]
                if topk_weights.numel() > 0:
                    max_vals.append(topk_weights.max())
                    min_vals.append(topk_weights.min())
            if len(max_vals) > 0:
                max_val = float(torch.stack(max_vals).mean().cpu())
                min_val = float(torch.stack(min_vals).mean().cpu())
            elif args.router_fn == 'sigmoid':
                max_val = 0.0
                min_val = 0.0
        elif args.router_fn == 'sigmoid':
            max_val = float(router_weights.max(dim=-1)[0].mean().cpu())
            min_val = float(router_weights.min(dim=-1)[0].mean().cpu())
        else:
            max_val = float(router_weights.max(dim=-1)[0].mean().cpu())
            min_val = float(router_weights.min(dim=-1)[0].mean().cpu())
        stats['router_max'] = max_val
        stats['router_min'] = min_val
        # For concentration, normalize weights to sum to 1
        # IMPORTANT: For top-k routing, only consider active experts (not zeros)
        if args.topk > 0:
            # Compute concentration only over active (non-zero) experts
            max_vals_norm = []
            min_vals_norm = []
            for i in range(router_weights.shape[0]):
                # Get top-k weights for this sample (already normalized to sum to 1)
                topk_weights = torch.topk(router_weights[i], args.topk)[0]
                if topk_weights.numel() > 0 and topk_weights.sum() > 1e-10:
                    topk_weights_norm = topk_weights / (topk_weights.sum() + 1e-10)
                    max_vals_norm.append(topk_weights_norm.max())
                    min_vals_norm.append(topk_weights_norm.min())
            if len(max_vals_norm) > 0:
                max_val_norm = float(torch.stack(max_vals_norm).mean().cpu())
                min_val_norm = float(torch.stack(min_vals_norm).mean().cpu())
            else:
                max_val_norm = 1.0 / M_active
                min_val_norm = 1.0 / M_active
        else:
            # Soft routing: use all experts
            router_weights_norm = router_weights / (router_weights.sum(dim=-1, keepdim=True) + 1e-10)
            max_val_norm = float(router_weights_norm.max(dim=-1)[0].mean().cpu())
            min_val_norm = float(router_weights_norm.min(dim=-1)[0].mean().cpu())
        stats['max_concentration'] = 0.5 + 0.5 * (M_active * max_val_norm - 1) / (M_active - 1) if M_active > 1 else 1.0
        stats['min_concentration'] = M_active * min_val_norm / 2
        
        # Expert activations - skip for very large N to avoid OOM
        # For N >= 2048, expert_outputs tensor [M, P, N] is too large to fit in GPU memory
        skip_expert_stats = (args.N >= 2048)
        if skip_expert_stats and t == 0:
            print(f"  Skipping memory-intensive expert stats for N={args.N} (>= 2048) to avoid OOM")

        if not skip_expert_stats:
            scale1 = model.expert1_scale
            scale2 = model.expert2_scale
            expert_outputs = compute_expert_outputs_batched(
                h1, model.w_expert1, model.w_expert2, scale1, scale2, model.apply_expert_nonlin,
                model_init=model_init, args=args
            )

            # Expert activation histograms (50 bins)
            expert_hists = []
            for i in range(model.M):
                expert_flat = expert_outputs[i].flatten().cpu()
                # Skip histogram if contains NaN or Inf
                if torch.isfinite(expert_flat).all():
                    hist = torch.histogram(expert_flat, bins=50)
                    expert_hists.append((hist.hist.numpy(), hist.bin_edges.numpy()))
                elif args.router_fn == 'sigmoid':
                    # Store empty histogram for invalid data
                    expert_hists.append((np.zeros(50), np.zeros(51)))
            stats['expert_hists'] = expert_hists

            # h^L (combined) histogram
            combined = (expert_outputs.permute(1, 2, 0) * router_weights.unsqueeze(1)).sum(dim=-1)  # [batch, N]
        else:
            # For large N, skip expert output stats and compute combined differently
            stats['expert_hists'] = []
            # Compute combined output without storing full expert_outputs
            combined = torch.zeros(h1.shape[0], h1.shape[1], device=h1.device)
            scale1 = model.expert1_scale
            scale2 = model.expert2_scale
            for m in range(model.M):
                e = model.apply_expert_nonlin(h1 @ (scale1 * model.w_expert1[m]))
                expert_out = e @ (scale2 * model.w_expert2[m])
                combined += expert_out * router_weights[:, m:m+1]
        combined_flat = combined.flatten().cpu()
        if torch.isfinite(combined_flat).all():
            hist = torch.histogram(combined_flat, bins=50)
            stats['h_L_hist'] = (hist.hist.numpy(), hist.bin_edges.numpy())
        else:
            stats['h_L_hist'] = (np.zeros(50), np.zeros(51))
        
        # Entropy
        router_safe = torch.clamp(router_weights, min=1e-10)
        entropy = -(router_weights * torch.log(router_safe)).sum(dim=-1).mean()
        stats['entropy'] = float((entropy / np.log(model.M)).cpu())
        
        # Sparsity
        stats['sparsity'] = float((router_weights < 0.01).float().mean().cpu())
        
        # Routing consistency
        if prev_stats and 'router_weights' in prev_stats:
            prev_assignments = prev_stats['router_weights'].argmax(axis=-1)
            curr_assignments = router_weights.argmax(dim=-1).detach().cpu().numpy()
            stats['routing_consistency'] = float((prev_assignments == curr_assignments).mean())
        else:
            stats['routing_consistency'] = float('nan')
        
        # Expert pairwise L2 differences - skip for large N
        if not skip_expert_stats:
            expert_diffs = []
            for i in range(model.M):
                for j in range(i+1, model.M):
                    diff = torch.sqrt(torch.mean((expert_outputs[i] - expert_outputs[j])**2))
                    expert_diffs.append(float(diff.cpu()))
            stats['expert_l2_diffs'] = expert_diffs
        else:
            stats['expert_l2_diffs'] = []
        
        # Activation statistics by class
        y_np = y.detach().cpu().numpy()
        unique_labels = np.unique(y_np)
        stats['layer_stats_by_class'] = {}
        
        for label in unique_labels:
            label_mask = (y_np == label)
            if not label_mask.any():
                continue
            
            stats['layer_stats_by_class'][int(label)] = {
                'input': {
                    'mean': float(h1[label_mask].mean().cpu()),
                    'var': float(h1[label_mask].var(dim=-1).mean().cpu())
                },
                'router': {
                    'mean': float(router_logits[label_mask].mean().cpu()),
                    'var': float(router_logits[label_mask].var(dim=-1).mean().cpu())
                },
                'experts': {}
            }
            
            # Skip per-expert stats for large N
            if not skip_expert_stats:
                for m in range(model.M):
                    expert_m = expert_outputs[m][label_mask]
                    stats['layer_stats_by_class'][int(label)]['experts'][m] = {
                        'mean': float(expert_m.mean().cpu()),
                        'var': float(expert_m.var(dim=-1).mean().cpu())
                    }
        
        # Clear intermediate tensors
        if not skip_expert_stats:
            del expert_outputs
        torch.cuda.empty_cache()
    
    return stats


def compute_effective_updates(model, model_prev, X, args, model_init=None):
    """Compute per-step effective update ||ΔW^l x^{l-1}||_RMS for each layer.

    Measures how much a single gradient step moves the layer output — the key
    quantity for verifying μP feature-learning scaling (should be O(1) in width).
    """
    updates = {'raw': {}, 'normalized': {}}
    
    with torch.no_grad():
        # Input layer activations
        h = X @ (model.input_scale * model.w_in)
        h1 = model.apply_nonlin(h)
        
        # W_in update: includes input_scale in forward pass
        dW_in = model.w_in - model_prev.w_in
        update_in = (model.input_scale * dW_in).T @ X.T
        updates['raw']['W_in'] = float(torch.sqrt(torch.mean(update_in**2)).cpu())
        X_rms = torch.sqrt(torch.mean(X**2, dim=1, keepdim=True)) + 1e-12
        update_in_norm = (model.input_scale * dW_in).T @ (X / X_rms).T
        updates['normalized']['W_in'] = float(torch.sqrt(torch.mean(update_in_norm**2)).cpu())
        
        # Router update: includes router_scale in forward pass
        dQ = model.w_router - model_prev.w_router
        update_Q = (model.router_scale * dQ).T @ h1.T
        updates['raw']['Q'] = float(torch.sqrt(torch.mean(update_Q**2)).cpu())
        h1_rms = torch.sqrt(torch.mean(h1**2, dim=1, keepdim=True)) + 1e-12
        update_Q_norm = (model.router_scale * dQ).T @ (h1 / h1_rms).T
        updates['normalized']['Q'] = float(torch.sqrt(torch.mean(update_Q_norm**2)).cpu())
        
        # Expert updates - use efficient batched matmul with weight differences
        dW1_all = model.w_expert1 - model_prev.w_expert1  # [M, N, N_expert]
        update_exp1 = torch.stack([h1 @ (dW1_all.mul(model.expert1_scale)[m]) for m in range(dW1_all.shape[0])], dim=1)  # [batch, M, N_expert]
        update_exp1_list = torch.mean(update_exp1**2, dim=(0, 2))  # [M]
        
        h1_rms = torch.sqrt(torch.mean(h1**2, dim=1, keepdim=True)) + 1e-12
        update_exp1_norm = torch.stack([(h1 / h1_rms) @ (dW1_all.mul(model.expert1_scale)[m]) for m in range(dW1_all.shape[0])], dim=1)
        update_exp1_norm_list = torch.mean(update_exp1_norm**2, dim=(0, 2))
        updates['raw']['W_exp1'] = float(torch.sqrt(torch.mean(update_exp1_list)).cpu())
        updates['normalized']['W_exp1'] = float(torch.sqrt(torch.mean(update_exp1_norm_list)).cpu())
        
        # Second expert layer
        dW2_all = model.w_expert2 - model_prev.w_expert2  # [M, N_expert, N]
        # Compute first expert layer output: [batch, N] -> [batch, M, N_expert]
        e1_all = torch.stack([h1 @ (model.w_expert1.mul(model.expert1_scale)[m]) for m in range(model.w_expert1.shape[0])], dim=1)
        e1_all = model.apply_expert_nonlin(e1_all)  # [batch, M, N_expert]
        e1_all = e1_all.contiguous()
        
        update_exp2 = torch.stack([e1_all[:, m, :] @ (dW2_all.mul(model.expert2_scale)[m]) for m in range(dW2_all.shape[0])], dim=1)  # [batch, M, N]
        update_exp2_list = torch.mean(update_exp2**2, dim=(0, 2))  # [M]
        
        e1_rms = torch.sqrt(torch.mean(e1_all**2, dim=2, keepdim=True)) + 1e-12
        update_exp2_norm = torch.stack([(e1_all / e1_rms)[:, m, :] @ (dW2_all.mul(model.expert2_scale)[m]) for m in range(dW2_all.shape[0])], dim=1)
        update_exp2_norm_list = torch.mean(update_exp2_norm**2, dim=(0, 2))
        updates['raw']['W_exp2'] = float(torch.sqrt(torch.mean(update_exp2_list)).cpu())
        updates['normalized']['W_exp2'] = float(torch.sqrt(torch.mean(update_exp2_norm_list)).cpu())
        
        # Output layer: need combined expert output
        # Compute router weights and expert outputs
        router_logits = h1 @ (model.router_scale * model.w_router)
        
        router_weights = _compute_router_weights(router_logits, model.M, args)

        # Compute expert outputs using efficient batched matmul
        expert_outputs = compute_expert_outputs_batched(
            h1, model.w_expert1, model.w_expert2, model.expert1_scale, model.expert2_scale, model.apply_expert_nonlin,
            model_init=model_init, args=args
        )
        expert_outputs = expert_outputs.transpose(0, 1).transpose(1, 2)  # [M, batch, N] -> [batch, N, M]
        combined = (expert_outputs * router_weights.unsqueeze(1)).sum(dim=-1)  # [batch, N]

        # Apply RMSNorm if enabled
        if hasattr(model, 'post_agg_nonlin') and model.post_agg_nonlin == 'rmsnorm':
            combined = model.apply_rmsnorm(combined)
        elif hasattr(model, 'post_agg_nonlin') and model.post_agg_nonlin == 'sigmoid':
            combined = torch.sigmoid(args.post_agg_mult * combined)

        # Add residual connection if enabled
        if hasattr(args, 'use_residual') and args.use_residual:
            combined = combined + h1

        dW_out = model.w_out - model_prev.w_out
        update_out = (args.gamma * model.out_scale * dW_out).T @ combined.T
        updates['raw']['W_out'] = float(torch.sqrt(torch.mean(update_out**2)).cpu())
        combined_rms = torch.sqrt(torch.mean(combined**2, dim=1, keepdim=True)) + 1e-12
        update_out_norm = (args.gamma * model.out_scale * dW_out).T @ (combined / combined_rms).T
        updates['normalized']['W_out'] = float(torch.sqrt(torch.mean(update_out_norm**2)).cpu())
    
    return updates


def compute_total_effective_updates_per_layer(model_init, model_curr, X, args):
    """Compute cumulative effective update ||(W^l(t)-W^l(0)) x^{l-1}||_RMS for each layer.

    Unlike compute_effective_updates (which measures one step), this measures the
    total displacement from initialisation, used for the coordinate-check RCC plots.
    """
    updates = {'raw': {}, 'normalized': {}}
    
    with torch.no_grad():
        # Input layer activations
        h = X @ (model_curr.input_scale * model_curr.w_in)
        h1 = model_curr.apply_nonlin(h)
        
        # W_in update: includes input_scale in forward pass
        dW_in = model_curr.w_in - model_init.w_in
        update_in = (model_curr.input_scale * dW_in).T @ X.T
        updates['raw']['W_in'] = float(torch.sqrt(torch.mean(update_in**2)).cpu())
        X_rms = torch.sqrt(torch.mean(X**2, dim=1, keepdim=True)) + 1e-12
        update_in_norm = (model_curr.input_scale * dW_in).T @ (X / X_rms).T
        updates['normalized']['W_in'] = float(torch.sqrt(torch.mean(update_in_norm**2)).cpu())
        
        # Router update: includes router_scale in forward pass
        dQ = model_curr.w_router - model_init.w_router
        update_Q = (model_curr.router_scale * dQ).T @ h1.T
        updates['raw']['Q'] = float(torch.sqrt(torch.mean(update_Q**2)).cpu())
        h1_rms = torch.sqrt(torch.mean(h1**2, dim=1, keepdim=True)) + 1e-12
        update_Q_norm = (model_curr.router_scale * dQ).T @ (h1 / h1_rms).T
        updates['normalized']['Q'] = float(torch.sqrt(torch.mean(update_Q_norm**2)).cpu())
        
        # Expert updates - use efficient batched matmul with weight differences
        dW1_all = model_curr.w_expert1 - model_init.w_expert1  # [M, N, N_expert]
        update_exp1 = torch.stack([h1 @ (dW1_all.mul(model_curr.expert1_scale)[m]) for m in range(dW1_all.shape[0])], dim=1)  # [batch, M, N_expert]
        update_exp1_list = torch.mean(update_exp1**2, dim=(0, 2))  # [M]
        
        h1_rms = torch.sqrt(torch.mean(h1**2, dim=1, keepdim=True)) + 1e-12
        update_exp1_norm = torch.stack([(h1 / h1_rms) @ (dW1_all.mul(model_curr.expert1_scale)[m]) for m in range(dW1_all.shape[0])], dim=1)
        update_exp1_norm_list = torch.mean(update_exp1_norm**2, dim=(0, 2))
        updates['raw']['W_exp1'] = float(torch.sqrt(torch.mean(update_exp1_list)).cpu())
        updates['normalized']['W_exp1'] = float(torch.sqrt(torch.mean(update_exp1_norm_list)).cpu())
        
        # Second expert layer
        dW2_all = model_curr.w_expert2 - model_init.w_expert2  # [M, N_expert, N]
        # Compute first expert layer output: [batch, N] -> [batch, M, N_expert]
        e1_all = torch.stack([h1 @ (model_curr.w_expert1.mul(model_curr.expert1_scale)[m]) for m in range(model_curr.w_expert1.shape[0])], dim=1)
        e1_all = model_curr.apply_expert_nonlin(e1_all)  # [batch, M, N_expert]
        e1_all = e1_all.contiguous()
        
        update_exp2 = torch.stack([e1_all[:, m, :] @ (dW2_all.mul(model_curr.expert2_scale)[m]) for m in range(dW2_all.shape[0])], dim=1)  # [batch, M, N]
        update_exp2_list = torch.mean(update_exp2**2, dim=(0, 2))  # [M]
        
        e1_rms = torch.sqrt(torch.mean(e1_all**2, dim=2, keepdim=True)) + 1e-12
        update_exp2_norm = torch.stack([(e1_all / e1_rms)[:, m, :] @ (dW2_all.mul(model_curr.expert2_scale)[m]) for m in range(dW2_all.shape[0])], dim=1)
        update_exp2_norm_list = torch.mean(update_exp2_norm**2, dim=(0, 2))
        updates['raw']['W_exp2'] = float(torch.sqrt(torch.mean(update_exp2_list)).cpu())
        updates['normalized']['W_exp2'] = float(torch.sqrt(torch.mean(update_exp2_norm_list)).cpu())
        
        # Output layer: need combined expert output
        # Compute router weights and expert outputs
        router_logits = h1 @ (model_curr.router_scale * model_curr.w_router)
        
        router_weights = _compute_router_weights(router_logits, model_curr.M, args)

        # Compute expert outputs using efficient batched matmul
        expert_outputs = compute_expert_outputs_batched(
            h1, model_curr.w_expert1, model_curr.w_expert2, model_curr.expert1_scale, model_curr.expert2_scale, model_curr.apply_expert_nonlin,
            model_init=model_init, args=args
        )
        expert_outputs = expert_outputs.transpose(0, 1).transpose(1, 2)  # [M, batch, N] -> [batch, N, M]
        combined = (expert_outputs * router_weights.unsqueeze(1)).sum(dim=-1)  # [batch, N]

        # Apply RMSNorm if enabled
        if hasattr(model_curr, 'post_agg_nonlin') and model_curr.post_agg_nonlin == 'rmsnorm':
            combined = model_curr.apply_rmsnorm(combined)
        elif hasattr(model_curr, 'post_agg_nonlin') and model_curr.post_agg_nonlin == 'sigmoid':
            combined = torch.sigmoid(args.post_agg_mult * combined)

        # Add residual connection if enabled
        if hasattr(args, 'use_residual') and args.use_residual:
            combined = combined + h1

        dW_out = model_curr.w_out - model_init.w_out
        update_out = (args.gamma * model_curr.out_scale * dW_out).T @ combined.T
        updates['raw']['W_out'] = float(torch.sqrt(torch.mean(update_out**2)).cpu())
        combined_rms = torch.sqrt(torch.mean(combined**2, dim=1, keepdim=True)) + 1e-12
        update_out_norm = (args.gamma * model_curr.out_scale * dW_out).T @ (combined / combined_rms).T
        updates['normalized']['W_out'] = float(torch.sqrt(torch.mean(update_out_norm**2)).cpu())
    
    return updates

        
def compute_propagating_updates(model_init, model_curr, X, args):
    """Compute propagating update ||W^l(0) Δx^{l-1}||_RMS for each layer.

    Measures how feature changes from earlier layers propagate forward through
    the frozen initial weights — the complementary quantity to effective updates
    for the coordinate check (should also be O(1) in width under correct μP).
    Δx^l = activation_curr^l(X) - activation_init^l(X).
    """
    updates = {'raw': {}, 'normalized': {}}
    
    with torch.no_grad():
        # Compute activations with current and initial models
        h_curr = X @ (model_curr.input_scale * model_curr.w_in)
        h_curr = model_curr.apply_nonlin(h_curr)
        
        h_init = X @ (model_init.input_scale * model_init.w_in)
        h_init = model_init.apply_nonlin(h_init)
        
        # Router: Q_0 @ Δh (includes router_scale)
        dh = h_curr - h_init
        prop_Q = (model_init.router_scale * model_init.w_router).T @ dh.T
        updates['raw']['Q'] = float(torch.sqrt(torch.mean(prop_Q**2)).cpu())
        dh_rms = torch.sqrt(torch.mean(dh**2, dim=1, keepdim=True)) + 1e-12
        prop_Q_norm = (model_init.router_scale * model_init.w_router).T @ (dh / dh_rms).T
        updates['normalized']['Q'] = float(torch.sqrt(torch.mean(prop_Q_norm**2)).cpu())
        
        # Expert layers - use efficient batched matmul
        prop_exp1 = torch.stack([dh @ (model_init.w_expert1.mul(model_init.expert1_scale)[m]) for m in range(model_init.w_expert1.shape[0])], dim=1)
        prop_exp1_list = torch.mean(prop_exp1**2, dim=(0, 2))  # [M]
        
        prop_exp1_norm = torch.stack([(dh / dh_rms) @ (model_init.w_expert1.mul(model_init.expert1_scale)[m]) for m in range(model_init.w_expert1.shape[0])], dim=1)
        prop_exp1_norm_list = torch.mean(prop_exp1_norm**2, dim=(0, 2))
        updates['raw']['W_exp1'] = float(torch.sqrt(torch.mean(prop_exp1_list)).cpu())
        updates['normalized']['W_exp1'] = float(torch.sqrt(torch.mean(prop_exp1_norm_list)).cpu())
        
        # Second expert layer
        # Compute first expert layer output: [batch, N] -> [batch, M, N_expert]
        e1_curr = torch.stack([h_curr @ (model_curr.w_expert1.mul(model_curr.expert1_scale)[m]) for m in range(model_curr.w_expert1.shape[0])], dim=1)
        e1_curr = model_curr.apply_expert_nonlin(e1_curr)  # [batch, M, N_expert]
        e1_curr = e1_curr.contiguous()
        
        e1_init = torch.stack([h_init @ (model_init.w_expert1.mul(model_init.expert1_scale)[m]) for m in range(model_init.w_expert1.shape[0])], dim=1)
        e1_init = model_init.apply_expert_nonlin(e1_init)  # [batch, M, N_expert]
        e1_init = e1_init.contiguous()
        
        de1 = e1_curr - e1_init  # [batch, M, N_expert]
        de1 = de1.contiguous()
        
        prop_exp2 = torch.stack([de1[:, m, :] @ (model_init.w_expert2.mul(model_init.expert2_scale)[m]) for m in range(model_init.w_expert2.shape[0])], dim=1)
        prop_exp2_list = torch.mean(prop_exp2**2, dim=(0, 2))  # [M]
        
        de1_rms = torch.sqrt(torch.mean(de1**2, dim=2, keepdim=True)) + 1e-12
        prop_exp2_norm = torch.stack([(de1 / de1_rms)[:, m, :] @ (model_init.w_expert2.mul(model_init.expert2_scale)[m]) for m in range(model_init.w_expert2.shape[0])], dim=1)
        prop_exp2_norm_list = torch.mean(prop_exp2_norm**2, dim=(0, 2))
        updates['raw']['W_exp2'] = float(torch.sqrt(torch.mean(prop_exp2_list)).cpu())
        updates['normalized']['W_exp2'] = float(torch.sqrt(torch.mean(prop_exp2_norm_list)).cpu())
        
        # Output layer: need Δcombined (change in combined expert output)
        # Current model combined output
        router_logits_curr = h_curr @ (model_curr.router_scale * model_curr.w_router)
        
        router_weights_curr = _compute_router_weights(router_logits_curr, model_curr.M, args)
        
        # Compute expert outputs using efficient batched matmul
        expert_outputs_curr = compute_expert_outputs_batched(
            h_curr, model_curr.w_expert1, model_curr.w_expert2,
            model_curr.expert1_scale, model_curr.expert2_scale, model_curr.apply_expert_nonlin,
            model_init=model_init, args=args
        )
        expert_outputs_curr = expert_outputs_curr.transpose(0, 1).transpose(1, 2)  # [M, batch, N] -> [batch, N, M]
        combined_curr = (expert_outputs_curr * router_weights_curr.unsqueeze(1)).sum(dim=-1)

        # Initial model combined output
        router_logits_init = h_init @ (model_init.router_scale * model_init.w_router)
        
        router_weights_init = _compute_router_weights(router_logits_init, model_init.M, args)
        
        # Compute expert outputs using efficient batched matmul
        # For init model, pass model_init=None since this is the initial state (t=0)
        expert_outputs_init = compute_expert_outputs_batched(
            h_init, model_init.w_expert1, model_init.w_expert2,
            model_init.expert1_scale, model_init.expert2_scale, model_init.apply_expert_nonlin,
            model_init=None, args=args
        )
        expert_outputs_init = expert_outputs_init.transpose(0, 1).transpose(1, 2)  # [M, batch, N] -> [batch, N, M]
        combined_init = (expert_outputs_init * router_weights_init.unsqueeze(1)).sum(dim=-1)

        # Apply RMSNorm if enabled (to both curr and init)
        if hasattr(model_init, 'post_agg_nonlin') and model_init.post_agg_nonlin == 'rmsnorm':
            combined_curr = model_init.apply_rmsnorm(combined_curr)
            combined_init = model_init.apply_rmsnorm(combined_init)
        elif hasattr(model_init, 'post_agg_nonlin') and model_init.post_agg_nonlin == 'sigmoid':
            combined_curr = torch.sigmoid(args.post_agg_mult * combined_curr)
            combined_init = torch.sigmoid(args.post_agg_mult * combined_init)
        
        # Add residual connection if enabled (to both curr and init)
        if hasattr(args, 'use_residual') and args.use_residual:
            combined_curr = combined_curr + h_curr
            combined_init = combined_init + h_init
        
        dcombined = combined_curr - combined_init
        prop_out = (args.gamma * model_init.out_scale * model_init.w_out).T @ dcombined.T
        updates['raw']['W_out'] = float(torch.sqrt(torch.mean(prop_out**2)).cpu())
        dcombined_rms = torch.sqrt(torch.mean(dcombined**2, dim=1, keepdim=True)) + 1e-12
        prop_out_norm = (args.gamma * model_init.out_scale * model_init.w_out).T @ (dcombined / dcombined_rms).T
        updates['normalized']['W_out'] = float(torch.sqrt(torch.mean(prop_out_norm**2)).cpu())
    
    return updates


def compute_h_L(model, X, args, model_init=None):
    """Compute h^L (combined expert output) for given input X"""
    with torch.no_grad():
        return model(X, model_init=model_init if hasattr(args, 'separate_aggregation') and args.separate_aggregation else None, return_combined=True)


def compute_gradient_norms(model):
    """Compute RMS gradient norms for each parameter group (input, router, experts, output)."""
    norms = {}
    if model.w_in.grad is not None:
        norms['W_in'] = float(torch.sqrt(torch.mean(model.w_in.grad**2)).cpu())
    if model.w_router.grad is not None:
        norms['Q'] = float(torch.sqrt(torch.mean(model.w_router.grad**2)).cpu())
    if model.w_expert1.grad is not None:
        norms['W_exp1'] = float(torch.sqrt(torch.mean(model.w_expert1.grad**2)).cpu())
    if model.w_expert2.grad is not None:
        norms['W_exp2'] = float(torch.sqrt(torch.mean(model.w_expert2.grad**2)).cpu())
    if model.w_out.grad is not None:
        norms['W_out'] = float(torch.sqrt(torch.mean(model.w_out.grad**2)).cpu())
    return norms


def compute_h_L_rms_diff(model, X, h_L_init, args, model_init=None):
    """Compute RMS(h^L_t - h^L_0) where h^L is the combined expert output"""
    h_L = compute_h_L(model, X, args, model_init)
    diff = h_L - h_L_init
    return float(torch.sqrt(torch.mean(diff**2)).cpu())


def compute_h_L_rms(model, X, args, model_init=None):
    """Compute RMS(h^L) where h^L is the combined expert output"""
    h_L = compute_h_L(model, X, args, model_init)
    return float(torch.sqrt(torch.mean(h_L**2)).cpu())


def compute_activation_rms(model, X, args, model_init=None):
    """Compute RMS of all layer activations for coordinate check"""
    rms = {}
    
    with torch.no_grad():
        # h^1: First hidden layer after input
        h = X @ (model.input_scale * model.w_in)
        h1 = model.apply_nonlin(h)
        rms['h1'] = float(torch.sqrt(torch.mean(h1**2)).cpu())
        
        # Router logits
        router_logits = h1 @ (model.router_scale * model.w_router)
        rms['router_logits'] = float(torch.sqrt(torch.mean(router_logits**2)).cpu())
        
        # Expert activations - use efficient batched matmul
        expert_outputs = compute_expert_outputs_batched(
            h1, model.w_expert1, model.w_expert2, model.expert1_scale, model.expert2_scale, model.apply_expert_nonlin,
            model_init=model_init, args=args
        )  # [M, batch, N]
        
        # Compute intermediate activation (after first expert layer)
        e1_nonlin = torch.stack([h1 @ (model.w_expert1.mul(model.expert1_scale)[m]) for m in range(model.w_expert1.shape[0])], dim=1)  # [batch, M, N_expert]
        e1_nonlin = model.apply_expert_nonlin(e1_nonlin)
        expert_h2_list = torch.mean(e1_nonlin**2, dim=(0, 2))  # [M]
        
        expert_out_list = torch.mean(expert_outputs**2, dim=(1, 2))  # [M]
        
        rms['expert_h2'] = float(torch.sqrt(torch.mean(expert_h2_list)).cpu())
        rms['expert_out'] = float(torch.sqrt(torch.mean(expert_out_list)).cpu())
        
        # Combined expert output (h^L)
        h_L = model(X, model_init=model_init if hasattr(args, 'separate_aggregation') and args.separate_aggregation else None, return_combined=True)
        rms['h_L'] = float(torch.sqrt(torch.mean(h_L**2)).cpu())
        
        # RMSNorm output (if enabled)
        if hasattr(model, 'post_agg_nonlin') and model.post_agg_nonlin == 'rmsnorm':
            h_L_normed = model.apply_rmsnorm(h_L)
            rms['h_L_normed'] = float(torch.sqrt(torch.mean(h_L_normed**2)).cpu())
        elif hasattr(model, 'post_agg_nonlin') and model.post_agg_nonlin == 'sigmoid':
            h_L_sigmoid = torch.sigmoid(args.post_agg_mult * h_L)
            rms['h_L_sigmoid'] = float(torch.sqrt(torch.mean(h_L_sigmoid**2)).cpu())
        
        # Residual connection output (if enabled)
        if hasattr(args, 'use_residual') and args.use_residual:
            h_L_residual = h_L + h1
            rms['h_L_residual'] = float(torch.sqrt(torch.mean(h_L_residual**2)).cpu())
    
    return rms


def compute_output_gradient(model, X, y, args, model_init=None):
    """Compute dL/d(output) - gradient of loss w.r.t. network output

    Processes in chunks using eval_chunk_size to avoid OOM for large models.
    """
    model.eval()

    # Use eval_chunk_size for memory efficiency
    chunk_size = args.eval_chunk_size if hasattr(args, 'eval_chunk_size') else 1024
    num_samples = X.shape[0]
    num_chunks = (num_samples + chunk_size - 1) // chunk_size

    grad_sq_sum = 0.0
    total_elements = 0

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, num_samples)

        X_chunk = X[start_idx:end_idx]
        y_chunk = y[start_idx:end_idx]

        # Forward pass with gradients enabled
        with torch.enable_grad():
            logits = model(X_chunk, model_init=model_init if hasattr(args, 'separate_aggregation') and args.separate_aggregation else None)
            logits.requires_grad_(True)

            # Compute loss
            if args.use_cross_entropy:
                loss = F.cross_entropy(logits, y_chunk)
            else:
                if args.num_classes == 2:
                    y_targets = y_chunk.float() * 2 - 1
                    loss = F.mse_loss(logits.squeeze(-1), y_targets)
                else:
                    y_targets = F.one_hot(y_chunk, args.num_classes).float()
                    loss = F.mse_loss(logits, y_targets)

            # Compute gradient
            grad = torch.autograd.grad(loss, logits, create_graph=False)[0]

            # Accumulate squared gradients
            grad_sq_sum += float(torch.sum(grad**2).cpu())
            total_elements += grad.numel()

        # Clear cache after each chunk to avoid memory buildup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return float(np.sqrt(grad_sq_sum / total_elements))


def compute_weight_rms_norms(model):
    """Compute RMS norm for each trainable weight matrix

    Returns:
        dict: Dictionary with RMS norms for each weight parameter
            - W_in: Input layer weights
            - Q: Router weights
            - W_exp1: First expert layer weights (averaged over all experts)
            - W_exp2: Second expert layer weights (averaged over all experts)
            - W_out: Output layer weights
            - W_exp1_mean: Mean of per-expert RMS norms (first layer)
            - W_exp1_std: Std of per-expert RMS norms (first layer)
            - W_exp2_mean: Mean of per-expert RMS norms (second layer)
            - W_exp2_std: Std of per-expert RMS norms (second layer)
    """
    norms = {}

    with torch.no_grad():
        # Input layer: [N_in, N]
        norms['W_in'] = float(torch.sqrt(torch.mean(model.w_in**2)).cpu())

        # Router: [N, M]
        norms['Q'] = float(torch.sqrt(torch.mean(model.w_router**2)).cpu())

        # Expert layer 1: [M, N, N_expert] - compute RMS over all experts
        norms['W_exp1'] = float(torch.sqrt(torch.mean(model.w_expert1**2)).cpu())

        # Expert layer 2: [M, N_expert, N] - compute RMS over all experts
        norms['W_exp2'] = float(torch.sqrt(torch.mean(model.w_expert2**2)).cpu())

        # Output layer: [N, N_out]
        norms['W_out'] = float(torch.sqrt(torch.mean(model.w_out**2)).cpu())

        # Compute per-expert statistics (mean and std)
        expert1_norms = []
        expert2_norms = []
        for m in range(model.M):
            expert1_norms.append(float(torch.sqrt(torch.mean(model.w_expert1[m]**2)).cpu()))
            expert2_norms.append(float(torch.sqrt(torch.mean(model.w_expert2[m]**2)).cpu()))

        norms['W_exp1_mean'] = float(np.mean(expert1_norms))
        norms['W_exp1_std'] = float(np.std(expert1_norms))
        norms['W_exp2_mean'] = float(np.mean(expert2_norms))
        norms['W_exp2_std'] = float(np.std(expert2_norms))

    return norms

def compute_hagg_decomposition(model_init, model_curr, X, args):
    """Decompose h^agg(t) = sum_i phi_i(t) * W3i(t) * h2i(t) into 4 terms.

    Writing W3i(t) = W3i(0) + dW3i and h2i(t) = h2i(0) + dh2i:

      base:        sum_i phi_i(t) * W3i(0)  * h2i(0)   -- initial (base) term
      propagating: sum_i phi_i(t) * W3i(0)  * dh2i(t)  -- propagating term
      effective:   sum_i phi_i(t) * dW3i(t) * h2i(0)   -- effective term
      cross:       sum_i phi_i(t) * dW3i(t) * dh2i(t)  -- cross/higher-order term

    W3i = expert2_scale * w_expert2[i]  (the second expert-layer weight matrix)
    h2i = apply_expert_nonlin(h1 @ expert1_scale * w_expert1[i])

    phi(t) is the current router weight vector (same convention as compute_stats).
    Initial quantities use model_init weights/activations; current use model_curr.

    Returns a dict with RMS of each term: 'base', 'propagating', 'effective', 'cross'.
    """
    with torch.no_grad():
        # --- Current h1 and router weights phi(t) ---
        h_curr = X @ (model_curr.input_scale * model_curr.w_in)
        h1_curr = model_curr.apply_nonlin(h_curr)
        router_logits = h1_curr @ (model_curr.router_scale * model_curr.w_router)

        router_weights = _compute_router_weights(router_logits, model_curr.M, args)

        # --- Initial h1 ---
        h_init = X @ (model_init.input_scale * model_init.w_in)
        h1_init = model_init.apply_nonlin(h_init)

        scale1 = model_curr.expert1_scale
        scale2 = model_curr.expert2_scale

        # Accumulate the 4 terms by looping over experts (memory efficient)
        B, N = X.shape[0], model_curr.N
        base_term  = torch.zeros(B, N, device=X.device, dtype=X.dtype)
        prop_term  = torch.zeros(B, N, device=X.device, dtype=X.dtype)
        eff_term   = torch.zeros(B, N, device=X.device, dtype=X.dtype)
        cross_term = torch.zeros(B, N, device=X.device, dtype=X.dtype)

        for m in range(model_curr.M):
            phi_m = router_weights[:, m:m+1]  # [batch, 1]

            h2_init_m = model_curr.apply_expert_nonlin(
                h1_init @ (scale1 * model_init.w_expert1[m])
            )  # [batch, N_expert]
            h2_curr_m = model_curr.apply_expert_nonlin(
                h1_curr @ (scale1 * model_curr.w_expert1[m])
            )  # [batch, N_expert]
            dh2_m = h2_curr_m - h2_init_m

            W3_0_m = scale2 * model_init.w_expert2[m]                          # [N_expert, N]
            dW3_m  = scale2 * (model_curr.w_expert2[m] - model_init.w_expert2[m])  # [N_expert, N]

            base_term  += phi_m * (h2_init_m @ W3_0_m)
            prop_term  += phi_m * (dh2_m     @ W3_0_m)
            eff_term   += phi_m * (h2_init_m @ dW3_m)
            cross_term += phi_m * (dh2_m     @ dW3_m)

        def rms(t):
            return float(torch.sqrt(torch.mean(t ** 2)).cpu())

        return {
            'base':        rms(base_term),
            'propagating': rms(prop_term),
            'effective':   rms(eff_term),
            'cross':       rms(cross_term),
        }


def _apply_post_agg_transforms(combined, model, args, h1_const):
    """Apply post-aggregation nonlinearity and residual connection.

    Args:
        combined: Combined expert output [batch, N]
        model: The model (for accessing post_agg_nonlin)
        args: Arguments (for use_residual)
        h1_const: Constant h1 tensor (no grad) for residual connection

    Returns:
        Transformed combined output [batch, N]
    """
    # Apply post-aggregation nonlinearity if enabled
    if hasattr(model, 'post_agg_nonlin') and model.post_agg_nonlin == 'rmsnorm':
        combined = model.apply_rmsnorm(combined)
    elif hasattr(model, 'post_agg_nonlin') and model.post_agg_nonlin == 'sigmoid':
        combined = torch.sigmoid(args.post_agg_mult * combined)

    # Add residual connection if enabled
    if hasattr(args, 'use_residual') and args.use_residual:
        combined = combined + h1_const

    return combined


def _compute_preact_grads(model, model_init_for_sep_agg, X, y, args):
    """Return g_{μ,i} = ∂L/∂h^{2,in}_{μ,i} for each expert i via autograd.

    h^{2,in}_{μ,i} = h1_μ @ (expert1_scale * w_expert1[i]) is the pre-activation
    of the expert nonlinearity (between the two expert weight matrices).

    Each h^{2,in}_{μ,i} is made a differentiable leaf; the full forward pass
    (post-aggregation, output layer, loss) is rebuilt from those leaves so that
    backward() yields the correct per-expert preactivation gradients without
    touching any model parameter gradients.

    Args:
        model: the model whose weights define this forward pass.
        model_init_for_sep_agg: model_init for separate aggregation scaling
            (None when computing at t=0, model_init when computing at t>0).
        X: input batch [B, D].
        y: integer labels [B].
        args: training arguments.

    Returns:
        List of M tensors, each [B, N_expert], giving g_{μ,i}.
    """
    # --- Compute all constants with no grad ---
    with torch.no_grad():
        h = X @ (model.input_scale * model.w_in)
        h1 = model.apply_nonlin(h)

        router_logits = h1 @ (model.router_scale * model.w_router)
        router_weights = _compute_router_weights(router_logits, model.M, args)

        # Pre-activation values and detached second-layer weights
        h_pre_vals = [h1 @ (model.expert1_scale * model.w_expert1[m]) for m in range(model.M)]
        w3 = [model.expert2_scale * model.w_expert2[m].detach() for m in range(model.M)]
        w_out_d = model.w_out.detach()
        h1_const = h1  # no-grad tensor, used for residual

    # --- Build differentiable graph from h_pre leaves ---
    with torch.enable_grad():
        h_pre_leaves = [v.detach().requires_grad_(True) for v in h_pre_vals]

        B, N = X.shape[0], model.N
        combined = torch.zeros(B, N, device=X.device, dtype=X.dtype)
        for m in range(model.M):
            h_out_m = model.apply_expert_nonlin(h_pre_leaves[m])
            combined = combined + router_weights[:, m:m+1] * (h_out_m @ w3[m])

        # Mirror model forward: separate aggregation
        if hasattr(args, 'separate_aggregation') and args.separate_aggregation:
            if model_init_for_sep_agg is None:
                combined = combined / (model.M ** 0.5)

        combined = _apply_post_agg_transforms(combined, model, args, h1_const)

        logits = combined @ (args.gamma * model.out_scale * w_out_d)

        if args.use_cross_entropy:
            loss = F.cross_entropy(logits, y)
        else:
            if args.num_classes == 2:
                y_targets = y.float() * 2 - 1
                loss = F.mse_loss(logits.squeeze(-1), y_targets)
            else:
                y_targets = F.one_hot(y, args.num_classes).float()
                loss = F.mse_loss(logits, y_targets)

        loss.backward()

    return [
        h_pre_leaves[m].grad.detach() if h_pre_leaves[m].grad is not None
        else torch.zeros_like(h_pre_vals[m])
        for m in range(model.M)
    ]


def compute_expert_grad_h1_decomposition(model_init, model_curr, X, y, args):
    """Decompose the expert-induced gradient on h^1 into 4 terms.

    The expert contribution to ∇_{h^1_μ} L is:
        sum_i (W^{2,i}(t))^T g_{μ,i}(t)
    where g_{μ,i}(t) = ∂L/∂h^{2,in}_{μ,i}(t) is obtained via autograd and
    W^{2,i} = expert1_scale * w_expert1[i]  (first expert weight matrix).

    Writing W^{2,i}(t) = W^{2,i}(0) + ΔW^{2,i}(t)
    and     g_{μ,i}(t) = g_{μ,i}(0) + Δg_{μ,i}(t),
    all deltas relative to init, yields the exact four-term decomposition:

      base:        sum_i g_{μ,i}(0)   @ W^{2,i}(0)^T   (initial weights + gradients)
      propagating: sum_i Δg_{μ,i}(t)  @ W^{2,i}(0)^T   (initial weights, changed gradients)
      effective:   sum_i g_{μ,i}(0)   @ ΔW^{2,i}(t)^T  (changed weights, initial gradients)
      cross:       sum_i Δg_{μ,i}(t)  @ ΔW^{2,i}(t)^T  (both changed)

    Args:
        model_init: frozen initial model.
        model_curr: current model at time t.
        X: coord-check input batch [B, D].
        y: integer labels [B].
        args: training arguments.

    Returns:
        Dict with RMS scalar for each term: 'base', 'propagating', 'effective', 'cross'.
    """
    model_init_arg = (model_init
                      if hasattr(args, 'separate_aggregation') and args.separate_aggregation
                      else None)

    # g_{μ,i}(0): initial model, t=0 convention (no sep-agg delta)
    g_init = _compute_preact_grads(model_init, None, X, y, args)
    # g_{μ,i}(t): current model
    g_curr = _compute_preact_grads(model_curr, model_init_arg, X, y, args)

    B, N = X.shape[0], model_curr.N
    base_term  = torch.zeros(B, N, device=X.device, dtype=X.dtype)
    prop_term  = torch.zeros(B, N, device=X.device, dtype=X.dtype)
    eff_term   = torch.zeros(B, N, device=X.device, dtype=X.dtype)
    cross_term = torch.zeros(B, N, device=X.device, dtype=X.dtype)

    with torch.no_grad():
        for m in range(model_curr.M):
            W2_0 = model_init.expert1_scale * model_init.w_expert1[m]   # [N, N_expert]
            dW2  = (model_curr.expert1_scale * model_curr.w_expert1[m]
                    - W2_0)                                               # [N, N_expert]
            g0   = g_init[m]                                             # [B, N_expert]
            dg   = g_curr[m] - g_init[m]                                 # [B, N_expert]

            # [B, N_expert] @ [N_expert, N] = [B, N]
            base_term  += g0 @ W2_0.T
            prop_term  += dg @ W2_0.T
            eff_term   += g0 @ dW2.T
            cross_term += dg @ dW2.T

    def rms(t):
        return float(torch.sqrt(torch.mean(t ** 2)).cpu())

    return {
        'base':        rms(base_term),
        'propagating': rms(prop_term),
        'effective':   rms(eff_term),
        'cross':       rms(cross_term),
    }