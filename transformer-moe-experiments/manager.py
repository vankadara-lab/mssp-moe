import torch
from contextlib import contextmanager

class MOEManager:
    """
    basic wrapper class for tracking, storing, and aggregating auxiliary
    losses across multiple MoE layers in the model
    """

    def __init__(self):
        self.aux_loss = []
        self.router_z_loss = []
        self.expert_stats = []
        self.output_l2_stats = []
        self.router_weight_extremes = []
        self.sigmoid_ratio_stats = []
        self._recomputing = False

    @contextmanager
    def recompute_context(self):
        """Context manager that suppresses stat collection during checkpoint recomputation."""
        self._recomputing = True
        try:
            yield
        finally:
            self._recomputing = False

    def reset_aux_loss(self):
        self.aux_loss = []

    def reset_router_z_loss(self):
        self.router_z_loss = []

    def add_aux_loss(self, loss):
        if self._recomputing:
            return
        self.aux_loss.append(loss)

    def add_router_z_loss(self, loss):
        if self._recomputing:
            return
        self.router_z_loss.append(loss)
    
    def aggregate_aux_loss(self):
        return sum(self.aux_loss)

    def aggregate_router_z_loss(self):
        return sum(self.router_z_loss)

    def add_expert_stats(self, used_capacity, num_tokens, top_k, avg_weights):
        if self._recomputing:
            return
        self.expert_stats.append({
            'used_capacity': used_capacity.detach(),
            'num_tokens': num_tokens,
            'top_k': top_k,
            'avg_weights': avg_weights.detach(),
        })

    def aggregate_expert_stats(self, n_moe_layers):
        """Average expert stats across micro-batches for each MoE layer.

        Stats are appended in layer order per forward pass (layer 0, 1, ..., n-1,
        then layer 0, 1, ... for the next micro-batch). This groups by layer
        position and averages across micro-batches.

        Returns (tokens, weights) where each is a list of n_moe_layers CPU
        tensors of shape [n_exp].
        - tokens: fraction of tokens routed to each expert
        - weights: average softmax routing weight per expert per token
        """
        if not self.expert_stats:
            return [], []
        token_results = []
        weight_results = []
        for layer_i in range(n_moe_layers):
            layer_entries = self.expert_stats[layer_i::n_moe_layers]
            fractions = []
            weights = []
            for entry in layer_entries:
                uc = entry['used_capacity'].float()
                total = entry['num_tokens']
                fractions.append(uc / total)
                weights.append(entry['avg_weights'].float())
            token_results.append(torch.stack(fractions).mean(dim=0).cpu())
            weight_results.append(torch.stack(weights).mean(dim=0).cpu())
        return token_results, weight_results

    def reset_expert_stats(self):
        self.expert_stats = []

    # --- output L2 norm ---
    def add_output_l2(self, l2_norm):
        if self._recomputing:
            return
        self.output_l2_stats.append(l2_norm.detach())

    def aggregate_output_l2(self, n_moe_layers):
        """Average L2 norms across micro-batches for each MoE layer.
        Returns a list of n_moe_layers CPU scalars."""
        if not self.output_l2_stats:
            return []
        results = []
        for layer_i in range(n_moe_layers):
            layer_entries = self.output_l2_stats[layer_i::n_moe_layers]
            results.append(torch.stack(layer_entries).mean().cpu())
        return results

    def reset_output_l2(self):
        self.output_l2_stats = []

    # --- router weight extremes ---
    def add_router_weight_extremes(self, w_max, w_min):
        if self._recomputing:
            return
        self.router_weight_extremes.append((w_max.detach(), w_min.detach()))

    def aggregate_router_weight_extremes(self, n_moe_layers):
        """Max-of-maxes, min-of-mins across micro-batches for each MoE layer.
        Returns (maxes, mins) where each is a list of n_moe_layers CPU scalars."""
        if not self.router_weight_extremes:
            return [], []
        maxes = []
        mins = []
        for layer_i in range(n_moe_layers):
            layer_entries = self.router_weight_extremes[layer_i::n_moe_layers]
            layer_maxes = torch.stack([e[0] for e in layer_entries])
            layer_mins = torch.stack([e[1] for e in layer_entries])
            maxes.append(layer_maxes.max().cpu())
            mins.append(layer_mins.min().cpu())
        return maxes, mins

    def reset_router_weight_extremes(self):
        self.router_weight_extremes = []

    # --- sigmoid ratio ---
    def add_sigmoid_ratio(self, ratio):
        if self._recomputing:
            return
        self.sigmoid_ratio_stats.append(ratio.detach())

    def aggregate_sigmoid_ratio(self, n_moe_layers):
        """Average sigmoid ratios across micro-batches for each MoE layer.
        Returns a list of n_moe_layers CPU tensors of shape [n_exp]."""
        if not self.sigmoid_ratio_stats:
            return []
        results = []
        for layer_i in range(n_moe_layers):
            layer_entries = self.sigmoid_ratio_stats[layer_i::n_moe_layers]
            results.append(torch.stack(layer_entries).mean(dim=0).cpu())
        return results

    def reset_sigmoid_ratio(self):
        self.sigmoid_ratio_stats = []

MANAGER = MOEManager()