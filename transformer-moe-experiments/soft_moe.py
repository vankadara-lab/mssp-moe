"""
Efficient soft MoE layer for the case where every token is routed to every
expert (top_k == n_exp) with sigmoid weighting.

Eliminates the capacity/ranking/masking overhead of the general-purpose Router,
reducing routing to a single linear projection + sigmoid.
"""

import torch
import torch.nn as nn

from manager import MANAGER
from nano_moe import MLPExperts


class SoftMoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.top_k == config.n_exp, (
            f"SoftMoELayer requires top_k == n_exp, got top_k={config.top_k}, n_exp={config.n_exp}"
        )
        self.n_exp = config.n_exp
        self.top_k = config.top_k
        self.moe_scaling_alpha = config.moe_scaling_alpha
        self.base_top_k = config.base_top_k
        self.router_forward_mult = config.router_forward_mult
        self.use_router_z_loss = config.use_router_z_loss
        # routing: single linear projection (no bias, same as Router.w_g)
        self.w_g = nn.Linear(config.n_embd, config.n_exp, bias=False)

        # experts: batched BMM
        self.experts = MLPExperts(config)

    def forward(self, x):
        B, T, n_embd = x.size()
        num_tokens = B * T

        # run routing in full precision to avoid instability
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

        with torch.amp.autocast(device_type=device_type, enabled=False):
            logits = self.w_g(x)  # [B, T, n_exp]
            weights = torch.sigmoid(logits)  # [B, T, n_exp]

            # scale routing weights by (base_top_k / top_k)^alpha (no-op when alpha = 0 or top_k == base_top_k)
            if self.moe_scaling_alpha != 0.0 and self.top_k != self.base_top_k:
                weights = weights * (self.base_top_k / self.top_k) ** self.moe_scaling_alpha

            if self.router_forward_mult != 1.0:
                weights = weights * self.router_forward_mult

            # sigmoid ratio diagnostic
            with torch.no_grad():
                sig_ratio = weights / weights.sum(dim=-1, keepdim=True)
                MANAGER.add_sigmoid_ratio(sig_ratio.mean(dim=(0, 1)))

            # router z-loss
            if self.use_router_z_loss:
                z_loss = torch.mean(torch.logsumexp(logits, dim=-1) ** 2.0)
                MANAGER.add_router_z_loss(z_loss)

            # router weight extremes
            with torch.no_grad():
                w_flat = weights.view(-1)
                MANAGER.add_router_weight_extremes(w_flat.max(), w_flat.min())

        # all tokens → all experts via BMM (expand is a zero-copy view)
        x_flat = x.view(num_tokens, n_embd)
        exp_input = x_flat.unsqueeze(0).expand(self.n_exp, -1, -1)  # [n_exp, num_tokens, n_embd]
        exp_out = self.experts(exp_input)  # [n_exp, num_tokens, n_embd]

        # weighted sum across experts
        w = weights.view(num_tokens, self.n_exp).t().unsqueeze(-1)  # [n_exp, num_tokens, 1]
        output = (w * exp_out).sum(dim=0)  # [num_tokens, n_embd]

        # output L2 diagnostic
        with torch.no_grad():
            MANAGER.add_output_l2(torch.linalg.vector_norm(output, ord=2))

        # expert stats: every expert gets every token
        with torch.no_grad():
            avg_w = weights.view(num_tokens, self.n_exp).mean(dim=0)
            used = torch.full((self.n_exp,), float(num_tokens), device=x.device)
            MANAGER.add_expert_stats(used, num_tokens, self.top_k, avg_w)

        return output.view(B, T, n_embd)
