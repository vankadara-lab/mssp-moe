"""
Efficient batched MoE layer implementation using batch matrix multiplication (BMM).

MLPExperts stores all expert weights as batched nn.Parameter tensors and processes
them with torch.bmm, avoiding per-expert loops.

Based on the nanoMoE-master implementation (ColossalAI OpenMoE style) but adapted
to the refactored codebase (no dropout, init handled by GPT._init_weights).
"""

import torch
import torch.nn as nn
from torch.nn import functional as F

from manager import MANAGER


class MLPExperts(nn.Module):
    """
    Multiple MLP-based experts that process input in batch using torch.bmm.
    Based on ColossalAI OpenMoE but simplified with optional bias and no dropout.
    Link: https://github.com/hpcaitech/ColossalAI/blob/main/colossalai/moe/experts.py
    """
    def __init__(self, config):
        super().__init__()
        self.bias = config.bias
        self.expert_hidden_forward_mult = config.expert_hidden_forward_mult

        hidden_dim = int(config.moe_ratio * config.n_embd)
        self.c_fc = nn.Parameter(torch.empty(config.n_exp, config.n_embd, hidden_dim))
        self.c_proj = nn.Parameter(torch.empty(config.n_exp, hidden_dim, config.n_embd))
        self.fc_bias = nn.Parameter(torch.empty(config.n_exp, 1, hidden_dim)) if self.bias else None
        self.proj_bias = nn.Parameter(torch.empty(config.n_exp, 1, config.n_embd)) if self.bias else None
        self.gelu = nn.GELU()

    def forward(self, x):
        x = torch.bmm(x, self.c_fc)
        if self.bias:
            x += self.fc_bias
        if self.expert_hidden_forward_mult != 1.0:
            x = x * self.expert_hidden_forward_mult
        x = self.gelu(x)
        x = torch.bmm(x, self.c_proj)
        if self.bias:
            x += self.proj_bias
        if self.expert_hidden_forward_mult != 1.0:
            x = x * self.expert_hidden_forward_mult
        return x


class NanoMoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Import here to avoid circular imports
        from model import Router
        self.router = Router(config)
        self.experts = MLPExperts(config)

    def forward(self, x):
        B, T, n_embd = x.size()
        num_tokens = B * T
        used_capacity, cb_weight, sec_mask = self.router(x)

        # cb_weight: [num_tokens, n_exp, exp_capacity]
        # sec_mask:  [num_tokens, n_exp, exp_capacity] (bool)

        # exp_weight: [n_exp, num_tokens] — sum routing weights across capacity slots
        exp_weight = cb_weight.sum(dim=-1).permute(1, 0)  # [n_exp, num_tokens]

        # exp_mask: binary mask [n_exp, num_tokens] — which tokens go to which expert
        exp_mask = sec_mask.any(dim=-1).float().permute(1, 0)  # [n_exp, num_tokens]

        # reshape tokens into expert batches via matrix multiply
        # exp_mask: [n_exp, num_tokens], x_flat: [num_tokens, n_embd]
        # exp_batches: [n_exp, num_tokens, n_embd] (sparse — zeros for non-assigned tokens)
        x_flat = x.view(num_tokens, n_embd)
        exp_batches = exp_mask.unsqueeze(-1) * x_flat.unsqueeze(0)  # [n_exp, num_tokens, n_embd]

        # process through experts
        exp_out = self.experts(exp_batches)  # [n_exp, num_tokens, n_embd]

        # aggregate outputs: weight and sum across experts
        # exp_weight: [n_exp, num_tokens] -> [n_exp, num_tokens, 1]
        output = (exp_weight.unsqueeze(-1) * exp_out).sum(dim=0)  # [num_tokens, n_embd]

        with torch.no_grad():
            MANAGER.add_output_l2(torch.linalg.vector_norm(output, ord=2))

        return output.view(B, T, n_embd)
