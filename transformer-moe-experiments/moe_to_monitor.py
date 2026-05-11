"""
Per-expert nn.Module MoE layer implementation, useful for monitoring/debugging.

Each expert is an individual module in a ModuleDict, making it easy to attach
hooks and inspect individual expert behavior with torch_module_monitor.
"""

import torch
import torch.nn as nn

from manager import MANAGER


class Expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(config.moe_ratio * config.n_embd)
        self.c_fc   = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.gelu   = nn.GELU()
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.expert_hidden_forward_mult = config.expert_hidden_forward_mult

    def forward(self, x):  # x: [num_tokens, n_embd]
        x = self.c_fc(x)
        if self.expert_hidden_forward_mult != 1.0:
            x = x * self.expert_hidden_forward_mult
        x = self.gelu(x)
        x = self.c_proj(x)
        if self.expert_hidden_forward_mult != 1.0:
            x = x * self.expert_hidden_forward_mult
        return x


class MonitorMoELayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Import here to avoid circular imports
        from model import Router
        self.n_exp = config.n_exp
        self.router = Router(config)
        self.experts = nn.ModuleDict({
            f"expert{i}": Expert(config) for i in range(config.n_exp)
        })
        self.top_k = config.top_k

    def forward(self, x):
        B, T, n_embd = x.size()
        num_tokens = B * T
        used_capacity, cb_weight, sec_mask = self.router(x)
        x_flat = x.view(num_tokens, n_embd)
        output = torch.zeros(num_tokens, n_embd, device=x.device, dtype=x.dtype)

        for i, expert in enumerate(self.experts.values()):
            # Which tokens go to this expert (any capacity slot set)
            token_mask = sec_mask[:, i, :].any(dim=-1)  # [B*T]
            if not token_mask.any():
                continue
            expert_input = x_flat[token_mask]              # [N_sel, n_embd]
            expert_output = expert(expert_input)            # [N_sel, n_embd]
            # Sum routing weight across capacity slots (only one is nonzero per token)
            weights = cb_weight[token_mask, i, :].sum(-1)  # [N_sel]
            output[token_mask] += weights.unsqueeze(-1) * expert_output

        with torch.no_grad():
            MANAGER.add_output_l2(torch.linalg.vector_norm(output, ord=2))

        return output.view(B, T, n_embd)
