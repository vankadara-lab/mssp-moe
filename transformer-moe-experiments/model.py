"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import RMSNorm
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from manager import MANAGER
from nano_moe import NanoMoELayer
from moe_to_monitor import MonitorMoELayer
from soft_moe import SoftMoELayer
from scaling import get_scaling_config, classify_parameter, compute_fans


def build_rope_cache(seq_len, n_elem, device=None, base=10000, condense_ratio=1):
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)
    return torch.cos(idx_theta), torch.sin(idx_theta)


def apply_rope(x, cos, sin):
    head_size = x.size(-1)
    x1 = x[..., :head_size // 2]
    x2 = x[..., head_size // 2:]
    rotated = torch.cat((-x2, x1), dim=-1)
    roped = (x * cos) + (rotated * sin)
    return roped.to(dtype=x.dtype)


class CausalSelfAttention(nn.Module):

    def __init__(self, config, scaling_config=None):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.attn_bias)
        # qk norm
        self.q_norm = RMSNorm(config.n_embd // config.n_head, eps=config.rms_norm_epsilon, elementwise_affine=False) if config.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(config.n_embd // config.n_head, eps=config.rms_norm_epsilon, elementwise_affine=False) if config.qk_norm else nn.Identity()
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.attn_bias)
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # attention logit scale from scaling config
        head_dim = config.n_embd // config.n_head
        if scaling_config is not None:
            self.attn_scale = scaling_config.attn_scale(head_dim, config)
        else:
            self.attn_scale = None

    def forward(self, x, cos, sin):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # apply qk norm on the head dimension (as in https://arxiv.org/pdf/2309.14322)
        if self.config.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # apply rotary position embeddings (RoPE)
        rope_n_elem = int((self.config.n_embd // self.config.n_head) * self.config.rope_rotary_percentage)
        q_roped = apply_rope(q[..., :rope_n_elem], cos, sin)
        k_roped = apply_rope(k[..., :rope_n_elem], cos, sin)
        q = torch.cat((q_roped, q[..., rope_n_elem:]), dim=-1)
        k = torch.cat((k_roped, k[..., rope_n_elem:]), dim=-1)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        sdpa_kwargs = dict(attn_mask=None, is_causal=True)
        if self.attn_scale is not None:
            sdpa_kwargs['scale'] = self.attn_scale
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, **sdpa_kwargs)

        # re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # output projection
        y = self.c_proj(y)
        return y

class Router(nn.Module):

    def __init__(self, config):
        super().__init__()

        # router settings
        self.top_k = config.top_k
        self.n_exp = config.n_exp
        assert self.top_k >= 1 and self.top_k <= config.n_exp
        self.use_noisy_top_k = config.use_noisy_top_k
        self.drop_tokens = config.drop_tokens
        self.train_capacity = config.train_capacity
        self.eval_capacity = config.eval_capacity
        self.min_capacity = config.min_capacity
        self.routing_fn = config.routing_fn

        # moe scaling alpha: scale routing weights by (base_top_k / top_k)^alpha
        self.moe_scaling_alpha = config.moe_scaling_alpha
        self.base_top_k = config.base_top_k
        self.router_forward_mult = config.router_forward_mult

        # auxiliary / load balancing loss settings
        self.use_aux_loss = config.use_aux_loss
        self.use_router_z_loss = config.use_router_z_loss

        # linear projection for (noisy) softmax gating
        # no bias is used, see page 4 eq (4) in (https://arxiv.org/abs/1701.06538)
        self.w_g = nn.Linear(config.n_embd, config.n_exp, bias=False)
        self.w_noise = nn.Linear(config.n_embd, config.n_exp, bias=False) if self.use_noisy_top_k else None
    
    def forward(self, x):
        # run the router in full precision to avoid instability during training
        # see discussion on pg. 9 here: https://arxiv.org/abs/2101.03961
        # setting enabled to False in autocast automatically puts everything in float32
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

        with torch.amp.autocast(device_type=device_type, enabled=False):
            B, T, _ = x.size()
            num_tokens = B * T

            # eq (4) in (https://arxiv.org/abs/1701.06538)
            logits = self.w_g(x)  # [B, T, n_exp]
            if self.use_noisy_top_k:
                # optionally add noise into the router
                noise = F.softplus(self.w_noise(x))
                noise *= torch.randn_like(noise)
                logits += noise

            # sigmoid ratio diagnostic: sigmoid(logits) / sum(sigmoid(logits)), averaged over tokens
            with torch.no_grad():
                sig = torch.sigmoid(logits)  # [B, T, n_exp]
                sig_ratio = sig / sig.sum(dim=-1, keepdim=True)  # [B, T, n_exp]
                sig_ratio = sig_ratio.mean(dim=(0, 1))  # [n_exp]
                MANAGER.add_sigmoid_ratio(sig_ratio)

            # router z loss, computed on logits (before softmax)
            # this loss prevents router logits from becoming too large
            if self.use_router_z_loss:
                z_loss = self.compute_router_z_loss(logits)
                MANAGER.add_router_z_loss(z_loss)

            # find top k experts for each token
            top_k_logits, top_k_indices = logits.topk(self.top_k, dim=-1) # [B, T, k]

            # normalize expert probabilities
            if self.routing_fn == "softmax":
                # softmax over top-k: fill non-selected with -inf so softmax zeros them out
                # Shazeer et al (https://arxiv.org/abs/1701.06538) does only topk
                # see page 4 eq (3)-(5)
                router_probs = torch.full_like(logits, float('-inf'))  # [B, T, n_exp]
                router_probs.scatter_(-1, top_k_indices, top_k_logits)
                router_probs = F.softmax(router_probs, dim=-1)
            elif self.routing_fn == "sigmoid":
                # sigmoid applied independently per selected expert (no normalization)
                router_probs = torch.zeros_like(logits)  # [B, T, n_exp]
                router_probs.scatter_(-1, top_k_indices, torch.sigmoid(top_k_logits))
            else:
                raise ValueError(f"Unknown routing_fn: {self.routing_fn}")

            # scale routing weights by (base_top_k / top_k)^alpha (no-op when alpha = 0 or top_k == base_top_k)
            if self.moe_scaling_alpha != 0.0 and self.top_k != self.base_top_k:
                router_probs = router_probs * (self.base_top_k / self.top_k) ** self.moe_scaling_alpha

            if self.router_forward_mult != 1.0:
                router_probs = router_probs * self.router_forward_mult

            # compute auxiliary load balancing loss
            # this loss encourages equal probability assigned to each expert
            # and equal load balancing of tokens assigned to each expert
            if self.use_aux_loss:
                aux_loss = self.compute_aux_loss(router_probs, top_k_indices)
                MANAGER.add_aux_loss(aux_loss)

            # make a multi-hot mask of chosen experts, size [B, T, n_exp]
            # entries are 0 if expert not chosen and 1 if expert chosen
            exp_mask = F.one_hot(top_k_indices, num_classes=self.n_exp)  # [B, T, k, n_exp]
            exp_mask = exp_mask.view(num_tokens, self.top_k, self.n_exp)  # [B * T, k, n_exp]
            exp_mask = exp_mask.permute(1, 0, 2) # [k, B * T, n_exp]

            if self.drop_tokens:
                # --- capacity-based path: drop tokens exceeding expert capacity ---
                exp_capacity = self.get_capacity(num_tokens)

                # compute cumulative sum of each token over experts, this stores
                # the index of each token within the batch of each expert
                # NOTE: cumsum should count all top-1 first, top-2 second, etc.
                # so that we prioritize top experts when dropping tokens (this is
                # done by putting k dimension first for the reshape operation)
                exp_rank = exp_mask.reshape(self.top_k * num_tokens, self.n_exp)  # [k * B * T, n_exp]
                exp_rank = torch.cumsum(exp_rank, dim=0) - 1  # cumulative sum of expert selections [k * B * T, n_exp]
                exp_rank = exp_rank.reshape(self.top_k, num_tokens, self.n_exp)  # [k, B * T, n_exp]

                # mask out (set to zero) entries that go beyond expert capacity
                # compute amount of used capacity by taking a sum over mask
                exp_mask *= torch.lt(exp_rank, exp_capacity) # [k, B * T, n_exp]
                used_capacity = torch.sum(exp_mask, dim=(0, 1)) # [n_exp]

                # mask rank to only include tokens that are selected
                # perform a sum so each row only contains index of token
                # for the expert that is selected in that row
                # result is a matrix that contains the position of each token
                # in the batch of its corresponding expert
                exp_rank = torch.sum(exp_mask * exp_rank, dim=-1)  # [k, B * T]

                # mask probabilities to only include selected experts
                router_probs = router_probs.view(num_tokens, self.n_exp)[None, :] # [1, B * T, n_exp]
                exp_weights = exp_mask * router_probs # [k, B * T, n_exp]

                # track per-expert token counts and average routing weights
                avg_weights = exp_weights.sum(dim=(0, 1)) / num_tokens  # [n_exp]
                MANAGER.add_expert_stats(used_capacity, num_tokens, self.top_k, avg_weights)

                # router weight extremes: max/min of active (non-zero) routing weights
                with torch.no_grad():
                    active_mask = exp_mask.bool()  # [k, B * T, n_exp]
                    active_weights = exp_weights[active_mask]
                    if active_weights.numel() > 0:
                        MANAGER.add_router_weight_extremes(active_weights.max(), active_weights.min())
                    else:
                        zero = torch.tensor(0.0, device=x.device)
                        MANAGER.add_router_weight_extremes(zero, zero)

                # convert rank into one-hot vectors over the available capacity
                # stores the position of each token within the capacity of the selected expert
                exp_rank_sc = F.one_hot(exp_rank, num_classes=exp_capacity) # [k, B * T, exp_capacity]

                # create a vector that stores, for each token, the weight of selected
                # experts at token's position in the capacity of that expert
                # size of tensor is [B * T, n_exp, exp_capacity]
                cb_weight = torch.sum(exp_weights.unsqueeze(3) * exp_rank_sc.unsqueeze(2), dim=0)
                sec_mask = cb_weight.bool() # binary mask of selected experts for each token
                return used_capacity, cb_weight, sec_mask

            else:
                # --- no capacity limit: skip rank encoding entirely ---
                used_capacity = torch.sum(exp_mask, dim=(0, 1))  # [n_exp]

                router_probs = router_probs.view(num_tokens, self.n_exp)[None, :]  # [1, B*T, n_exp]
                exp_weights = exp_mask * router_probs  # [k, B*T, n_exp]

                # track per-expert token counts and average routing weights
                avg_weights = exp_weights.sum(dim=(0, 1)) / num_tokens  # [n_exp]
                MANAGER.add_expert_stats(used_capacity, num_tokens, self.top_k, avg_weights)

                # router weight extremes: max/min of active (non-zero) routing weights
                with torch.no_grad():
                    active_mask = exp_mask.bool()
                    active_weights = exp_weights[active_mask]
                    if active_weights.numel() > 0:
                        MANAGER.add_router_weight_extremes(active_weights.max(), active_weights.min())
                    else:
                        zero = torch.tensor(0.0, device=x.device)
                        MANAGER.add_router_weight_extremes(zero, zero)

                # collapse k dimension, add trivial capacity dim of 1
                cb_weight = exp_weights.sum(dim=0).unsqueeze(-1)  # [B*T, n_exp, 1]
                sec_mask = cb_weight.bool()
                return used_capacity, cb_weight, sec_mask
    
    def compute_aux_loss(self, expert_probs: torch.Tensor, indices: torch.Tensor):
        """
        Computes Switch Transformer auxiliary loss (https://arxiv.org/abs/2101.03961)
        See equations (4)-(6) on page 7
        """

        # equation (5): compute ratio of tokens allocated to each expert
        # total number of tokens is defined as total tokens in batch * k
        # (k = 1) for the Switch Transformer
        with torch.no_grad():
            one_hot_indices = F.one_hot(indices, num_classes=self.n_exp)  # [B, T, k, n_exp]
            one_hot_indices = torch.sum(one_hot_indices.float(), dim=2)  # [B, T, n_exp] (sum over k dimension)
            tokens_per_expert = torch.mean(one_hot_indices.float(), dim=(0, 1))

        # equation (6): compute ratio of router probability allocated to each expert
        prob_per_expert = torch.mean(expert_probs.float(), dim=(0, 1))

        # equation (4): take a scaled dot product between prob/token allocation vectors
        # multiply the result by the number of experts
        return self.n_exp * torch.sum(prob_per_expert * tokens_per_expert)
    
    def compute_router_z_loss(self, logits: torch.Tensor):
        """
        Computes ST-MoE router z loss (https://arxiv.org/abs/2202.08906)
        See equation (5) on page 7
        """
        z_loss = torch.logsumexp(logits, dim=-1) ** 2.0  # [B, T, n_exp]

        # sum over all tokens and divide by total number of tokens
        return torch.mean(z_loss)

    def get_capacity(self, tokens_per_batch):
        # expert capacity is given by (tokens_per_batch / num_experts) * capacity_factor
        # see eq (3) in Switch Transformer (https://arxiv.org/abs/2101.03961)
        capacity_factor = self.train_capacity if self.training else self.eval_capacity
        capacity = math.floor(self.top_k * capacity_factor * tokens_per_batch / self.n_exp)
        capacity += capacity % 2 # make sure capacity is an even number
        capacity = max(capacity, self.min_capacity) # use min capacity
        assert capacity > 0
        return int(capacity)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = config.mlp_ratio * config.n_embd
        self.c_fc    = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.dense_hidden_forward_mult = config.dense_hidden_forward_mult

    def forward(self, x):
        x = self.c_fc(x)
        if self.dense_hidden_forward_mult != 1.0:
            x = x * self.dense_hidden_forward_mult
        x = self.gelu(x)
        x = self.c_proj(x)
        if self.dense_hidden_forward_mult != 1.0:
            x = x * self.dense_hidden_forward_mult
        return x

class Block(nn.Module):

    def __init__(self, config, use_moe=False, scaling_config=None):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, eps=config.rms_norm_epsilon, elementwise_affine=config.rms_norm_elementwise_affine)
        self.attn = CausalSelfAttention(config, scaling_config=scaling_config)
        self.ln_2 = RMSNorm(config.n_embd, eps=config.rms_norm_epsilon, elementwise_affine=config.rms_norm_elementwise_affine)
        self.use_moe = use_moe
        if use_moe:
            if config.moe_implementation == "nanomoe":
                self.expert_module = NanoMoELayer(config)
            elif config.moe_implementation == "monitor":
                self.expert_module = MonitorMoELayer(config)
            elif config.moe_implementation == "softmoe":
                self.expert_module = SoftMoELayer(config)
            else:
                raise ValueError(f"Unknown moe_implementation: {config.moe_implementation}")
        else:
            self.mlp = MLP(config)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.ln_1(x), cos, sin)
        if self.use_moe:
            x = x + self.expert_module(self.ln_2(x))
        else:
            x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    sequence_length: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    mlp_ratio: int = 4  # dense MLP hidden dim = mlp_ratio * n_embd
    bias: bool = False # bias in Linears (MLP layers)
    attn_bias: bool = False # separate bias toggle for attention c_attn and c_proj
    weight_tying: bool = False # tie wte and lm_head weights
    rms_norm_epsilon: float | None = None # if None, defaults to torch.finfo(x.dtype).eps
    rms_norm_elementwise_affine: bool = True
    qk_norm: bool = True # apply RMSNorm to q and k in attention

    # RoPE configs
    rope_base: int = 10000          # base frequency for RoPE inverse frequencies
    rope_rotary_percentage: float = 1.0  # fraction of head_size to apply RoPE to (1.0 = full)
    rope_condense_ratio: int = 1    # position index condensation ratio

    # MoE-related configs
    moe_implementation: str = "nanomoe"  # "nanomoe" (batched BMM) or "monitor" (per-expert modules)
    n_exp: int = 1 # if n_exp = 1 we just use regular MLP layers
    top_k: int = 2
    use_aux_loss: bool = False # apply auxiliary loss (from Switch Transformer) in router
    use_router_z_loss: bool = False # apply router z loss (from ST-MoE)
    use_noisy_top_k: bool = False
    aux_loss_weight: float = 0.01 # default setting from Switch Transformer (see top of page 8)
    router_z_loss_weight: float = 0.001 # default setting from ST-MoE (see page 8 eq. 6)
    drop_tokens: bool = True  # enforce expert capacity limit; False = no token dropping
    train_capacity: float = 1.25  # default setting from ST-MoE (see top of page 6)
    eval_capacity: float = 2.0
    min_capacity: int = 4  # minimum batch size to send to any single expert
    stride: int = 2 # one in every stride layers are converted to an MoE
    n_dense_layers_before_moe: int = 0  # first n layers are always dense, regardless of stride
    routing_fn: str = "softmax"  # "softmax" or "sigmoid" for expert weight computation
    fixed_router: bool = False
    moe_ratio: int | float = 4  # expert MLP hidden dim = moe_ratio * n_embd

    # gradient checkpointing
    gradient_checkpointing: bool = False

    # scaling configs
    scaling: str = "standard"  # scaling config name (see scaling.py registry)
    moe_scaling_alpha: float = 0.0  # scale routing weights by (base_top_k / top_k)^alpha (0 = no-op)
    base_top_k: int = 1  # reference top_k for moe_scaling_alpha (scaling is 1 when top_k == base_top_k)

    # forward-pass multipliers (for HP tuning, reported under scaling)
    input_forward_mult: float = 1.0
    dense_hidden_forward_mult: float = 1.0
    expert_hidden_forward_mult: float = 1.0
    output_forward_mult: float = 1.0
    router_forward_mult: float = 1.0

    # per-role LR multipliers (additional factor on top of scaling config)
    attn_qkv_lr_mult: float = 1.0
    attn_proj_lr_mult: float = 1.0
    mlp_up_lr_mult: float = 1.0
    mlp_down_lr_mult: float = 1.0
    expert_up_lr_mult: float = 1.0
    expert_down_lr_mult: float = 1.0
    router_lr_mult: float = 1.0
    input_lr_mult: float = 1.0
    output_lr_mult: float = 1.0

    # per-role init multipliers (additional factor on init std, on top of scaling config)
    attn_qkv_init_mult: float = 1.0
    attn_proj_init_mult: float = 1.0
    mlp_up_init_mult: float = 1.0
    mlp_down_init_mult: float = 1.0
    expert_up_init_mult: float = 1.0
    expert_down_init_mult: float = 1.0
    router_init_mult: float = 1.0
    input_init_mult: float = 1.0
    output_init_mult: float = 1.0

    # tied expert initialization: copy expert 0 weights to all other experts
    tied_expert_init: bool = False


class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.sequence_length is not None
        self.config = config

        # load scaling config
        self.scaling = get_scaling_config(config.scaling)

        if config.n_exp == 1:
            # create normal transformer blocks
            blocks = nn.ModuleList([Block(config, scaling_config=self.scaling) for _ in range(config.n_layer)])
        else:
            # create transformer blocks, placing an MoE block every <stride> layers
            blocks = []
            for i in range(config.n_layer):
                # we start with a dense layer, then every <stride> layers use MoE
                use_moe = (i >= config.n_dense_layers_before_moe) and ((i % config.stride) == (config.stride - 1))
                blocks.append(Block(config, use_moe=use_moe, scaling_config=self.scaling))
            blocks = nn.ModuleList(blocks)

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = blocks,
            ln_f = RMSNorm(config.n_embd, eps=config.rms_norm_epsilon, elementwise_affine=config.rms_norm_elementwise_affine),
        ))
        rope_n_elem = int((config.n_embd // config.n_head) * config.rope_rotary_percentage)
        cos, sin = build_rope_cache(config.sequence_length, rope_n_elem)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.weight_tying:
            self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # forward-pass scales (compose config multipliers with scaling config methods)
        self.effective_input_forward_mult = config.input_forward_mult * self.scaling.input_embed_scale(config)
        self.effective_output_forward_mult = config.output_forward_mult * self.scaling.output_logit_scale(config)

        # initialize all weights via the scaling config
        self._apply_init()

        # freeze router weights for uniform routing
        if config.fixed_router:
            for m in self.modules():
                if isinstance(m, (SoftMoELayer, Router)):
                    nn.init.zeros_(m.w_g.weight)
                    m.w_g.weight.requires_grad = False

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        When weight_tying is enabled, wte and lm_head share the same tensor
        so they are only counted once.  When disabled, both are separate and
        both are counted.  RoPE buffers are not parameters, so nothing to subtract.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    @torch.no_grad()
    def _apply_init(self):
        """Initialize all parameters using the scaling config."""
        for name, p in self.named_parameters():
            role = classify_parameter(name)
            if role == 'bias':
                self.scaling.init_bias(p)
            else:
                fan_in, fan_out = compute_fans(p, role)
                self.scaling.init_weight(role, p, fan_in, fan_out, self.config)
                mult = self._role_init_mult(role)
                if mult != 1.0:
                    p.mul_(mult)

        if self.config.tied_expert_init and self.config.n_exp > 1:
            self._tie_expert_init()

    def _tie_expert_init(self):
        """Copy expert 0 weights/biases to all other experts in each MoE layer."""
        for block in self.transformer.h:
            moe = getattr(block, 'expert_module', None)
            if moe is None:
                continue

            if isinstance(moe, (NanoMoELayer, SoftMoELayer)):
                experts = moe.experts
                for i in range(1, experts.c_fc.shape[0]):
                    experts.c_fc[i].copy_(experts.c_fc[0])
                    experts.c_proj[i].copy_(experts.c_proj[0])
                    if experts.fc_bias is not None:
                        experts.fc_bias[i].copy_(experts.fc_bias[0])
                    if experts.proj_bias is not None:
                        experts.proj_bias[i].copy_(experts.proj_bias[0])

            elif isinstance(moe, MonitorMoELayer):
                expert_list = list(moe.experts.values())
                expert0 = expert_list[0]
                for expert in expert_list[1:]:
                    expert.c_fc.weight.copy_(expert0.c_fc.weight)
                    expert.c_proj.weight.copy_(expert0.c_proj.weight)
                    if expert.c_fc.bias is not None:
                        expert.c_fc.bias.copy_(expert0.c_fc.bias)
                    if expert.c_proj.bias is not None:
                        expert.c_proj.bias.copy_(expert0.c_proj.bias)

            else:
                raise ValueError(f"tied_expert_init: unknown MoE layer type {type(moe).__name__}")

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.sequence_length, f"Cannot forward sequence of length {t}, block size is only {self.config.sequence_length}"

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        x = tok_emb
        if self.effective_input_forward_mult != 1.0:
            x = x * self.effective_input_forward_mult
        cos = self.cos[:t]
        sin = self.sin[:t]
        for block in self.transformer.h:
            if self.config.gradient_checkpointing and self.training:
                x = torch_checkpoint(
                    block, x, cos, sin,
                    use_reentrant=False,
                    context_fn=lambda: (nullcontext(), MANAGER.recompute_context()),
                )
            else:
                x = block(x, cos, sin)
        x = self.transformer.ln_f(x)

        if self.effective_output_forward_mult != 1.0:
            x = x * self.effective_output_forward_mult

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

            # compute total_loss = ce_loss + weighted aux/z-loss (used for backprop)
            total_loss = ce_loss
            aux_loss = None
            z_loss = None
            if self.config.n_exp > 1 and self.config.use_aux_loss:
                aux_loss = MANAGER.aggregate_aux_loss()
                total_loss = total_loss + self.config.aux_loss_weight * aux_loss
                MANAGER.reset_aux_loss()
            if self.config.n_exp > 1 and self.config.use_router_z_loss:
                z_loss = MANAGER.aggregate_router_z_loss()
                total_loss = total_loss + self.config.router_z_loss_weight * z_loss
                MANAGER.reset_router_z_loss()
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            ce_loss = total_loss = aux_loss = z_loss = None

        return logits, ce_loss, total_loss, aux_loss, z_loss

    def _role_lr_mult(self, role):
        mapping = {
            'attn_qkv': self.config.attn_qkv_lr_mult,
            'attn_proj': self.config.attn_proj_lr_mult,
            'mlp_up': self.config.mlp_up_lr_mult,
            'mlp_down': self.config.mlp_down_lr_mult,
            'expert_up': self.config.expert_up_lr_mult,
            'expert_down': self.config.expert_down_lr_mult,
            'router': self.config.router_lr_mult,
            'embedding': self.config.input_lr_mult,
            'output': self.config.output_lr_mult,
        }
        return mapping.get(role, 1.0)

    def _role_init_mult(self, role):
        mapping = {
            'attn_qkv': self.config.attn_qkv_init_mult,
            'attn_proj': self.config.attn_proj_init_mult,
            'mlp_up': self.config.mlp_up_init_mult,
            'mlp_down': self.config.mlp_down_init_mult,
            'expert_up': self.config.expert_up_init_mult,
            'expert_down': self.config.expert_down_init_mult,
            'router': self.config.router_init_mult,
            'embedding': self.config.input_init_mult,
            'output': self.config.output_init_mult,
        }
        return mapping.get(role, 1.0)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, eps=1e-8, optimizer_type="adamw"):
        # role-aware parameter grouping using scaling config
        groups = {}
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            role = classify_parameter(name)
            lr_s = self.scaling.lr_scale(role, self.config)
            lr_s *= self._role_lr_mult(role)
            eps_s = self.scaling.eps_scale(role, self.config)
            group_eps = eps * eps_s
            # weight decay for all tensors, but we correct for the learning rate scaling so that
            # all tensors have the same effective weight decay NOTE this is important for principled scaling experiments
            # but it significantly changes the role of the weight decay hyperparameter compared to standard practice.
            # In this setup, one would likely want to decrease the weight decay value as we increase model size and training time
            # see e.g. https://arxiv.org/pdf/2505.13738? for the role of weight decay in the standard setup
            wd = weight_decay / lr_s
            key = (wd, lr_s, group_eps)
            groups.setdefault(key, []).append(p)

        optim_groups = [
            {'params': params, 'weight_decay': wd, 'lr_scale': lr_s, 'eps': group_eps}
            for (wd, lr_s, group_eps), params in groups.items()
        ]
        num_param_groups = len(optim_groups)
        total_params = sum(p.numel() for g in optim_groups for p in g['params'])
        total_tensors = sum(len(g['params']) for g in optim_groups)
        print(f"num parameter groups: {num_param_groups}, {total_tensors} tensors, {total_params:,} parameters (all decayed)")

        if optimizer_type == "adamw":
            # Create AdamW optimizer and use the fused version if it is available
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and device_type == 'cuda'
            extra_args = dict(fused=True) if use_fused else dict()
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
            print(f"using fused AdamW: {use_fused}")
        elif optimizer_type == "sgd":
            # Strip weight_decay from param groups; SGD only needs params and lr_scale
            sgd_groups = [
                {'params': g['params'], 'lr_scale': g['lr_scale']}
                for g in optim_groups
            ]
            optimizer = torch.optim.SGD(sgd_groups, lr=learning_rate)
            print("using SGD")
        else:
            raise ValueError(f"Unknown optimizer_type: {optimizer_type}")

        return optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at sequence_length
            idx_cond = idx if idx.size(1) <= self.config.sequence_length else idx[:, -self.config.sequence_length:]
            # forward the model to get the logits for the index in the sequence
            logits, _, _, _, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx   
