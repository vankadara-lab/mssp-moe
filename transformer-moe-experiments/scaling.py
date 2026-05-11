"""
Scaling configurations for weight initialization, per-parameter LR scaling,
and forward-pass scaling (attention logits, output logits, input embeddings).

Each ScalingConfig subclass bundles all these decisions into one pluggable object.
StandardScaling provides vanilla 1/sqrt(fan_in) init. Future configs (muP, etc.)
can be added without touching model.py or train.py.
"""

import torch


# ---------------------------------------------------------------------------
# Parameter role classification
# ---------------------------------------------------------------------------

def classify_parameter(name: str) -> str:
    """Map a named parameter to its scaling role.

    Roles: embedding, output, router, attn_qkv, attn_proj,
           mlp_up, mlp_down, expert_up, expert_down, bias, norm.
    """
    # Biases (check first — covers .bias, fc_bias, proj_bias)
    if name.endswith('.bias') or name.endswith('fc_bias') or name.endswith('proj_bias'):
        return 'bias'

    # Norm layers
    if 'ln_' in name or 'ln_f' in name:
        return 'norm'

    # Embedding
    if 'wte' in name:
        return 'embedding'

    # Output head
    if 'lm_head' in name:
        return 'output'

    # Router
    if 'w_g' in name or 'w_noise' in name:
        return 'router'

    # Attention QKV (must check before generic c_proj)
    if 'attn.c_attn' in name:
        return 'attn_qkv'

    # Attention output projection
    if 'attn.c_proj' in name:
        return 'attn_proj'

    # Expert MLP (must check before regular MLP — both contain c_fc / c_proj)
    if 'expert' in name and 'c_fc' in name:
        return 'expert_up'
    if 'expert' in name and 'c_proj' in name:
        return 'expert_down'

    # Regular MLP
    if 'mlp.c_fc' in name:
        return 'mlp_up'
    if 'mlp.c_proj' in name:
        return 'mlp_down'

    raise ValueError(f"Cannot classify parameter: {name}")


def compute_fans(param: torch.Tensor, role: str = None) -> tuple[int, int]:
    """Return (fan_in, fan_out) for a parameter tensor.

    Handles 2-D nn.Linear weights [out, in], 2-D nn.Embedding weights
    [num_embeddings, embedding_dim], and 3-D batched expert parameters
    [n_exp, in, out].
    """
    if param.dim() == 2:
        if role == 'embedding':
            # nn.Embedding: weight is [num_embeddings, embedding_dim]
            return param.shape[0], param.shape[1]
        # nn.Linear convention: weight is [out_features, in_features]
        return param.shape[1], param.shape[0]
    elif param.dim() == 3:
        # Batched experts: [n_exp, in_features, out_features]
        return param.shape[1], param.shape[2]
    else:
        return 1, 1
    

def is_input_like(role: str) -> bool:
    """Return True if the role of the parameter is input-like. This applies to 

    - embedding layer
    - all biases (in norms and projection layers)
    - rms norm / layernorm weights
    """
    return role in {'embedding', 'norm', 'bias'}


# ---------------------------------------------------------------------------
# ScalingConfig base class
# ---------------------------------------------------------------------------

class ScalingConfig:
    """Base class for scaling configurations.

    Subclasses override methods to define initialization, per-group LR, and
    forward-pass multipliers.  The base implementation provides a simple
    default (normal init with std=0.02, flat LR, standard attention scaling).
    """

    BASE_WIDTH = 256
    BASE_EXPERTS = 8

    # --- initialization ----------------------------------------------------

    @torch.no_grad()
    def init_weight(self, role: str, param: torch.Tensor,
                    fan_in: int, fan_out: int, config) -> None:
        """Initialize a weight tensor in-place."""
        torch.nn.init.normal_(param, mean=0.0, std=0.02)

    @torch.no_grad()
    def init_bias(self, param: torch.Tensor) -> None:
        """Initialize a bias tensor in-place."""
        torch.nn.init.zeros_(param)

    # --- per-group learning-rate -------------------------------------------

    def lr_scale(self, role: str, config) -> float:
        """LR multiplier for this parameter role.  Default: 1.0."""
        return 1.0

    def eps_scale(self, role: str, config) -> float:
        """Epsilon multiplier for this parameter role.  Default: 1.0."""
        return 1.0

    # --- forward-pass scales -----------------------------------------------

    def attn_scale(self, head_dim: int, config) -> float | None:
        """Attention logit scale passed to scaled_dot_product_attention.

        Return ``None`` to use PyTorch's default (1 / sqrt(head_dim)).
        """
        return None

    def output_logit_scale(self, config) -> float:
        """Multiplier applied to hidden states before the lm_head projection."""
        return 1.0

    def input_embed_scale(self, config) -> float:
        """Multiplier applied to the embedding output."""
        return 1.0

# ---------------------------------------------------------------------------
# StandardScaling
# ---------------------------------------------------------------------------

class StandardScaling(ScalingConfig):
    """Vanilla 1/sqrt(fan_in) normal init for all linear/expert weights.

    Embeddings: normal_(0, 0.02).  Biases: zeros.  Norms: skipped (PyTorch defaults).
    """

    @torch.no_grad()
    def init_weight(self, role, param, fan_in, fan_out, config):
        if role == 'norm':
            return  # keep PyTorch defaults (weights to 1, bias to 0)

        if role == 'embedding':
            torch.nn.init.normal_(param, mean=0.0, std=0.02)
            return

        std = (1. / fan_in) ** 0.5
        torch.nn.init.normal_(param, mean=0.0, std=std)

# ---------------------------------------------------------------------------
# MuP for dense model trained with AdamW
# ---------------------------------------------------------------------------

class MupDense(ScalingConfig):
    """muP for dense transformer with AdamW (multiplier-free form).

    All forward-pass scaling is absorbed into init variance and per-parameter LR.

    Last-layer output (lm_head) is initialized to zero.

    Base width: 256 (at this width, behavior matches StandardScaling).
    """

    @torch.no_grad()
    def init_weight(self, role, param, fan_in, fan_out, config):
        if role == 'norm':
            return # keep PyTorch defaults (weights to 1, bias to 0)

        if role == 'embedding':
            torch.nn.init.normal_(param, mean=0.0, std=0.02)
            return

        if role == 'output':
            torch.nn.init.zeros_(param)
            return

        # all hidden matrices: standard 1/sqrt(fan_in)
        std = (1. / fan_in) ** 0.5
        torch.nn.init.normal_(param, mean=0.0, std=std)

    def lr_scale(self, role, config):
        if is_input_like(role):
            return 1.0
        # all matrix params (hidden + output): 1/m
        return 1 / config.n_embd

# ============================================================================
# Fixed-E: M fixed at M_BASE, N and N_expert scale proportionally
# ============================================================================

class MupFixedE(ScalingConfig):
    """muP for fixed experts and trainable router.

    Layer | std      | LR       | eps
    -----------------------------------
    Input | n^0      | n^0      | n^-1
    ------------------------------------
    Hidden| n^-1/2   | n^-1     | n^-1
    ------------------------------------
    Output| n^-1     | n^-1     | n^0

    Router is output-like. Last layer initialized to zero.
    """

    @torch.no_grad()
    def init_weight(self, role, param, fan_in, fan_out, config):
        if role == 'norm':
            return # keep PyTorch defaults (weights to 1, bias to 0)

        if role == 'embedding':
            torch.nn.init.normal_(param, mean=0.0, std=0.02)
            return
        
        if role == "router":
            torch.nn.init.normal_(param, mean=0.0, std=1. / fan_in)
            return

        if role == 'output':
            torch.nn.init.zeros_(param)
            return

        # hidden
        std = (1. / fan_in) ** 0.5
        torch.nn.init.normal_(param, mean=0.0, std=std)

    def lr_scale(self, role, config):
        if is_input_like(role):
            return 1.0
        # all other n^-1
        return self.BASE_WIDTH / config.n_embd
    
    def eps_scale(self, role, config):
        if role == 'output' or role == 'router':
            return 1.0
        
        # all other n^-1
        return 1 / config.n_embd
    
# ============================================================================
# All-scaling variants: M, N, N_expert all scale proportionally
# ============================================================================

class MupAllScale(ScalingConfig):
    """muP for scaling number of experts.

    Layer  | std      | LR       | eps
    ---------------------------------------
    Input  | n^0      | n^0      | n^-1
    ---------------------------------------
    Router | n^-1/2   | n^-1     | m^-1
    ---------------------------------------
    Exp in | n^-1/2   | n^-1     | n^-1m^-1
    ---------------------------------------
    Exp out| n^-1/2   | n^-1     | n^-1m^-1
    ---------------------------------------
    Output | n^-1     | n^-1     | n^0
    ---------------------------------------

    Last layer initialized to zero.
    """

    @torch.no_grad()
    def init_weight(self, role, param, fan_in, fan_out, config):
        if role == 'norm':
            return # keep PyTorch defaults (weights to 1, bias to 0)

        if role == 'embedding':
            torch.nn.init.normal_(param, mean=0.0, std=0.02)
            return

        if role == 'output':
            torch.nn.init.zeros_(param)
            return

        # router & hidden n^-1/2
        std = (1. / fan_in) ** 0.5
        torch.nn.init.normal_(param, mean=0.0, std=std)

    def lr_scale(self, role, config):
        if is_input_like(role):
            return 1.0
        # all other n^-1
        return self.BASE_WIDTH / config.n_embd

    def eps_scale(self, role, config):
        if role == 'output':
            return 1.0

        if role == 'router':
            return 1 / config.n_exp

        if role == 'expert_up' or role == 'expert_down':
            return (1 / config.n_embd) * (1 / config.n_exp)

        # all other n^-1
        return 1 / config.n_embd


class MupBaselineAllScale(MupAllScale):
    """Same as MupAllScale but uses the canonical muP readout init (std = 1/N)
    instead of the zero-init variant. All other scales (LR, eps, forward mults)
    are inherited unchanged.
    """

    @torch.no_grad()
    def init_weight(self, role, param, fan_in, fan_out, config):
        if role == 'output':
            torch.nn.init.normal_(param, mean=0.0, std=1.0 / config.n_embd)
            return
        super().init_weight(role, param, fan_in, fan_out, config)


# ============================================================================
# Bottleneck: M, N scale proportionally, N_expert stays constant
# ============================================================================

class MupBottleneck(ScalingConfig):
    """adapted from MuPBottleneckAdamConfigMultfree
    """

    @torch.no_grad()
    def init_weight(self, role, param, fan_in, fan_out, config):
        if role == 'norm':
            return # keep PyTorch defaults (weights to 1, bias to 0)

        if role == 'embedding':
            torch.nn.init.normal_(param, mean=0.0, std=0.02)
            return

        if role == 'output':
            torch.nn.init.zeros_(param)
            return
        
        if role == 'expert_down': # M^-1/2 scaling term for expert down projection
            torch.nn.init.normal_(param, mean=0.0, std=(config.n_exp / fan_in) ** 0.5)
            return

        # router & hidden
        std = (1. / fan_in) ** 0.5
        torch.nn.init.normal_(param, mean=0.0, std=std)

    def lr_scale(self, role, config):
        if is_input_like(role) or role == 'expert_down':
            return 1.0
        # all other n^-1
        return self.BASE_WIDTH / config.n_embd
    
    def eps_scale(self, role, config):
        if role == 'embedding':
            return (1 / config.n_embd)

        if role == "router" or role == 'expert_up':
            return (1 / config.n_exp)

        if role == 'expert_down':
            return (1 / config.n_embd) * (1 / config.n_exp)

        if role == 'output':
            return 1.0

        # hidden
        return 1 / config.n_embd  # epsilon in anderen layern ist einfach: output layer N**0, else N**(-1)


class MupBottleneckNoExpertDownInit(MupBottleneck):
    """Like MupBottleneck but expert_down init uses standard 1/sqrt(fan_in)
    (no n_exp factor). lr_scale and eps_scale are inherited unchanged.

    At BASE_WIDTH=256 with n_exp=32, the old init std for expert_down is
    sqrt(n_exp/fan_in); the new one is sqrt(1/fan_in). Width-256 equivalence
    with the old config is recovered by setting expert_down_init_mult=sqrt(M)
    (here sqrt(32) ≈ 5.6569).
    """

    @torch.no_grad()
    def init_weight(self, role, param, fan_in, fan_out, config):
        if role == 'expert_down':
            std = (1. / fan_in) ** 0.5
            torch.nn.init.normal_(param, mean=0.0, std=std)
            return
        super().init_weight(role, param, fan_in, fan_out, config)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SCALING_CONFIGS = {
    "standard": StandardScaling,
    "mup_dense": MupDense,
    "mup_fixed_e": MupFixedE,
    "mup_allscale": MupAllScale,
    "mup_baseline_allscale": MupBaselineAllScale,
    "mup_bottleneck": MupBottleneck,
    "mup_bottleneck_no_expert_down_init": MupBottleneckNoExpertDownInit,
}


def get_scaling_config(name: str) -> ScalingConfig:
    """Look up a scaling config by name."""
    if name not in SCALING_CONFIGS:
        available = ', '.join(SCALING_CONFIGS.keys())
        raise ValueError(f"Unknown scaling config: {name!r}. Available: {available}")
    return SCALING_CONFIGS[name]()
