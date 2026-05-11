
"""
Scaling configurations for MoE models.
Each configuration specifies initialization schemes, learning rate scalings, and forward pass multipliers.
"""
import numpy as np

# Base dimensions
N_BASE = 128
M_BASE = 8
N_EXPERT_BASE = 128 # only for setting correct shapes
N_EXPERT_BASE_BOTTLENECK = 16 # only for setting correct shapes

class ScalingConfig:
    """Base class for scaling configurations"""
    def __init__(self, name, description=""):
        self.name = name
        self.description = description

    def adjust_dimensions(self, N, M, N_expert, base_N=64, width_factor=0.25, expert_factor=16):
        """Adjust M and N_expert based on N for this config. Returns (M, N_expert)."""
        return M, N_expert

    def get_init_scale(self, param_name, N, M, N_expert, fanin, fanout):
        """Return initialization scale for a parameter (multiplies N(0,1) weights)"""
        return 1.0

    def get_forward_scale(self, param_name, N, M, N_expert, fanin, fanout):
        """Return forward pass scale (multiplies weights during forward)"""
        return 1.0 / np.sqrt(fanin)

    def get_lr_scale(self, param_name, N, M, N_expert, base_lr, fanin):
        """Return learning rate scale for a parameter"""
        return base_lr

    def should_init_zero(self, param_name):
        """Whether to initialize parameter to zero"""
        return (param_name=='w_out' and 'mup' in self.name)

    def get_adam_eps_scale(self, param_name, N, M, N_expert):
        """Return Adam epsilon scale for a parameter (default 1.0 = no scaling)"""
        return 1.0

    def get_default_alpha(self):
        """Return default router_fn_scale_alpha for this config"""
        return 0.0


# ============================================================================
# Fixed-E variants: M fixed at M_BASE, N and N_expert scale proportionally
# ============================================================================

class SPFixedEConfig(ScalingConfig):
    """Fixed E with SP: M fixed, N and N_expert scale proportionally"""
    def adjust_dimensions(self, N, M, N_expert, base_N=64, width_factor=0.25, expert_factor=16):
        scale = N / N_BASE
        return M_BASE, int(N_EXPERT_BASE * scale)

    def get_init_scale(self, param_name, N, M, N_expert, fanin, fanout):
        return 1.0 / np.sqrt(fanin)

    def get_forward_scale(self, param_name, N, M, N_expert, fanin, fanout):
        return 1.0

    def get_lr_scale(self, param_name, N, M, N_expert, base_lr, fanin):
        return base_lr / N


class MuPFixedEConfig(ScalingConfig): # should init router to 0, else expect CLT to LLN effects in router prop updates
    """Fixed E with μP multfree: M fixed, N and N_expert scale proportionally"""
    def adjust_dimensions(self, N, M, N_expert, base_N=64, width_factor=0.25, expert_factor=16):
        scale = N / N_BASE
        return M_BASE, int(N_EXPERT_BASE * scale)

    def get_init_scale(self, param_name, N, M, N_expert, fanin, fanout):
        if param_name in ['w_router', 'w_out']:
            return 1.0 / N
        return 1.0 / np.sqrt(fanin)

    def get_forward_scale(self, param_name, N, M, N_expert, fanin, fanout):
        return 1.0

    def get_lr_scale(self, param_name, N, M, N_expert, base_lr, fanin):
        width_factor = N # / N_BASE
        if param_name == 'w_router':
            return base_lr * width_factor**(-1)
        elif param_name == 'w_out':
            return base_lr * width_factor**(-1)
        elif param_name in ['w_expert1', 'w_expert2']:
            return base_lr
        return base_lr * width_factor  # w_in


class NTPFixedEConfig(ScalingConfig):
    """Fixed E with NTP: M fixed, N and N_expert scale proportionally"""
    def adjust_dimensions(self, N, M, N_expert, base_N=64, width_factor=0.25, expert_factor=16):
        scale = N / N_BASE
        return M_BASE, int(N_EXPERT_BASE * scale)

    def get_forward_scale(self, param_name, N, M, N_expert, fanin, fanout):
        return 1.0 / np.sqrt(fanin)

    def get_lr_scale(self, param_name, N, M, N_expert, base_lr, fanin):
        return base_lr


class MuPAdamFixedEConfig(ScalingConfig): # should init router to 0, else expect CLT to LLN effects in router prop updates
    """Fixed E with μP Adam: M fixed, N and N_expert scale proportionally, router scaled like output"""
    def adjust_dimensions(self, N, M, N_expert, base_N=64, width_factor=0.25, expert_factor=16):
        scale = N / N_BASE
        return M_BASE, int(N_EXPERT_BASE * scale)

    def get_init_scale(self, param_name, N, M, N_expert, fanin, fanout):
        if param_name in ['w_router', 'w_out']:
            return 1.0 / N
        return 1.0 / np.sqrt(fanin)

    def get_forward_scale(self, param_name, N, M, N_expert, fanin, fanout):
        return 1.0

    def get_lr_scale(self, param_name, N, M, N_expert, base_lr, fanin):
        return base_lr / fanin

    def get_adam_eps_scale(self, param_name, N, M, N_expert):
        width_factor = N # / N_BASE
        if param_name == 'w_in':
            return width_factor**(-1)
        elif param_name in ['w_expert1', 'w_expert2']:
            return width_factor**(-1)
        return width_factor**0  # w_router, w_out


class MuPAdamFixedEGlobalEpsConfig(ScalingConfig):
    """μP Adam heuristic with all scaling (multfree): M, N, N_expert scale proportionally. Global eps."""
    def adjust_dimensions(self, N, M, N_expert, base_N=64, width_factor=0.25, expert_factor=16):
        scale = N / N_BASE
        return M_BASE, int(N_EXPERT_BASE * scale)

    def get_forward_scale(self, param_name, N, M, N_expert, fanin, fanout):
        return 1.0

    def get_init_scale(self, param_name, N, M, N_expert, fanin, fanout):
        if param_name in ['w_out', 'w_router']:
            return 1 / N
        return 1 / np.sqrt(fanin)

    def get_lr_scale(self, param_name, N, M, N_expert, base_lr, fanin):
        return base_lr / fanin

    def get_adam_eps_scale(self, param_name, N, M, N_expert):
        return 1.0



# ============================================================================
# All-scaling variants: M, N, N_expert all scale proportionally
# ============================================================================

class AllScalingBase(ScalingConfig):
    """Base class for all-scaling configs: M, N, N_expert scale proportionally"""
    def adjust_dimensions(self, N, M, N_expert, base_N=64, width_factor=0.25, expert_factor=16):
        scale = N / N_BASE
        return int(M_BASE * scale), int(N_EXPERT_BASE * scale)

    def get_forward_scale(self, param_name, N, M, N_expert, fanin, fanout):
        return 1.0


class MuPMSSPAllScalingConfig(AllScalingBase):
    """μP LLN with all scaling (multfree): M, N, N_expert scale proportionally"""
    def get_init_scale(self, param_name, N, M, N_expert, fanin, fanout):
        if param_name == 'w_out':
            return 1 / N
        return 1 / np.sqrt(fanin)

    def get_lr_scale(self, param_name, N, M, N_expert, base_lr, fanin):
        width_factor = N # / N_BASE
        expert_factor = M # / M_BASE
        if param_name == 'w_out':
            return base_lr * width_factor**(-1)
        elif param_name == 'w_in':
            return base_lr * width_factor**1
        elif param_name == 'w_router':
            return base_lr * width_factor**0
        return base_lr * expert_factor**1  # experts

    def get_default_alpha(self):
        return 1.0


class MuPAdamAllScalingGlobalEpsConfig(AllScalingBase):
    """μP Adam heuristic with all scaling (multfree): M, N, N_expert scale proportionally. Large router init, but no layerwise epsilon."""
    def get_init_scale(self, param_name, N, M, N_expert, fanin, fanout):
        if param_name in ['w_out']:
            return 1 / N
        return 1 / np.sqrt(fanin)

    def get_lr_scale(self, param_name, N, M, N_expert, base_lr, fanin):
        return base_lr / fanin

    def get_adam_eps_scale(self, param_name, N, M, N_expert):
        return 1.0

    def get_default_alpha(self):
        return 1.0


class NTPAllScalingConfig(AllScalingBase):
    """NTP with all scaling: M, N, N_expert scale proportionally"""
    def get_forward_scale(self, param_name, N, M, N_expert, fanin, fanout):
        return 1.0 / np.sqrt(fanin)

    def get_lr_scale(self, param_name, N, M, N_expert, base_lr, fanin):
        return base_lr

    def get_default_alpha(self):
        return 1.0


class MuPMSSPAdamAllScalingConfig(AllScalingBase):
    """μP Adam with all scaling (ours): M, N, N_expert scale proportionally"""
    def get_init_scale(self, param_name, N, M, N_expert, fanin, fanout):
        if param_name == 'w_out':
            return 1.0 / N
        return 1.0 / np.sqrt(fanin)

    def get_lr_scale(self, param_name, N, M, N_expert, base_lr, fanin):
        return base_lr / fanin

    def get_adam_eps_scale(self, param_name, N, M, N_expert):
        width_factor = N # / N_BASE
        expert_factor = M # / M_BASE
        if param_name == 'w_in':
            return width_factor**(-1)
        elif param_name == 'w_router':
            return expert_factor**(-1)
        elif param_name in ['w_expert1', 'w_expert2']:
            return width_factor**(-1) * expert_factor**(-1)
        return width_factor**0  # w_out

    def get_default_alpha(self):
        return 1.0


# ============================================================================
# Bottleneck variants: M, N scale proportionally, N_expert stays constant
# ============================================================================

class BottleneckBase(ScalingConfig):
    """Base class for bottleneck configs: M, N scale proportionally, N_expert fixed"""
    def adjust_dimensions(self, N, M, N_expert, base_N=64, width_factor=0.25, expert_factor=16):
        scale = N / N_BASE
        return int(M_BASE * scale), N_EXPERT_BASE_BOTTLENECK

    def get_forward_scale(self, param_name, N, M, N_expert, fanin, fanout):
        return 1.0


class MuPBottleneckConfig(BottleneckBase):
    """μP with bottleneck scaling (multfree): M, N scale proportionally, N_expert fixed"""
    def get_init_scale(self, param_name, N, M, N_expert, fanin, fanout):
        if param_name in ['w_out']:
            return 1 / N
        return 1 / np.sqrt(fanin)

    def get_lr_scale(self, param_name, N, M, N_expert, base_lr, fanin):
        width_factor = N # / N_BASE
        expert_factor = M # / M_BASE
        if param_name == 'w_out':
            return base_lr * width_factor**(-1)
        elif param_name == 'w_in':
            return base_lr * width_factor**1
        elif param_name in ['w_router', 'w_expert1']:
            return base_lr * expert_factor / width_factor
        elif param_name == 'w_expert2':
            return base_lr * expert_factor * width_factor
        raise ValueError(f'{param_name} not specified in scaling config.')

    def get_default_alpha(self):
        return 1.0

class MSSPBottleneckConfig(BottleneckBase):
    """μP with bottleneck scaling (multfree): M, N scale proportionally, N_expert fixed"""
    def get_init_scale(self, param_name, N, M, N_expert, fanin, fanout):
        if param_name in ['w_out']:
            return 1 / N
        elif param_name == 'w_expert2':
            return np.sqrt(M) / np.sqrt(fanin)
        return 1 / np.sqrt(fanin)

    def get_lr_scale(self, param_name, N, M, N_expert, base_lr, fanin):
        width_factor = N # / N_BASE
        expert_factor = M # / M_BASE
        if param_name == 'w_out':
            return base_lr * width_factor**(-1)
        elif param_name == 'w_in':
            return base_lr * width_factor**1
        elif param_name in ['w_router', 'w_expert1']:
            return base_lr * expert_factor / width_factor
        elif param_name == 'w_expert2':
            return base_lr * expert_factor * width_factor
        raise ValueError(f'{param_name} not specified in scaling config.')

    def get_default_alpha(self):
        return 1.0


class MSSPAdamBottleneckConfig(BottleneckBase):
    """μP with bottleneck scaling (multfree): M, N scale proportionally, N_expert fixed"""
    def get_init_scale(self, param_name, N, M, N_expert, fanin, fanout):
        if param_name in ['w_out']:
            return 1 / N
        elif param_name == 'w_expert2':
            return np.sqrt(M) / np.sqrt(fanin)
        return 1 / np.sqrt(fanin)

    def get_lr_scale(self, param_name, N, M, N_expert, base_lr, fanin):
        return base_lr / fanin

    def get_adam_eps_scale(self, param_name, N, M, N_expert):
        width_factor = N # / N_BASE
        expert_factor = M # / M_BASE
        if param_name == 'w_in':
            return width_factor ** (-1)
        elif param_name == 'w_router':
            return expert_factor ** (-1)
        elif param_name == 'w_expert1':
            return expert_factor ** (-1)
        elif param_name == 'w_expert2':
            return (width_factor * expert_factor) ** (-1)
        elif param_name == 'w_out':
            return 1.0

    def get_default_alpha(self):
        return 1.0

class MuPAdamBottleneckConfig(BottleneckBase):
    """μP with bottleneck scaling (multfree): M, N scale proportionally, N_expert fixed"""
    def get_init_scale(self, param_name, N, M, N_expert, fanin, fanout):
        if param_name in ['w_out']:
            return 1 / N
        return 1 / np.sqrt(fanin)

    def get_lr_scale(self, param_name, N, M, N_expert, base_lr, fanin):
        return base_lr / fanin

    def get_adam_eps_scale(self, param_name, N, M, N_expert):
        width_factor = N # / N_BASE
        expert_factor = M # / M_BASE
        if param_name == 'w_in':
            return width_factor ** (-1)
        elif param_name == 'w_router':
            return expert_factor ** (-1)
        elif param_name == 'w_expert1':
            return expert_factor ** (-1)
        elif param_name == 'w_expert2':
            return (width_factor * expert_factor) ** (-1)
        elif param_name == 'w_out':
            return 1.0

    def get_default_alpha(self):
        return 1.0

class MuPAdamBottleneckGlobalEpsConfigMultfree(BottleneckBase):
    """μP with bottleneck scaling (multfree): M, N scale proportionally, N_expert fixed"""
    def get_init_scale(self, param_name, N, M, N_expert, fanin, fanout):
        if param_name in ['w_out']:
            return 1 / N
        return 1 / np.sqrt(fanin)

    def get_lr_scale(self, param_name, N, M, N_expert, base_lr, fanin):
        return base_lr / fanin

    def get_adam_eps_scale(self, param_name, N, M, N_expert):
        return 1.0

    def get_default_alpha(self):
        return 1.0


class SPBottleneckConfig(BottleneckBase):
    """SP with bottleneck scaling: M, N scale proportionally, N_expert fixed
    Base: N=128, M=128, N_expert=16
    """

    def get_init_scale(self, param_name, N, M, N_expert, fanin, fanout):
        return 1 / np.sqrt(fanin)

    def get_lr_scale(self, param_name, N, M, N_expert, base_lr, fanin):
        width_factor = N / 128  # Base N=128
        return base_lr * width_factor**(-1)

    def get_default_alpha(self):
        return 1.0


# CAREFUL WHEN CHANGING THE BELOW KEYS. MULTIPLIER SETTING IN moe_training.py REQUIRES THESE KEYS
# Registry of all configurations
CONFIGS = {
    # Fixed-E variants
    'fixed_E_sp': SPFixedEConfig('fixed_E_sp'),
    'fixed_E_mup_multfree': MuPFixedEConfig('fixed_E_mup_multfree'), # ours SGD (ours: Q=0, mup baseline: Q init=1/N)
    'fixed_E_ntp': NTPFixedEConfig('fixed_E_ntp'),
    'fixed_E_mup_adam': MuPAdamFixedEConfig('fixed_E_mup_adam'), # ours Adam (ours: Q=0, mup baseline: Q init=1/N)
    'fixed_E_mup_adam_globaleps': MuPAdamFixedEGlobalEpsConfig('fixed_E_mup_adam_globaleps'), # Adam second baseline, heuristic is correct here, but does not give epsilon scaling

    # Bottleneck variants
    'mup_bottleneck_stdinit': MuPBottleneckConfig('mup_bottleneck_stdinit'), # baseline SGD
    'mup_bottleneck_ours': MSSPBottleneckConfig('mup_bottleneck_ours'), # ours SGD (sqrt(M/N_expert) expert2 init)
    'mup_bottleneck_adam_multfree': MSSPAdamBottleneckConfig('mup_bottleneck_adam_multfree'), # ours Adam (uses sqrt(M/N_expert) expert2 init)
    'mup_bottleneck_adam_stdinit_multfree': MuPAdamBottleneckConfig('mup_bottleneck_adam_stdinit_multfree'), # Adam baseline
    'mup_bottleneck_adam_globaleps_multfree': MuPAdamBottleneckGlobalEpsConfigMultfree('mup_bottleneck_adam_globaleps_multfree'), # potential second baseline Adam
    'sp_bottleneck': SPBottleneckConfig('sp_bottleneck'),

    # All-scaling variants
    'mup_stdinit_allscaling_multfree': MuPMSSPAllScalingConfig('mup_stdinit_allscaling_multfree'), # SGD mup baseline and ours when sharing expert init
    'ntp_allscaling': NTPAllScalingConfig('ntp_allscaling'),
    'mup_adam_allscaling_stdinit_ours': MuPMSSPAdamAllScalingConfig('mup_adam_allscaling_stdinit_ours'), # Adam mup baseline and ours when sharing expert init
    'mup_adam_globaleps_allscaling_multfree': MuPAdamAllScalingGlobalEpsConfig('mup_adam_globaleps_allscaling_multfree'), # Adam second baseline
}

def get_config(name):
    """Get a scaling configuration by name"""
    if name not in CONFIGS:
        raise ValueError(f"Unknown config: {name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[name]


