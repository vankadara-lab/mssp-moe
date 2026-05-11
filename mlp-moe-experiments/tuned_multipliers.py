"""
Optimal Hyperparameters
"""

TUNED_MULTIPLIERS = {
    # SGD Configs (Base LR: 0.04)
    'fixed_E_mup_multfree': {
        'soft': {
            'base_lr': 0.04,
            'init_std_mult': 1.0,
            'lr_mult_in': 0.016,
            'lr_mult_out': 256.0,
            'lr_mult_router': 1.0,
            'lr_mult_expert1': 1.0,
            'lr_mult_expert2': 1.0,
            'best_acc': 0.3003,
        },
        'topk': {
            'base_lr': 0.04,
            'init_std_mult': 1.0,
            'lr_mult_in': 0.016,
            'lr_mult_out': 256.0,
            'lr_mult_router': 1.0,
            'lr_mult_expert1': 1.0,
            'lr_mult_expert2': 1.0,
            'best_acc': 0.3003,
        }
    },
    'mup_bottleneck_stdinit': {
        'soft_ll1overN': {
            'base_lr': 0.01,
            'init_std_mult': 0.25,
            'lr_mult_in': 0.00390625,
            'lr_mult_out': 1048576.0,
            'lr_mult_router': 16.0,
            'lr_mult_expert1': 64.0,
            'lr_mult_expert2': 0.00390625,
            'best_acc': 0.296,  # val_top5_acc (TinyImageNet)
        },
        'soft': {
            'base_lr': 0.01,
            'init_std_mult': 0.25,
            'lr_mult_in': 0.0039,
            'lr_mult_out': 65536.0,
            'lr_mult_router': 64.0,
            'lr_mult_expert1': 1024.0,
            'lr_mult_expert2': 0.015625,
            'best_acc': 0.2865,  # val_top5_acc (TinyImageNet)
        },
        'topk': {
            'base_lr': 0.01,
            'init_std_mult': 1.0,
            'lr_mult_in': 0.0039,
            'lr_mult_out': 256.0,
            'lr_mult_router': 64.0,
            'lr_mult_expert1': 4096.0,
            'lr_mult_expert2': 0.25,
            'best_acc': 0.2668,
        }
    },
    'mup_stdinit_allscaling_multfree': {
        'soft_no_shared': {
            'base_lr': 0.04,
            'init_std_mult': 1.0,
            'lr_mult_in': 1.0,
            'lr_mult_out': 0.25,
            'lr_mult_router': 64.0,
            'lr_mult_expert1': 1.0,
            'lr_mult_expert2': 1.0,
            'best_acc': 0.2679,
        },
        'soft_no_shared_ll1overN': {
            'base_lr': 0.04,
            'init_std_mult': 1.0,
            'lr_mult_in': 0.0625,
            'lr_mult_out': 4.0,
            'lr_mult_router': 4096.0,
            'lr_mult_expert1': 0.25,
            'lr_mult_expert2': 0.0625,
            'best_acc': 0.291,  # val_top5_acc (TinyImageNet)
        },
        'topk_no_shared': {
            'base_lr': 0.04,
            'init_std_mult': 1.0,
            'lr_mult_in': 1.0,
            'lr_mult_out': 0.25,
            'lr_mult_router': 64.0,
            'lr_mult_expert1': 1.0,
            'lr_mult_expert2': 1.0,
            'best_acc': 0.2679,
        },
        'soft': {
            'base_lr': 0.04,
            'init_std_mult': 1.0,
            'lr_mult_in': 4.0,
            'lr_mult_out': 1.0,
            'lr_mult_router': 16.0,
            'lr_mult_expert1': 0.0625,
            'lr_mult_expert2': 0.25,
            'best_acc': 0.2880,  # val_top5_acc (TinyImageNet)
        },
        'topk': {
            'base_lr': 0.04,
            'init_std_mult': 1.0,
            'lr_mult_in': 0.0625,
            'lr_mult_out': 4.0,
            'lr_mult_router': 4.0,
            'lr_mult_expert1': 1.0,
            'lr_mult_expert2': 0.0625,
            'best_acc': 0.2560,  # val_top5_acc (TinyImageNet), val_acc (top-1) = 0.1050
        }
    },
    'mup_bottleneck_ours': {
        'soft': {
            'base_lr': 0.01,
            'init_std_mult': 0.25,
            'lr_mult_in': 0.015625,
            'lr_mult_out': 65536.0,
            'lr_mult_router': 4.0,
            'lr_mult_expert1': 1.0,
            'lr_mult_expert2': 0.015625,
            'best_acc': 0.3135,  # val_top5_acc (TinyImageNet)
        },
        'topk': {
            'base_lr': 0.01,
            'init_std_mult': 0.25,
            'lr_mult_in': 0.015625,
            'lr_mult_out': 65536.0,
            'lr_mult_router': 4.0,
            'lr_mult_expert1': 1.0,
            'lr_mult_expert2': 0.015625,
            'best_acc': 0.3135,  # val_top5_acc (TinyImageNet)
        }
    },

    # Adam Configs (Base LR: 0.16 - 4× higher!)
    'mup_adam_allscaling_stdinit_ours': {
        'soft_no_shared': {
            'base_lr': 0.16,
            'init_std_mult': 0.25,
            'lr_mult_in': 1.0,
            'lr_mult_out': 4.0,
            'lr_mult_router': 1.0,
            'lr_mult_expert1': 1.0,
            'lr_mult_expert2': 1.0,
            'best_acc': 0.2942,
        },
        'soft_no_shared_ll1overN': {
            'base_lr': 0.16,
            'init_std_mult': 0.25,
            'lr_mult_in': 1.0,
            'lr_mult_out': 4.0,
            'lr_mult_router': 1.0,
            'lr_mult_expert1': 4.0,
            'lr_mult_expert2': 0.0625,
            'best_acc': 0.3265,  # val_top5_acc (TinyImageNet)
        },
        'topk_no_shared': {
            'base_lr': 0.16,
            'init_std_mult': 0.25,
            'lr_mult_in': 1.0,
            'lr_mult_out': 4.0,
            'lr_mult_router': 1.0,
            'lr_mult_expert1': 1.0,
            'lr_mult_expert2': 1.0,
            'best_acc': 0.2942,
        },
        'soft': {
            'base_lr': 0.16,
            'init_std_mult': 0.25,
            'lr_mult_in': 4.0,
            'lr_mult_out': 64.0,
            'lr_mult_router': 0.25,
            'lr_mult_expert1': 0.25,
            'lr_mult_expert2': 0.00390625,
            'best_acc': 0.3255,  # val_top5_acc (TinyImageNet)
        },
        'topk': {
            'base_lr': 0.16,
            'init_std_mult': 0.25,
            'lr_mult_in': 4.0,
            'lr_mult_out': 16.0,
            'lr_mult_router': 0.25,
            'lr_mult_expert1': 0.0625,
            'lr_mult_expert2': 0.25,
            'best_acc': 0.2950,  # val_top5_acc (TinyImageNet), val_acc (top-1) = 0.1210
        }
    },
    'mup_adam_globaleps_allscaling_multfree': {
        'soft': {
            'base_lr': 0.16,
            'init_std_mult': 1.0,
            'lr_mult_in': 1.0,
            'lr_mult_out': 4.0,
            'lr_mult_router': 1.0,
            'lr_mult_expert1': 1.0,
            'lr_mult_expert2': 1.0,
            'best_acc': 0.2942,
        },
        'topk': {
            'base_lr': 0.16,
            'init_std_mult': 1.0,
            'lr_mult_in': 1.0,
            'lr_mult_out': 4.0,
            'lr_mult_router': 1.0,
            'lr_mult_expert1': 1.0,
            'lr_mult_expert2': 1.0,
            'best_acc': 0.2942,
        }
    },
    'fixed_E_mup_adam': {
        'soft': {
            'base_lr': 0.16,
            'init_std_mult': 1.0,
            'lr_mult_in': 4.0,
            'lr_mult_out': 1.0,
            'lr_mult_router': 1.0,
            'lr_mult_expert1': 1.0,
            'lr_mult_expert2': 1.0,
            'best_acc': 0.2930,
        },
        'topk': {
            'base_lr': 0.16,
            'init_std_mult': 1.0,
            'lr_mult_in': 4.0,
            'lr_mult_out': 1.0,
            'lr_mult_router': 1.0,
            'lr_mult_expert1': 1.0,
            'lr_mult_expert2': 1.0,
            'best_acc': 0.2930,
        }
    },
    'mup_bottleneck_adam_stdinit_multfree': {
        'soft_ll1overN': {
            'base_lr': 0.16,
            'init_std_mult': 0.25,
            'lr_mult_in': 4.0,
            'lr_mult_out': 64.0,
            'lr_mult_router': 0.25,
            'lr_mult_expert1': 0.0625,
            'lr_mult_expert2': 0.0625,
            'best_acc': 0.317,  # val_top5_acc (TinyImageNet)
        },
        'soft': {
            'base_lr': 0.16,
            'init_std_mult': 0.25,
            'lr_mult_in': 4.0,
            'lr_mult_out': 4.0,
            'lr_mult_router': 4.0,
            'lr_mult_expert1': 0.25,
            'lr_mult_expert2': 0.00390625,
            'best_acc': 0.3120,  # val_top5_acc (TinyImageNet)
        },
        'topk': {
            'base_lr': 0.16,
            'init_std_mult': 0.25,
            'lr_mult_in': 4.0,
            'lr_mult_out': 4.0,
            'lr_mult_router': 4.0,
            'lr_mult_expert1': 0.25,
            'lr_mult_expert2': 0.00390625,
            'best_acc': 0.3120,  # val_top5_acc (TinyImageNet)
        }
    },
    'mup_bottleneck_adam_globaleps_multfree': {
        'soft': {
            'base_lr': 0.16,
            'init_std_mult': 1.0,
            'lr_mult_in': 1.0,
            'lr_mult_out': 0.25,
            'lr_mult_router': 16.0,
            'lr_mult_expert1': 1.0,
            'lr_mult_expert2': 1.0,
            'best_acc': 0.2730,
        },
        'topk': {
            'base_lr': 0.16,
            'init_std_mult': 1.0,
            'lr_mult_in': 1.0,
            'lr_mult_out': 0.25,
            'lr_mult_router': 16.0,
            'lr_mult_expert1': 1.0,
            'lr_mult_expert2': 1.0,
            'best_acc': 0.2730,
        }
    },
    'mup_bottleneck_adam_multfree': {
        'soft': {
            'base_lr': 0.16,
            'init_std_mult': 0.25,
            'lr_mult_in': 4.0,
            'lr_mult_out': 4.0,
            'lr_mult_router': 4.0,
            'lr_mult_expert1': 0.25,
            'lr_mult_expert2': 0.00390625,
            'best_acc': 0.3267,  # val_top5_acc (TinyImageNet)
        },
        'topk': {
            'base_lr': 0.16,
            'init_std_mult': 0.25,
            'lr_mult_in': 4.0,
            'lr_mult_out': 4.0,
            'lr_mult_router': 4.0,
            'lr_mult_expert1': 0.25,
            'lr_mult_expert2': 0.00390625,
            'best_acc': 0.3267,  # val_top5_acc (TinyImageNet)
        }
    },
}


def get_optimal_values(config_name, routing_mode, shared_experts=True, last_layer_init=None):
    """Return optimal hyperparameters for *config_name* and *routing_mode*.

    Key lookup priority (most specific to least):
      1. {routing_mode}_no_shared_ll1overN  (if not shared_experts and last_layer_init='mup')
      2. {routing_mode}_ll1overN            (if last_layer_init='mup')
      3. {routing_mode}_no_shared           (if not shared_experts)
      4. {routing_mode}                     (fallback)
    """
    if config_name not in TUNED_MULTIPLIERS:
        return None
    config_opts = TUNED_MULTIPLIERS[config_name]
    ll1overN = last_layer_init == 'mup'

    if not shared_experts and ll1overN:
        key = f'{routing_mode}_no_shared_ll1overN'
        if key in config_opts:
            return config_opts[key]
    if ll1overN:
        key = f'{routing_mode}_ll1overN'
        if key in config_opts:
            return config_opts[key]
    if not shared_experts:
        key = f'{routing_mode}_no_shared'
        if key in config_opts:
            return config_opts[key]
    return config_opts.get(routing_mode)
