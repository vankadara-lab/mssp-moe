"""
Canonical label and title utilities for all MoE scaling plots.

Import from this module instead of defining local copies:

    from label_utils import (
        LABEL_MAP, format_label, get_optimizer_from_config,
        get_scaling_regime, format_subplot_title, format_latex_caption,
        get_figure_title,
    )
"""

import os

# ---------------------------------------------------------------------------
# Regime strings  (Unicode, no LaTeX math mode)
#   ∼  U+223C  TILDE OPERATOR  (renders as \asymp approximation, widely supported)
#   →  U+2192  RIGHTWARDS ARROW
#   ∞  U+221E  INFINITY
#   ₑ  U+2091  LATIN SUBSCRIPT SMALL LETTER E
# ---------------------------------------------------------------------------
REGIME_FIXED_E      = 'N, N\u2091 \u223c n \u2192 \u221e'   # N, Nₑ ∼ n → ∞
REGIME_BOTTLENECK   = 'M, N \u223c n \u2192 \u221e'          # M, N ∼ n → ∞
REGIME_ALLSCALING   = 'M, N, N\u2091 \u223c n \u2192 \u221e' # M, N, Nₑ ∼ n → ∞

# ---------------------------------------------------------------------------
# Canonical label map  (config key → short display name)
# ---------------------------------------------------------------------------
LABEL_MAP = {
    # Fixed-E
    'fixed_E_sp':                            'SP',
    'fixed_E_mup_multfree':                  'MSSP (ours)',
    'fixed_E_ntp':                           'NTP',
    'fixed_E_mup_adam':                      'MSSP (ours)',
    'fixed_E_mup_adam_globaleps':            'μP, Global Adam ε',
    # All-scaling
    'mup_stdinit_allscaling_multfree':       'MSSP (ours)',
    'ntp_allscaling':                        'NTP',
    'mup_adam_allscaling_stdinit_ours':      'MSSP (ours)',
    'mup_adam_globaleps_allscaling_multfree':'μP, Global Adam ε',
    # Bottleneck
    'sp_bottleneck':                         'SP',
    'mup_bottleneck_stdinit':                'μP baseline',
    'mup_bottleneck_ours':                   'MSSP (ours)',
    'mup_bottleneck_adam_multfree':          'MSSP (ours)',
    'mup_bottleneck_adam_stdinit_multfree':  'μP baseline',
    'mup_bottleneck_adam_globaleps_multfree':'μP, Global Adam ε',
}


def format_label(text):
    """Return display label for a config name, stripping routing/init suffixes."""
    base_text = text
    routing_suffix = ''
    init_suffix = ''

    # Routing suffix (_soft, _k2, …)
    if '_soft' in text:
        base_text = text.replace('_soft', '')
        routing_suffix = ' (soft)'
    elif '_k' in text:
        parts = text.split('_')
        for i, part in enumerate(parts):
            if part.startswith('k') and part[1:].isdigit():
                base_text = '_'.join(parts[:i] + parts[i + 1:])
                routing_suffix = f' (top-{part[1:]})'
                break

    # Router-init suffix (_rinitzero, _rinitmup, …)
    if '_rinit' in base_text:
        parts = base_text.split('_rinit')
        base_text = parts[0]
        rinit_type = parts[1] if len(parts) > 1 else ''
        if rinit_type == 'zero':
            init_suffix = ' [router init=0]'
        elif rinit_type == 'mup':
            init_suffix = ' [router init=1/N]'
        elif rinit_type == 'ntp':
            init_suffix = ' [router init=1/\u221aN]'
        else:
            init_suffix = f' [router init={rinit_type}]'

    label = LABEL_MAP.get(base_text)
    if label is None:
        label = base_text.replace('standard', 'NTP').replace('sp', 'NTP').replace('_', ' ').title()

    return label + routing_suffix + init_suffix


def get_optimizer_from_config(config_name, metadata=None):
    """Infer optimizer from config name. Returns display string: 'Adam' or 'SGD'."""
    if 'adam' in config_name.lower():
        return 'Adam'
    return 'SGD'


def get_scaling_regime(config_name):
    """Return the Unicode regime string for a config."""
    name = config_name.lower()
    if 'fixed_e' in name:
        return REGIME_FIXED_E
    if 'bottleneck' in name:
        return REGIME_BOTTLENECK
    if 'allscaling' in name:
        return REGIME_ALLSCALING
    return 'Other'


def format_subplot_title(config_name, metadata=None, include_regime=True):
    """Return '{label} (OPTIMIZER, regime)' title string."""
    base_label = format_label(config_name)
    optimizer = get_optimizer_from_config(config_name, metadata)
    if include_regime:
        regime = get_scaling_regime(config_name)
        return f'{base_label} ({optimizer}, {regime})'
    return f'{base_label} ({optimizer})'


def get_figure_title(config_name, results_dir=None):
    """Return the base figure title for a config, matching the joint-RCC suptitle logic.

    Applies directory-specific overrides so that 'MSSP (ours)' becomes 'μP' when the
    directory context indicates a run without shared experts (allscaling) or a fixed-E
    run without the llzero+rinitzero flags.
    """
    _d = os.path.basename(os.path.normpath(results_dir)) if results_dir else ''

    _is_allscaling = any(x in config_name for x in ['allscaling', 'allscale'])
    _is_fixed_E_mup = 'fixed_e' in config_name.lower()

    if _is_allscaling:
        _in_allscaling_dir = any(x in _d for x in ['allscaling', 'allscale'])
        # shared experts indicated by 'sharedexp' (RCC dirs) or 'bothshared' (LR sweep dirs)
        _no_shared = _in_allscaling_dir and 'sharedexp' not in _d and 'bothshared' not in _d
        if _no_shared:
            optimizer = get_optimizer_from_config(config_name)
            regime = get_scaling_regime(config_name)
            return f'μP ({optimizer}, {regime})'
    elif _is_fixed_E_mup:
        # MSSP = zero router init; μP = mup router init (routing mode doesn't matter)
        _has_rinitzero = (
            'rinitzero' in config_name.lower() or
            'rinitzero' in _d or 'routerzeroinit' in _d
        )
        _is_mssp = _has_rinitzero
        if not _is_mssp:
            optimizer = get_optimizer_from_config(config_name)
            regime = get_scaling_regime(config_name)
            return f'μP ({optimizer}, {regime})'

    return format_subplot_title(config_name)


def format_latex_caption(config_name, metadata=None):
    """Return a LaTeX-formatted bold caption string."""
    base_label = format_label(config_name)
    optimizer = get_optimizer_from_config(config_name, metadata)
    regime = get_scaling_regime(config_name)
    return f'\\textbf{{{base_label} ({optimizer}, {regime})}}'
