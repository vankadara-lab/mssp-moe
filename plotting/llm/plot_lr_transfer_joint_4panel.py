#!/usr/bin/env python
"""Joint 4-panel LR-transfer figure.

Layout (1×4, left pair shares y-axis, right pair shares y-axis):
  [μP Regime II] [MSSP Regime II] | [μP Regime III] [MSSP Regime III]

Projects:
  Panel 0: effbottleneck_sparse_no_mhalf_lr_transfer   → μP Regime II
  Panel 1: effbottleneck_sparse_lr_transfer             → MSSP Regime II
  Panel 2: allscale_sparse_lr_transfer_untied_tuned_multipliers → μP Regime III
  Panel 3: allscale_sparse_lr_transfer                  → MSSP Regime III
"""

import argparse
import os
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb


ENTITY = os.environ.get("WANDB_ENTITY", "")
OUT_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "llm_lr_transfer"

# Allscale warmup config (shared by both allscale projects)
ALLSCALE_LARGE_WIDTH = 2048
ALLSCALE_WARMUP_AT_LARGE = 2000
ALLSCALE_DEFAULT_WARMUP = 700
ALLSCALE_EXCLUDE_WARMUP = {1400}


PANELS = [
    {
        "project": "effbottleneck_sparse_no_mhalf_lr_transfer",
        "title": r"$\mu$P Regime II",
        "allscale_warmup": False,
    },
    {
        "project": "effbottleneck_sparse_lr_transfer",
        "title": "MSSP Regime II",
        "allscale_warmup": False,
    },
    {
        "project": "allscale_sparse_lr_transfer_untied_tuned_multipliers",
        "title": r"$\mu$P Regime III",
        "allscale_warmup": True,
    },
    {
        "project": "allscale_sparse_lr_transfer",
        "title": "MSSP Regime III",
        "allscale_warmup": True,
    },
]


# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------

def _set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 17.6,
        "axes.labelsize": 19.2,
        "axes.titlesize": 22.2,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------

def _round_lr(lr):
    return float(f"{lr:.6g}")


def fetch_simple(project):
    """data[metric][width][lr] = [values]  (no warmup filtering)."""
    api = wandb.Api()
    data = {m: defaultdict(lambda: defaultdict(list)) for m in ("val/loss", "train/loss")}
    for r in api.runs(f"{ENTITY}/{project}"):
        if r.state != "finished":
            continue
        w = r.config.get("n_embd")
        lr = r.config.get("learning_rate")
        if not (w and lr):
            continue
        lr = _round_lr(lr)
        for m in data:
            v = r.summary.get(m)
            if v is not None:
                data[m][w][lr].append(v)
    return data


def fetch_allscale(project):
    """data[metric][(width, warmup)][lr] = [values]  (warmup=1400 excluded)."""
    api = wandb.Api()
    data = {m: defaultdict(lambda: defaultdict(list)) for m in ("val/loss", "train/loss")}
    for r in api.runs(f"{ENTITY}/{project}"):
        if r.state != "finished":
            continue
        w = r.config.get("n_embd")
        lr = r.config.get("learning_rate")
        ws = r.config.get("warmup_steps")
        if not (w and lr and ws is not None):
            continue
        if ws in ALLSCALE_EXCLUDE_WARMUP:
            continue
        lr = _round_lr(lr)
        for m in data:
            v = r.summary.get(m)
            if v is not None:
                data[m][(w, ws)][lr].append(v)
    return data


def select_allscale_curves(data_by_metric):
    """Pick one warmup variant per width for each metric.
    Returns per_metric[metric][width][lr] = [values], shared LRs from val/loss.
    """
    per_metric = {}
    shared = None
    for m, data in data_by_metric.items():
        widths = sorted({w for (w, _) in data})
        primary = {}
        for w in widths:
            key = (w, ALLSCALE_WARMUP_AT_LARGE) if w == ALLSCALE_LARGE_WIDTH else (w, ALLSCALE_DEFAULT_WARMUP)
            if key in data:
                primary[w] = data[key]
        per_metric[m] = primary
        if m == "val/loss":
            lr_sets = [set(d) for d in primary.values()]
            shared = sorted(set.intersection(*lr_sets)) if lr_sets else []
    return per_metric, shared


def union_lrs(data_by_metric):
    all_lrs = set()
    for by_width in data_by_metric.values():
        for by_lr in by_width.values():
            all_lrs.update(by_lr)
    return sorted(all_lrs)


def load_panel(panel_cfg):
    project = panel_cfg["project"]
    print(f"  fetching {project} …")
    if panel_cfg["allscale_warmup"]:
        raw = fetch_allscale(project)
        per_metric, lrs = select_allscale_curves(raw)
    else:
        per_metric = fetch_simple(project)
        lrs = union_lrs(per_metric)
    print(f"    widths={sorted(per_metric.get('val/loss', {}))}  LRs={len(lrs)}")
    return per_metric, lrs


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------

def draw_panel(ax, data, lrs, show_legend, show_ylabel, ylabel="Validation loss",
               std_bands=False, exclude_widths=(), width_colors=None):
    widths = [w for w in sorted(data) if w not in exclude_widths]
    if width_colors is None:
        colors = {w: c for w, c in zip(widths, plt.cm.Blues(np.linspace(0.3, 0.95, len(widths))))}
    else:
        colors = width_colors

    for w in widths:
        w_lrs = [lr for lr in lrs if lr in data[w]]
        if not w_lrs:
            continue
        means = np.array([np.mean(data[w][lr]) for lr in w_lrs])

        if std_bands:
            stds = np.array([
                np.std(data[w][lr], ddof=1) if len(data[w][lr]) > 1 else np.nan
                for lr in w_lrs
            ])
            ax.fill_between(w_lrs, means - stds, means + stds,
                            color=colors[w], alpha=0.18, linewidth=0, zorder=2)

        ax.plot(w_lrs, means, "o-", color=colors[w],
                linewidth=3.6, markersize=11, label=f"$N={w}$", zorder=3)

        bi = int(np.nanargmin(means))
        ax.plot(w_lrs[bi], means[bi], marker="*", markersize=26,
                color=colors[w], markeredgecolor="black",
                markeredgewidth=1.2, linestyle="none", zorder=4)

    ax.set_xscale("log")
    ax.set_xlabel("Learning rate")
    if show_ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    if show_legend:
        ax.legend(loc="lower left", frameon=True)


def _apply_xticks(ax, ticks):
    ax.set_xscale("log")
    ax.set_xticks(ticks)
    labels = []
    for x in ticks:
        exp = int(np.floor(np.log10(x)))
        coeff = x / (10 ** exp)
        if np.isclose(coeff, 1.0):
            labels.append(rf"$10^{{{exp}}}$")
        else:
            labels.append(rf"${coeff:.1f}\times10^{{{exp}}}$")
    ax.set_xticklabels(labels)
    ax.minorticks_off()


def save_fig(fig, stem):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(p, dpi=200, bbox_inches="tight", pad_inches=0.02)
        print(f"-> {p}")
    plt.close(fig)


def make_joint_figure(panels, metric, stem, std_bands=False, exclude_2048_left=False):
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
    metric_label = "Validation loss" if metric == "val/loss" else "Training loss"
    fig = plt.figure(figsize=(20, 4.84))
    gs = GridSpec(1, 2, figure=fig, wspace=0.12)
    gs_left  = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], wspace=0.04)
    gs_right = GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1], wspace=0.04)
    axes = [
        fig.add_subplot(gs_left[0]),
        fig.add_subplot(gs_left[1]),
        fig.add_subplot(gs_right[0]),
        fig.add_subplot(gs_right[1]),
    ]
    axes[1].sharey(axes[0])
    axes[3].sharey(axes[2])

    # Global colormap: all widths across all panels, same color per width everywhere
    all_widths = sorted({w for pm, _, _ in panels for w in pm.get(metric, {})})
    width_colors = {w: c for w, c in zip(all_widths, plt.cm.Blues(np.linspace(0.3, 0.95, len(all_widths))))}

    for i, (ax, (per_metric, lrs, title)) in enumerate(zip(axes, panels)):
        data = per_metric.get(metric, {})
        show_legend = (i == 0)
        show_ylabel = (i == 0 or i == 2)
        exc = (2048,) if (exclude_2048_left and i in (0, 1)) else ()
        draw_panel(ax, data, lrs, show_legend=show_legend,
                   show_ylabel=show_ylabel, ylabel=metric_label, std_bands=std_bands,
                   exclude_widths=exc, width_colors=width_colors)
        # Add dummy legend entries for excluded widths so legend is always complete
        if show_legend and exc:
            for w in sorted(exc):
                ax.plot([], [], "o-", color=width_colors[w],
                        linewidth=3.6, markersize=11, label=f"$N={w}$")
            ax.legend(loc="lower left", frameon=True)
        ax.set_title(title)
        if lrs:
            _apply_xticks(ax, [lrs[1] if len(lrs) > 1 else lrs[0], lrs[-1]])
        if i in (1, 3):
            ax.tick_params(labelleft=False)

    fig.tight_layout(pad=0.3)

    # Push inner-boundary labels away from each other
    fig.canvas.draw()
    for i, ax in enumerate(axes):
        labels = ax.get_xticklabels()
        if not labels:
            continue
        if i in (0, 2):   # right-align the rightmost label (at inner boundary)
            labels[-1].set_ha("right")
        if i in (1, 3):   # left-align the leftmost label (at inner boundary)
            labels[0].set_ha("left")

    save_fig(fig, stem)


def make_individual_figures(panels, metric, stem_prefix, std_bands=False):
    metric_label = "Validation loss" if metric == "val/loss" else "Training loss"
    for per_metric, lrs, title in panels:
        data = per_metric.get(metric, {})
        fig, ax = plt.subplots(figsize=(8.5, 4.84))
        draw_panel(ax, data, lrs, show_legend=True, show_ylabel=True,
                   ylabel=metric_label, std_bands=std_bands)
        if lrs:
            _apply_xticks(ax, [lrs[0], lrs[-1]])
        ax.set_title(title)
        fig.tight_layout(pad=0.2)
        safe_title = title.replace("$", "").replace("\\", "").replace(" ", "_").lower()
        save_fig(fig, f"{stem_prefix}_{safe_title}")


def share_base_width(panels, pair, width):
    """Merge data for `width` across both panels in `pair` so both show the same curve."""
    i, j = pair
    for metric in ("val/loss", "train/loss"):
        data_i = panels[i][0].get(metric, {})
        data_j = panels[j][0].get(metric, {})
        merged = defaultdict(list)
        for lr, vals in data_i.get(width, {}).items():
            merged[lr].extend(vals)
        for lr, vals in data_j.get(width, {}).items():
            merged[lr].extend(vals)
        if merged:
            data_i[width] = merged
            data_j[width] = merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", type=str, default=ENTITY,
                        help="WandB entity/team name (overrides WANDB_ENTITY env var)")
    parser.add_argument("--std_bands", action="store_true",
                        help="Draw ±1σ shaded bands across seeds")
    parser.add_argument("--exclude_2048_left", action="store_true",
                        help="Exclude N=2048 from the two Regime II (left) subplots")
    args = parser.parse_args()

    global ENTITY
    ENTITY = args.entity

    _set_style()

    panels = []
    for cfg in PANELS:
        per_metric, lrs = load_panel(cfg)
        panels.append((per_metric, lrs, cfg["title"]))

    # Share N=256 (base width) between μP and MSSP panels within each regime
    share_base_width(panels, pair=(0, 1), width=256)

    # Recompute LR union after merging
    for k in range(len(panels)):
        per_metric, _, title = panels[k]
        lrs = union_lrs(per_metric)
        panels[k] = (per_metric, lrs, title)

    suffix = "_std" if args.std_bands else ""

    # For joint figures: each panel independently uses all its LRs within
    # the [min, max] range of its own N=1024 data.
    joint_panels = []
    for pm, _, title in panels:
        n1024_lrs = sorted(pm.get("val/loss", {}).get(1024, {}).keys())
        if n1024_lrs:
            lr_min, lr_max = min(n1024_lrs), max(n1024_lrs)
            lrs = [lr for lr in union_lrs(pm) if lr_min <= lr <= lr_max]
        else:
            lrs = union_lrs(pm)
        print(f"  {title}: [{lrs[0]:.3e}, {lrs[-1]:.3e}]  ({len(lrs)} LRs)")
        joint_panels.append((pm, lrs, title))

    make_joint_figure(joint_panels, "val/loss",   f"lr_transfer_joint_4panel_valloss{suffix}",
                      std_bands=args.std_bands, exclude_2048_left=args.exclude_2048_left)
    make_joint_figure(joint_panels, "train/loss", f"lr_transfer_joint_4panel_trainloss{suffix}",
                      std_bands=args.std_bands, exclude_2048_left=args.exclude_2048_left)
    make_individual_figures(panels, "train/loss",
                            f"lr_transfer_individual_trainloss{suffix}",
                            std_bands=args.std_bands)


if __name__ == "__main__":
    main()
