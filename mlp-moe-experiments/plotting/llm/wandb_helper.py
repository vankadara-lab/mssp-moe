import os
import wandb
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import re


ENTITY = os.environ.get("WANDB_ENTITY", "")


def fetch_runs(project: str, entity: str = ENTITY, filters: dict | None = None):
    """Return all runs from a wandb project.

    Parameters
    ----------
    project : str
        Project name (without entity prefix).
    entity : str
        Wandb entity / team.
    filters : dict, optional
        Mongo-style filter passed to ``api.runs()``.
    """
    api = wandb.Api()
    return api.runs(f"{entity}/{project}", filters=filters or {})


def get_runs_and_widths(project: str, entity: str = ENTITY,
                        width_key: str = "n_embd",
                        filters: dict | None = None):
    """Return (runs, widths) — two lists sorted by width ascending.

    Runs whose config is missing *width_key* are skipped.
    """
    pairs = []
    for run in fetch_runs(project, entity=entity, filters=filters):
        if run.state != "finished":
            continue
        width = run.config.get(width_key)
        if width is not None:
            pairs.append((run, width))
    pairs.sort(key=lambda x: x[1])
    runs = [r for r, _ in pairs]
    widths = [w for _, w in pairs]
    return runs, widths


def get_runs_by_width_and_seed(project: str, entity: str = ENTITY,
                                 width_key: str = "n_embd",
                                 seed_key: str = "seed",
                                 filters: dict | None = None):
    """Return runs grouped by (width, seed).

    Returns
    -------
    width_seed_runs : dict
        Dictionary mapping (width, seed) -> run
    widths : list
        Sorted unique widths
    seeds : list
        Sorted unique seeds
    """
    from collections import defaultdict

    width_seed_runs = {}
    all_widths = set()
    all_seeds = set()

    for run in fetch_runs(project, entity=entity, filters=filters):
        if run.state != "finished":
            continue
        width = run.config.get(width_key)
        seed = run.config.get(seed_key)
        if width is not None and seed is not None:
            width_seed_runs[(width, seed)] = run
            all_widths.add(width)
            all_seeds.add(seed)

    widths = sorted(all_widths)
    seeds = sorted(all_seeds)

    return width_seed_runs, widths, seeds


def get_final_metric(run, key: str = "val/loss"):
    """Return the last logged value of *key* for a run, or NaN."""
    history = list(run.scan_history(keys=[key]))
    if history and key in history[-1]:
        return history[-1][key]
    return np.nan


def _get_metric_at_step(run, key: str, step: int):
    """Return the value of *key* at the given training *step*, or NaN.

    Uses the logged row whose ``_step`` is closest to *step* without
    exceeding it.
    """
    steps, values = get_metric_history(run, key)
    if len(steps) == 0:
        return np.nan
    idx = np.searchsorted(steps, step, side="right") - 1
    if idx < 0:
        return np.nan
    return values[idx]


def plot_lr_sweep(project: str, entity: str = ENTITY,
                  group_by: str = "n_embd",
                  loss_key: str = "val/loss",
                  lr_key: str = "learning_rate",
                  filters: dict | None = None,
                  ax: plt.Axes | None = None,
                  y_max: float | None = None,
                  all_steps: bool = False):
    """Plot final validation loss vs learning rate from a wandb LR sweep.

    Each unique value of *group_by* (default: width ``n_embd``) gets its own
    line so you can overlay multiple widths.

    Parameters
    ----------
    project : str
        Wandb project name (without entity).
    entity : str
        Wandb entity / team.
    group_by : str
        Config key used to group runs into separate lines (e.g. ``"n_embd"``).
    loss_key : str
        Logged metric key for validation loss.
    lr_key : str
        Config key for the max learning rate.
    filters : dict, optional
        Mongo-style filter forwarded to ``wandb.Api().runs()``.
    ax : matplotlib Axes, optional
        Axes to plot on.  Created if ``None``.
    y_max : float, optional
        Upper y-axis limit.
    all_steps : bool
        If True, overlay val-loss-vs-LR curves for every available eval step.
        Earlier steps are drawn as dashed lines with lower alpha; the final
        curve is always solid on top.

    Returns
    -------
    fig, ax
    """
    runs = list(fetch_runs(project, entity=entity, filters=filters))

    # group_value -> {lr: [loss, ...]}  (for the final metric)
    groups = defaultdict(lambda: defaultdict(list))
    # group_value -> {step: {lr: [loss, ...]}}  (for all_steps)
    step_groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for run in runs:
        cfg = run.config
        lr = cfg.get(lr_key)
        group_val = cfg.get(group_by)
        if lr is None or group_val is None:
            continue

        if all_steps:
            # Fetch full history once per run
            steps, values = get_metric_history(run, loss_key)
            if len(steps) == 0:
                continue
            groups[group_val][lr].append(values[-1])
            for s, v in zip(steps[:-1], values[:-1]):
                if int(s) == 0:
                    continue
                step_groups[group_val][int(s)][lr].append(v)
        else:
            final_loss = get_final_metric(run, key=loss_key)
            groups[group_val][lr].append(final_loss)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    for group_val in sorted(groups):
        data = groups[group_val]
        lrs = sorted(data)
        means = [np.nanmean(data[lr]) for lr in lrs]

        # --- earlier-step curves (background, dashed) ---
        if all_steps and step_groups[group_val]:
            sorted_steps = sorted(step_groups[group_val].keys())
            # Include the final step in the colormap range
            all_plot_steps = sorted_steps  # earlier steps only; final added below
            n_total = len(all_plot_steps) + 1  # +1 for the final curve
            cmap = plt.cm.viridis
            colors = [cmap(i / max(n_total - 1, 1)) for i in range(n_total)]

            for si, step in enumerate(sorted_steps):
                step_data = step_groups[group_val][step]
                s_lrs = sorted(step_data)
                s_means = [np.nanmean(step_data[lr]) for lr in s_lrs]
                ax.plot(s_lrs, s_means, '--', color=colors[si], linewidth=1,
                        label=f"{group_by}={group_val}, step={step}")
                best_si = int(np.nanargmin(s_means))
                ax.plot(s_lrs[best_si], s_means[best_si], '*',
                        markersize=10, color=colors[si], zorder=5)

            # main curve (solid, on top) — darkest color
            ax.plot(lrs, means, 'o-', color=colors[-1], linewidth=2,
                    markersize=6, label=f"{group_by}={group_val} (final)")
        else:
            ax.plot(lrs, means, 'o-', label=f"{group_by}={group_val}")

        # highlight minimum
        best_idx = int(np.nanargmin(means))
        best_lr, best_loss = lrs[best_idx], means[best_idx]
        ax.plot(best_lr, best_loss, '*', markersize=14, color='red', zorder=5)
        print(f"{group_by}={group_val}: best LR={best_lr:.2e}, final val loss={best_loss:.4f}")

    if y_max is not None:
        ax.set_ylim(top=y_max)
    ax.set_xscale("log")
    ax.set_xlabel("Max Learning Rate")
    ax.set_ylabel("Val Loss")
    ax.set_title(f"LR Sweep — {entity}/{project}")
    ax.legend()

    return fig, ax


def plot_avg_train_loss(project: str, entity: str = ENTITY,
                        group_by: str = "n_embd",
                        loss_key: str = "train/loss",
                        lr_key: str = "learning_rate",
                        n_steps: int = 10,
                        filters: dict | None = None,
                        ax: plt.Axes | None = None,
                        y_max: float | None = None):
    """Plot average training loss over the first *n_steps* steps vs LR.

    Same visual style as :func:`plot_lr_sweep` (one curve per width, red
    star at the optimum).

    Parameters
    ----------
    project, entity, group_by, lr_key, filters, ax, y_max
        Same as :func:`plot_lr_sweep`.
    loss_key : str
        Logged metric key for training loss.
    n_steps : int
        Number of initial training steps to average over.

    Returns
    -------
    fig, ax
    """
    runs = list(fetch_runs(project, entity=entity, filters=filters))

    # group_value -> {lr: [avg_loss, ...]}
    groups = defaultdict(lambda: defaultdict(list))

    for run in runs:
        cfg = run.config
        lr = cfg.get(lr_key)
        group_val = cfg.get(group_by)
        if lr is None or group_val is None:
            continue
        steps, values = get_metric_history(run, loss_key)
        if len(values) == 0:
            continue
        mask = steps <= steps[0] + n_steps - 1 if len(steps) > 0 else []
        n = max(np.sum(mask), 1)
        avg = np.mean(values[:n])
        groups[group_val][lr].append(avg)

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    for group_val in sorted(groups):
        data = groups[group_val]
        lrs = sorted(data)
        means = [np.nanmean(data[lr]) for lr in lrs]
        ax.plot(lrs, means, 'o-', label=f"{group_by}={group_val}")

        best_idx = int(np.nanargmin(means))
        best_lr, best_loss = lrs[best_idx], means[best_idx]
        ax.plot(best_lr, best_loss, '*', markersize=14, color='red', zorder=5)
        print(f"{group_by}={group_val}: best LR={best_lr:.2e}, avg train loss={best_loss:.4f}")

    if y_max is not None:
        ax.set_ylim(top=y_max)
    ax.set_xscale("log")
    ax.set_xlabel("Max Learning Rate")
    ax.set_ylabel(f"Avg Train Loss (first {n_steps} steps)")
    ax.set_title(f"Avg Train Loss — {entity}/{project}")
    ax.legend()

    return fig, ax


def get_metric_history(run, key: str):
    """Return arrays of (steps, values) for a logged metric key."""
    history = list(run.scan_history(keys=["_step", key]))
    steps = [row["_step"] for row in history if key in row]
    values = [row[key] for row in history if key in row]
    return np.array(steps), np.array(values)

