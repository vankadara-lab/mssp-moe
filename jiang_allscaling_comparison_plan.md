# Run MuPJiangAdam allscaling comparison vs MSSP baseline (MLP MoE)

## Context

Goal: evaluate the new scaling config [`mup_adam_jiang_allscaling_multfree`](mlp-moe-experiments/scaling_configs.py) (Jiang-style muP-Adam, router init = `1/N`, **global** Adam ε) against the current MSSP-Adam baseline [`mup_adam_allscaling_stdinit_ours`](mlp-moe-experiments/scaling_configs.py) (standard router init, per-layer Adam ε scaling) under the "μP all-scaling, soft routing, no shared experts, 1/N last-layer init" regime.

Deliverable: for the new config, reproduce the full three-stage protocol already run for the MSSP baseline — a 6D multiplier sweep to find its optimum, a width-transfer LR sweep, and a coordinate check at optimal LR — then produce the same plot set so we can compare **stability**, **transfer**, and **performance** side-by-side.

## Execution model

- **Location on cluster**: `/nfs/ghome/live/mhaas/mssp-moe` (under `mhaas/`, **not** under `sebastian/`).
- **Env**: fresh Python venv at `/nfs/ghome/live/mhaas/mssp-moe/venv` created with `/usr/bin/python3 -m venv`. **No singularity** — the moe-scaling container belongs to a different repo and is not used here.
- **Torch**: `torch` + `torchvision` from `https://download.pytorch.org/whl/cu124` (H200-compatible, matches CUDA runtime bundled with those wheels).
- **Skipped from `requirements.txt`**: `torch_module_monitor` — only imported from `transformer-moe-experiments/`, not needed for the MLP pipeline.
- **User's existing running jobs must not be interrupted** — new work runs as fresh sbatches, chained via `--dependency=afterok`, and will PEND behind the running moe-scaling jobs until the node frees up.

## Pipeline architecture (auto-queue)

Three sbatches chained via Slurm dependencies. A single orchestrator submits them all, then returns immediately with the JOBIDs:

```
submit_pipeline.sh
    sbatch stage_a_6d_sweep.sh                                     -> JID_A
    sbatch --dependency=afterok:JID_A stage_b_lr_sweep.sh          -> JID_B
    sbatch --dependency=afterok:JID_B stage_c_coord_check.sh       -> JID_C
```

If any stage fails, the downstream stages remain PENDING with reason `DependencyNeverSatisfied`. `scancel <jid>` them or restart the failed stage manually with a new `--dependency=afterok` chain.

### Stage A — 6D multiplier sweep
[`cluster/swc/stage_a_6d_sweep.sh`](mlp-moe-experiments/cluster/swc/stage_a_6d_sweep.sh) runs [`run_5d_sweep.py`](mlp-moe-experiments/run_5d_sweep.py) on `gatsby_h200` (all 8 GPUs, 1 worker/GPU).

**Center**: [`mup_adam_allscaling_stdinit_ours` → `soft_no_shared_ll1overN`](mlp-moe-experiments/tuned_multipliers.py):
```
base_lr=0.16, init_std_mult=0.25,
lr_mult_in=1.0, lr_mult_out=4.0, lr_mult_router=1.0,
lr_mult_expert1=4.0, lr_mult_expert2=0.0625
```

**Grid**: 5 pts/dim, log-4 spacing → 5⁶ = 15,625 configs. `center_first_order()` visits center values first; `--early_stop_threshold 2 --early_stop_check_step 100` aborts any config whose val_acc at step 100 falls below half the best-seen for its (config, N, dataset, t) bucket.

| Dim | Values |
|---|---|
| `--init_std_values` | 0.015625 0.0625 **0.25** 1 4 |
| `--lr_in_values` | 0.0625 0.25 **1** 4 16 |
| `--lr_out_values` | 0.25 1 **4** 16 64 |
| `--lr_router_values` | 0.0625 0.25 **1** 4 16 |
| `--lr_expert1_values` | 0.25 1 **4** 16 64 |
| `--lr_expert2_values` | 0.00390625 0.015625 **0.0625** 0.25 1 |

Results: `/ceph/scratch/mhaas/mssp-moe/results/5d_jiang_soft_noshared_ll1overN/mup_adam_jiang_allscaling_multfree/stats/nn_*.npz`. The script also auto-plots the 6D heatmaps via `plotting/plot_5d_sweep.py`.

### Stage B — pick optimum + LR transfer sweep + plot
[`cluster/swc/stage_b_lr_sweep.sh`](mlp-moe-experiments/cluster/swc/stage_b_lr_sweep.sh) does three things in one sbatch:

1. **Pick**: [`pick_optimum.py`](mlp-moe-experiments/pick_optimum.py) scans Stage A's `stats/*.npz`, picks the argmax over `val_top5_acc` (or `val_acc`), and **appends a sentinel-bracketed override block** to [`tuned_multipliers.py`](mlp-moe-experiments/tuned_multipliers.py) that mutates `TUNED_MULTIPLIERS` at import time:
   ```python
   # --- BEGIN auto-pick mup_adam_jiang_allscaling_multfree/soft_no_shared_ll1overN ---
   TUNED_MULTIPLIERS.setdefault('mup_adam_jiang_allscaling_multfree', {})['soft_no_shared_ll1overN'] = { ... }
   # --- END auto-pick ... ---
   ```
   Re-running the picker replaces the same block idempotently. Also dumps `optimum.json` next to the results for inspection.

2. **LR sweep**: [`run_lr_sweep.py`](mlp-moe-experiments/run_lr_sweep.py) with `--use_optimal` picks up the just-written entry via [`get_optimal_values`](mlp-moe-experiments/tuned_multipliers.py) (matches `soft_no_shared_ll1overN` because routing=soft, no `--share_expert_weights_*`, `--last_layer_init mup`).
   Widths: 128, 256, 512, 1024, 2048. Seeds: 42–45. Dataset: tinyimagenet.

3. **Plot**: `plotting/plot_lr_sweep.py --results_dir ... --config ...`.

### Stage C — Full coordinate check + auto-plot
[`cluster/swc/stage_c_coord_check.sh`](mlp-moe-experiments/cluster/swc/stage_c_coord_check.sh) runs [`run_full_coordinate_check.py`](mlp-moe-experiments/run_full_coordinate_check.py) with the same `--use_optimal` lookup. Same widths. The script auto-invokes `plotting/plot_coordinate_check.py` at the end (all coordinate-check figures are produced automatically).

## Post-pipeline: overlay with the MSSP baseline

To put Jiang and MSSP on the same axes, run the plotting scripts against a directory that contains **both** config subdirs — e.g., symlink the existing baseline stats into the new results dirs:

```bash
ln -s /ceph/scratch/mhaas/mssp-moe/results/<baseline-lr-sweep-dir>/mup_adam_allscaling_stdinit_ours \
      /ceph/scratch/mhaas/mssp-moe/results/lr_sweep_jiang_soft_noshared_ll1overN/mup_adam_allscaling_stdinit_ours
```

Then re-run `plot_lr_sweep.py`, `plot_coordinate_check.py`, `plot_rcc_exponents.py`, `plot_training_dynamics.py` against the combined dir. Discovery is dynamic — configs get picked up automatically as long as they're subdirs of `results_dir`.

Optional one-liner in [`plotting/label_utils.py`](mlp-moe-experiments/plotting/label_utils.py) LABEL_MAP so the legend reads nicely:
```python
'mup_adam_jiang_allscaling_multfree': 'μP, Jiang (router 1/N, global ε)',
```

## Files added / modified

**Added in this task:**
- `mlp-moe-experiments/cluster/swc/common.sh` — shared env, WANDB key, results-dir constants
- `mlp-moe-experiments/cluster/swc/stage_a_6d_sweep.sh` — Stage A sbatch
- `mlp-moe-experiments/cluster/swc/stage_b_lr_sweep.sh` — Stage B sbatch (pick + LR sweep + plot)
- `mlp-moe-experiments/cluster/swc/stage_c_coord_check.sh` — Stage C sbatch (coord check + auto-plot)
- `mlp-moe-experiments/cluster/swc/submit_pipeline.sh` — orchestrator (dependency chain)
- `mlp-moe-experiments/pick_optimum.py` — parses Stage A results, upserts a sentinel-bracketed override into `tuned_multipliers.py`

**Modified:**
- `mlp-moe-experiments/scaling_configs.py` — new `MuPJiangAdamAllScalingConfig` class + CONFIGS entry (typo fixed: constructor arg now matches dict key)
- `mlp-moe-experiments/tuned_multipliers.py` — Stage B's picker appends a new override block (runtime, not part of the initial commit)
- `mlp-moe-experiments/plotting/label_utils.py` — optional label entry

## Verification

**After Stage 0 (prerequisites)**:
```bash
ssh swc-gateway 'ls /nfs/ghome/live/mhaas/mssp-moe/venv/bin/python && \
  /nfs/ghome/live/mhaas/mssp-moe/venv/bin/python -c \
  "import torch, wandb, numpy, matplotlib; print(torch.__version__, torch.version.cuda)"'
```
Expect: `torch 2.x.x  cuda 12.4`.

**After Stage A**:
```bash
ssh swc-gateway 'ls /ceph/scratch/mhaas/mssp-moe/results/5d_jiang_soft_noshared_ll1overN/mup_adam_jiang_allscaling_multfree/stats | wc -l'
```
Expect ≤ 15625 (fewer if early-stopping culled outer points). `cat` the auto-appended block in `tuned_multipliers.py` and/or `optimum.json` — best_acc should be ≥ ~0.29 (baseline is 0.3265).

**After Stage B**: LR-transfer curves in `results/lr_sweep_.../plots/` should peak at a similar η across widths (the muP transfer property). Compare visually with the MSSP baseline.

**After Stage C**: coordinate-check plots should show width-independent effective/propagating updates. Divergence at large N signals a broken parameterization.

## Notes / risks

- **Wall-clock**: Stage A is the dominant cost. 15,625 configs × ~1–2 min × 8 concurrent workers ≈ 30+ h *if all run to completion*; realistic ~10–15 h with aggressive early-stopping.
- **Queue**: two `moe-sweep` jobs are running on `gpu-sr675-35`. Stage A will PEND (`Resources`) until the node frees. **Do not interrupt those jobs.**
- **Idempotency**: re-running `pick_optimum.py` overwrites the same sentinel block, so Stage B is safe to retry. Re-running Stage A/B/C overwrites results dirs; individual `moe_training.py` runs skip if the exact output file exists (see `moe_training.py`).
- **CUDA runtime**: cu124 torch wheels bundle the CUDA runtime; they only need a compatible NVIDIA driver on the H200 node. If a `CUDA driver version is insufficient` error appears, bump to cu126/cu128 by re-installing.
