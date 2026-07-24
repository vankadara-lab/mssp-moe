#!/usr/bin/env bash
# Shared env for mssp-moe SWC sbatch scripts. Source this at the top of each stage script.

set -euo pipefail

export MSSP_REPO=/nfs/ghome/live/mhaas/mssp-moe
export MSSP_MLP=${MSSP_REPO}/mlp-moe-experiments
export MSSP_VENV=${MSSP_REPO}/venv
export MSSP_LOGS=/nfs/ghome/live/mhaas/sebastian/logs/mssp-moe
export MSSP_SCRATCH=/ceph/scratch/mhaas/mssp-moe
export MSSP_RESULTS=${MSSP_SCRATCH}/results

# Comparison config (Jiang) and matching regime key
export JIANG_CONFIG=mup_adam_jiang_allscaling_multfree
export JIANG_REGIME_KEY=soft_no_shared_ll1overN
export BASELINE_CONFIG=mup_adam_allscaling_stdinit_ours

# Results dirs (stable, referenced by multiple stages)
export SWEEP_6D_DIR=${MSSP_RESULTS}/5d_jiang_${JIANG_REGIME_KEY}
export LR_SWEEP_DIR=${MSSP_RESULTS}/lr_sweep_jiang_${JIANG_REGIME_KEY}
export COORD_DIR=${MSSP_RESULTS}/coord_jiang_${JIANG_REGIME_KEY}

# WANDB (needed by lr sweep + coord check; the 6D sweep uses --minimal_stats and no wandb)
export WANDB_API_KEY=e68876358306a89d8031e5402c47df8274c696d0
unset WANDB_SERVICE || true

activate_venv() {
  # shellcheck disable=SC1091
  source "${MSSP_VENV}/bin/activate"
  export PYTHONUNBUFFERED=1
}
