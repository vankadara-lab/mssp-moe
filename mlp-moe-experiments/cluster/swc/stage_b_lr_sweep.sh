#!/usr/bin/env bash
#SBATCH --job-name=jiang-lr
#SBATCH --partition=gatsby_h200
#SBATCH --nodelist=gpu-sr675-35
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/ghome/live/mhaas/sebastian/logs/mssp-moe/lr-%j.out
#SBATCH --error=/nfs/ghome/live/mhaas/sebastian/logs/mssp-moe/lr-%j.err

SDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SDIR}/common.sh"
activate_venv

cd "${MSSP_MLP}"

echo "[stage-b] picking optimum from ${SWEEP_6D_DIR}"
python pick_optimum.py \
  --sweep_results "${SWEEP_6D_DIR}" \
  --config "${JIANG_CONFIG}" \
  --regime_key "${JIANG_REGIME_KEY}" \
  --tuned_multipliers_path "${MSSP_MLP}/tuned_multipliers.py" \
  --dump_json "${SWEEP_6D_DIR}/optimum.json"

echo "[stage-b] LR transfer sweep -> ${LR_SWEEP_DIR}"
mkdir -p "${LR_SWEEP_DIR}"
python run_lr_sweep.py \
  --config "${JIANG_CONFIG}" \
  --optimizer adam \
  --widths 128 256 512 1024 2048 \
  --routing_types soft \
  --last_layer_init mup \
  --use_optimal \
  --dataset tinyimagenet \
  --P 50000 --T 1000 \
  --seeds 42 43 44 45 \
  --num_gpus 8 \
  --results_dir "${LR_SWEEP_DIR}"

echo "[stage-b] plotting LR sweep"
python plotting/plot_lr_sweep.py \
  --results_dir "${LR_SWEEP_DIR}" \
  --config "${JIANG_CONFIG}" || echo "[stage-b] plot_lr_sweep exited non-zero (continuing)"

echo "[stage-b] done"
