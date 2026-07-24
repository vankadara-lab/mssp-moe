#!/usr/bin/env bash
#SBATCH --job-name=jiang-6d
#SBATCH --partition=gatsby_h200
#SBATCH --nodelist=gpu-sr675-35
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=32
#SBATCH --time=48:00:00
#SBATCH --output=/nfs/ghome/live/mhaas/sebastian/logs/mssp-moe/6d-%j.out
#SBATCH --error=/nfs/ghome/live/mhaas/sebastian/logs/mssp-moe/6d-%j.err

SDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SDIR}/common.sh"
activate_venv

cd "${MSSP_MLP}"

echo "[stage-a] 6D sweep of ${JIANG_CONFIG} centered at ${BASELINE_CONFIG}/${JIANG_REGIME_KEY}"
echo "[stage-a] results -> ${SWEEP_6D_DIR}"
mkdir -p "${SWEEP_6D_DIR}"

python run_5d_sweep.py \
  --config "${JIANG_CONFIG}" \
  --optimizer adam \
  --base_lr 0.16 \
  --N 128 --P 5000 --T 500 \
  --last_layer_init mup \
  --dataset tinyimagenet \
  --num_gpus 8 \
  --early_stop_threshold 2 \
  --early_stop_check_step 100 \
  --results_dir "${SWEEP_6D_DIR}" \
  --init_std_values 0.015625 0.0625 0.25 1 4 \
  --lr_in_values 0.0625 0.25 1 4 16 \
  --lr_out_values 0.25 1 4 16 64 \
  --lr_router_values 0.0625 0.25 1 4 16 \
  --lr_expert1_values 0.25 1 4 16 64 \
  --lr_expert2_values 0.00390625 0.015625 0.0625 0.25 1

echo "[stage-a] done"
