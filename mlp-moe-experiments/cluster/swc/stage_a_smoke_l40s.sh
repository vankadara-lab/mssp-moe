#!/usr/bin/env bash
#SBATCH --job-name=jiang-6d-smoke
#SBATCH --partition=gpu_lowp
#SBATCH --nodes=1
#SBATCH --gres=gpu:l40s:8
#SBATCH --cpus-per-task=16
#SBATCH --time=00:30:00
#SBATCH --output=/nfs/ghome/live/mhaas/sebastian/logs/mssp-moe/6d-smoke-%j.out
#SBATCH --error=/nfs/ghome/live/mhaas/sebastian/logs/mssp-moe/6d-smoke-%j.err

# Small end-to-end test of Stage A on L40S: 2^3 = 8 configs at N=128, T=200.
# Only exercises the sweep runner + moe_training + result serialization.

SDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
source "${SDIR}/common.sh"
activate_venv

cd "${MSSP_MLP}"

SMOKE_DIR=${MSSP_RESULTS}/smoke_5d_jiang_l40s
echo "[smoke] hostname=$(hostname) nvidia-smi head:"
nvidia-smi -L
echo "[smoke] results -> ${SMOKE_DIR}"
mkdir -p "${SMOKE_DIR}"

python run_5d_sweep.py \
  --config "${JIANG_CONFIG}" \
  --optimizer adam \
  --base_lr 0.16 \
  --N 128 --P 5000 --T 200 \
  --last_layer_init mup \
  --dataset tinyimagenet \
  --num_gpus 8 \
  --early_stop_threshold 2 \
  --early_stop_check_step 50 \
  --results_dir "${SMOKE_DIR}" \
  --init_std_values 0.25 \
  --lr_in_values 0.25 1 \
  --lr_out_values 4 \
  --lr_router_values 1 \
  --lr_expert1_values 1 4 \
  --lr_expert2_values 0.015625 0.0625

echo
echo "[smoke] stats produced:"
ls "${SMOKE_DIR}/${JIANG_CONFIG}/stats/" 2>/dev/null | head -20 || echo "  NONE"

echo
echo "[smoke] running picker (dry-run: writes to /tmp, not real tuned_multipliers.py)"
cp "${MSSP_MLP}/tuned_multipliers.py" /tmp/tuned_multipliers_smoke.py
python pick_optimum.py \
  --sweep_results "${SMOKE_DIR}" \
  --config "${JIANG_CONFIG}" \
  --regime_key soft_no_shared_ll1overN \
  --tuned_multipliers_path /tmp/tuned_multipliers_smoke.py \
  --dump_json "${SMOKE_DIR}/optimum.json"

echo
echo "[smoke] tail of /tmp/tuned_multipliers_smoke.py:"
tail -20 /tmp/tuned_multipliers_smoke.py
echo
echo "[smoke] done"
