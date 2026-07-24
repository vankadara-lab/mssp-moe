#!/usr/bin/env bash
# Submit the full Jiang-vs-MSSP pipeline as a Slurm dependency chain.
# Prints jobids and returns.

set -euo pipefail
SDIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "[submit] stage A (6D sweep)"
JID_A=$(sbatch --parsable "${SDIR}/stage_a_6d_sweep.sh")
echo "[submit] stage A jobid: ${JID_A}"

echo "[submit] stage B (pick + LR sweep + plot), depends on A"
JID_B=$(sbatch --parsable --dependency=afterok:${JID_A} "${SDIR}/stage_b_lr_sweep.sh")
echo "[submit] stage B jobid: ${JID_B}"

echo "[submit] stage C (coord check + auto-plot), depends on B"
JID_C=$(sbatch --parsable --dependency=afterok:${JID_B} "${SDIR}/stage_c_coord_check.sh")
echo "[submit] stage C jobid: ${JID_C}"

echo
echo "chain: A(${JID_A}) -> B(${JID_B}) -> C(${JID_C})"
echo "monitor: squeue -u \$USER"
echo "logs:    tail -f /nfs/ghome/live/mhaas/sebastian/logs/mssp-moe/{6d,lr,coord}-*.out"
