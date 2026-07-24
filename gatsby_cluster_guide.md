# Gatsby (SWC) cluster — quick guide for another Claude session

Hand-off notes for driving the SWC (Sainsbury Wellcome Centre / Gatsby) cluster from a
laptop. Written for a Claude session in a *different* repo; adapt paths as needed.

## Access

- SSH alias (already configured in `~/.ssh/config`): **`swc-gateway`** (HostName
  `hpc-gw2`). Only reachable **off VPN**.
- User: `mhaas`. Home: `/nfs/ghome/live/mhaas/`. Long-lived project dir:
  `/nfs/ghome/live/mhaas/sebastian/` (this is what the container binds).
- Bastion for outside SWC network: `swc-bastion` (`ssh.swc.ucl.ac.uk`).

Sanity check: `ssh swc-gateway 'hostname && sinfo -h -o "%P %a %l %D %t" | head'`.

## Layout

```
/nfs/ghome/live/mhaas/sebastian/
  <repo>/                       # git checkout (e.g. moe-scaling)
  logs/<repo>/                  # %j.out / %j.err from sbatch
  python-packages/              # PYTHONUSERBASE for pip-installed extras
  torch2_9_cuda12_8_image.sif   # singularity container (11 GB)
/ceph/scratch/mhaas/            # fast scratch, bound into container; put data here
```

The container can't be built on SWC (`.def` → `.sif` needs root); build on galvani /
ferranti and `scp` the SIF over.

## Nodes / partitions

- **`gatsby_h200`** — primary. Single node `gpu-sr675-35`, 8× H200 NVL. Pin with
  `-w gpu-sr675-35`. High priority for this account.
- Other partitions (`gpu_leena`, L40S nodes, …) are LOW priority. L40S only fits
  `d_model ≤ 512`. See [reference_gatsby_cluster.md](../../../.claude/projects/-Users-m-Documents-code-moe-scaling/memory/reference_gatsby_cluster.md).

Common queries:

```bash
ssh swc-gateway 'squeue -u mhaas'
ssh swc-gateway 'squeue -p gatsby_h200 -o "%.10i %.9P %.20j %.8u %.2t %.10M %R"'
ssh swc-gateway 'sinfo -p gatsby_h200 -o "%N %C %G %t"'
ssh swc-gateway 'scontrol show job <jobid>'
ssh swc-gateway 'sacct -j <jobid> --format=JobID,State,ExitCode,Elapsed,MaxRSS'
```

## Code sync: git only

Do **not** rsync source into the cluster. Workflow:

1. Locally: commit + `git push`.
2. Remotely: `ssh swc-gateway 'cd /nfs/ghome/live/mhaas/sebastian/<repo> && git pull'`.
3. Only pull at **queue boundaries** — never while a sweep is in flight, or workers
   will drift across code revisions mid-run
   ([feedback_gatsby_sync_at_queue_boundary.md](../../../.claude/projects/-Users-m-Documents-code-moe-scaling/memory/feedback_gatsby_sync_at_queue_boundary.md)).
4. Prep + push the replacement code **before** cancelling the current job
   ([feedback_prep_before_cancel.md](../../../.claude/projects/-Users-m-Documents-code-moe-scaling/memory/feedback_prep_before_cancel.md)).

## File sync: rsync / scp for artefacts

For datasets, PNGs, checkpoints — anything not in git:

```bash
# push a dataset up
rsync -avP --partial ./data/foo.bin \
  swc-gateway:/ceph/scratch/mhaas/<repo>/data/

# pull plots down (flat temp dir on remote, then untar locally)
ssh swc-gateway 'mkdir -p /tmp/pngs && \
  find /nfs/ghome/live/mhaas/sebastian/<repo>/results -name "*.png" \
    -exec cp {} /tmp/pngs/ \; && \
  tar -cf /tmp/pngs.tar -C /tmp/pngs .'
scp swc-gateway:/tmp/pngs.tar /tmp/
tar -xf /tmp/pngs.tar -C ./results/downloaded/
```

Note: macOS BSD `tar --strip-components=99` **silently skips** — do the flatten on
the remote, not in the extract
([feedback_tar_download.md](../../../.claude/projects/-Users-m-Documents-code-moe-scaling/memory/feedback_tar_download.md)).

## Container / env pattern

Every sbatch that runs Python does the same dance:

```bash
singularity exec --nv \
  --bind /nfs/ghome/live/mhaas/sebastian:/nfs/ghome/live/mhaas/sebastian \
  --bind /ceph/scratch/mhaas:/ceph/scratch/mhaas \
  torch2_9_cuda12_8_image.sif bash -c "
    export PYTHONUSERBASE=/nfs/ghome/live/mhaas/sebastian/python-packages
    export WANDB_API_KEY=<your key>          # ~/.netrc is not in the bind mount
    cd /nfs/ghome/live/mhaas/sebastian/<repo>
    python train.py $*
  "
```

`WANDB_API_KEY` **must** be set as an env var inside the container — `~/.netrc`
lives at `/nfs/ghome/live/mhaas/` (not `.../sebastian/`) and isn't bound in.

## Submitting jobs

Single-GPU H200:

```bash
ssh swc-gateway 'cd /nfs/ghome/live/mhaas/sebastian/<repo> && \
  sbatch cluster/swc/train_1xH200.sh config/foo.py --learning_rate=3e-4'
```

Sweep across 8 GPUs (flock-based work queue over a jobs file):

```bash
# jobs_foo.txt: one line per job, each is a train.py argv
ssh swc-gateway 'cd /nfs/ghome/live/mhaas/sebastian/<repo> && \
  sbatch cluster/swc/sweep_8xH200.sh cluster/swc/jobs_foo.txt'

# pack multiple runs per GPU (small models only)
ssh swc-gateway 'cd /nfs/ghome/live/mhaas/sebastian/<repo> && \
  sbatch --export=ALL,WORKERS_PER_GPU=4 cluster/swc/sweep_8xH200.sh \
    cluster/swc/jobs_foo.txt'
```

Gotchas for packed sweeps:

- **wandb service race**: multiple concurrent `wandb.init()` on the same node fight
  over a shared port file. Give each worker its own `WANDB_DIR=/tmp/wandb-$JOBID-gpu$g-slot$s`
  ([feedback_wandb_service_race_packing.md](../../../.claude/projects/-Users-m-Documents-code-moe-scaling/memory/feedback_wandb_service_race_packing.md)).
- **Loop order**: iterate `for slot; for gpu`, not `for gpu; for slot`, otherwise
  early jobs pile onto GPU 0 and later GPUs run empty
  ([feedback_sweep_loop_order.md](../../../.claude/projects/-Users-m-Documents-code-moe-scaling/memory/feedback_sweep_loop_order.md)).
- **Never trust sbatch `COMPLETED`** for packed runs — check the wandb run state.
  A crashed worker still lets the sbatch exit cleanly.
- **Driver-side wandb leak**: don't `import wandb` / call `wandb.Api()` in the
  script that runs `sbatch`. It spawns a gateway-side `wandb-core` service, sets
  `WANDB_SERVICE`, and the sbatches inherit it → workers fail silently
  ([feedback_wandb_service_env_leak_from_driver.md](../../../.claude/projects/-Users-m-Documents-code-moe-scaling/memory/feedback_wandb_service_env_leak_from_driver.md)).
  Fix: strip `WANDB_SERVICE` from the submit env, and `unset WANDB_SERVICE` in the
  inner shell.

## Monitoring

Live tail of a job's log:

```bash
ssh swc-gateway 'tail -f /nfs/ghome/live/mhaas/sebastian/logs/<repo>/<jobid>.out'
```

Recent log content across the last N files (Python helper if the repo has one):

```python
from cluster.mlcloud import swc_exec, swc_print_logs
swc_exec('squeue -u mhaas')
swc_print_logs('/nfs/ghome/live/mhaas/sebastian/logs/<repo>', num_files=5)
```

For wandb: use the `wandb.Api()` from your **laptop**, not on the cluster, to avoid
the env leak above. Cache scan_history results (they're slow).

## Cancelling

```bash
ssh swc-gateway 'scancel <jobid>'                    # one job
ssh swc-gateway 'scancel -u mhaas -p gatsby_h200'    # everything on the partition
```

`scancel` kills the whole sbatch — including any in-flight workers holding items
they popped off the flock queue. Those items are **lost from the file** (already
`sed -i 1d`); re-emit them before relaunching.

## Debugging first-time setup

If jobs fail at `wandb.init()`: check `WANDB_API_KEY` is exported inside the
`singularity exec` bash string, not just in the sbatch env.

If `import torch_module_monitor` fails: verify `PYTHONUSERBASE` is exported inside
the container and the package is installed under
`/nfs/ghome/live/mhaas/sebastian/python-packages/lib/python3.*/site-packages/`.

If DDP hangs at startup: bump `NCCL_TIMEOUT` and
`TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` (see [`train_1xH200.sh`](../cluster/swc/train_1xH200.sh)
for the values in use).
