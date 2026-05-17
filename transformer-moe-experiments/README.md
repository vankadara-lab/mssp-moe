# Transformer MoE Experiments

Transformer MoE experiments. Code is based on [nanoMoE](https://github.com/wolfecameron/nanoMoE) and
[nanoGPT](https://github.com/karpathy/nanoGPT) (see this
[blog post](https://cameronrwolfe.substack.com/nano-moe) for nanoMoE).

## Quick start

Setup: see the [top-level README](../README.md).

```bash
python data/dolma3_mix-150B-1025/prepare.py
python train.py config/allscale/sparse/d_model=256.py
```

Multi-GPU: `torchrun --standalone --nproc_per_node=N train.py <config>`.
Overrides: `python train.py <config> --learning_rate=3e-4 --n_exp=8`.

## Architectures

All configs share `n_head = n_embd / 64`, sequence length 1024, AdamW,
524 288 tokens per optimizer step.

| family | scaling | block pattern | configs |
|---|---|---|---|
| allscale-soft | `mup_allscale` | every layer MoE, top_k = n_exp | [`config/allscale/soft/`](config/allscale/soft) |
| allscale-sparse | `mup_allscale` | every layer MoE, top-k sparse | [`config/allscale/sparse/`](config/allscale/sparse) |
| bottleneck-soft | `mup_bottleneck` | every layer MoE, many small experts, soft | [`config/bottleneck/soft/`](config/bottleneck/soft) |
| bottleneck-sparse | `mup_bottleneck` | every layer MoE, many small experts, sparse | [`config/bottleneck/sparse/`](config/bottleneck/sparse) |
| efficient-bottleneck sparse-with-dense | `mup_bottleneck` | alternating dense MLP + bottleneck MoE | [`config/efficient-bottleneck/sparse-with-dense/`](config/efficient-bottleneck/sparse-with-dense) |

## Files

| | |
|---|---|
| `train.py` | training loop: gradient accumulation, mixed precision, DDP, checkpointing |
| `model.py` | `GPT` + `Block` + `CausalSelfAttention` + `Router` |
| `nano_moe.py` | sparse MoE layer — `MLPExperts` with 3D weight tensors + `torch.bmm` |
| `soft_moe.py` | soft MoE layer — every token → every expert, bypasses `Router` |
| `moe_to_monitor.py` | per-expert hooks for coordinate-check runs |
| `scaling.py` | scaling classes (`MupAllScale`, `MupBottleneck`, …) and `classify_parameter` |
| `manager.py` | global stats singleton used by MoE layers |
| `configurator.py` | exec's a config file, then applies `--key=value` overrides |


### LLM experiment plots (WandB)

Scripts in `plotting/` reproduce the LLM-scale figures from the paper. They fetch data
directly from WandB and require a `wandb login`. Set your WandB entity via `WANDB_ENTITY` (or
pass `--entity` to `plot_lr_transfer_joint_4panel.py`).

```bash
export WANDB_ENTITY=your_entity

python plotting/plot_lr_transfer_joint_4panel.py
python plotting/wandb_rcc_plots.py --project <wandb_project> --config_type <bottleneck|allscale|fixedE>
```

Fetched data is cached under `results/wandb_cache/`.
