# How to Scale Mixture-of-Experts: From μP to the Maximally Scale-Stable Parameterization

Code accompanying the paper:  
**"How to Scale Mixture-of-Experts: From μP to the Maximally Scale-Stable Parameterization"**

## Setup

```bash
pip install -r requirements.txt
```

## Repository Structure

```
moe_scaling/
├── moe_training.py              # MoE model and training loop
├── scaling_configs.py           # All scaling configurations (μP, NTP, SP, variants)
├── moe_logging.py               # Coordinate-check statistics (effective/propagating updates, etc.)
├── tuned_multipliers.py         # Tuned per-layer LR multipliers from hyperparameter sweeps
├── run_full_coordinate_check.py # Run coordinate checks across widths and seeds
├── run_lr_sweep.py              # Learning rate sweep for a single config
├── run_5d_sweep.py              # 5D hyperparameter grid search
├── plotting/                    # Paper figure scripts (see below)
└── utils/
    └── tiny_imagenet.py         # TinyImageNet dataset loader
```

## Quick Start

### Single Training Run
```bash
python moe_training.py \
  --N 256 --P 5000 --T 500 \
  --scaling_config mup_allscaling_multfree \
  --nonlin gelu --router_fn sigmoid \
  --results_dir results/single_run
```

### Coordinate Check
Verifies that key quantities stay O(1) in width — the defining property of a well-parameterized model.
```bash
python run_full_coordinate_check.py \
  --configs mup_bottleneck_ours \
  --widths 128 256 512 1024 2048 \
  --P 50000 --T 1000 \
  --seeds 42 43 44 45 \
  --num_gpus 8 \
  --use_optimal \
  --results_dir results/coord_check
```

### Learning Rate Sweep
```bash
python run_lr_sweep.py \
  --config fixed_E_mup_multfree \
  --widths 128 256 512 1024 2048 \
  --seeds 42 43 44 45 \
  --last_layer_init zero \
  --results_dir results/lr_sweep
```

### Hyperparameter Sweep (5D)
```bash
python run_5d_sweep.py \
  --config mup_allscaling_multfree \
  --N 128 --P 5000 --T 500 \
  --init_std_values 0.0625 0.25 1.0 4.0 16.0 \
  --lr_in_values 0.0625 0.25 1.0 4.0 16.0 \
  --lr_out_values 0.0625 0.25 1.0 4.0 16.0 \
  --num_gpus 8 \
  --results_dir results/5d_sweep
```

## Coordinate Check Metrics

The coordinate check tracks whether key quantities remain **O(1) across model widths**. A configuration passes if all metrics are width-independent:

| Metric | Definition | Interpretation |
|--------|-----------|----------------|
| Effective updates | `‖(W(t)−W(0)) x‖_RMS` | How much training has moved each layer's output |
| Propagating updates | `‖W(0) Δx‖_RMS` | How feature changes propagate forward through initial weights |
| Gradient norms | `‖∂L/∂W‖_RMS` | Per-layer gradient scale |
| Activation RMS | `‖h^l‖_RMS` | Per-layer activation scale |
| Router gradient | `‖∂L/∂φ‖_RMS` | Loss gradient w.r.t. router pre-activations |

For μP to hold, effective and propagating updates must both be O(1). Gradient norms and activation RMS being O(1) is a necessary but not sufficient condition.

## Scaling Regimes

Three regimes studied in the paper:

| Regime | Config prefix | N_expert | M (experts) |
|--------|--------------|----------|-------------|
| Fixed-E | `fixed_E_*` | N (scales) | 8 (fixed) |
| All-Scaling | `*_allscaling_*` | N (scales) | N/16 (scales) |
| Bottleneck | `*_bottleneck_*` | 16 (fixed) | N/16 (scales) |

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--N` | Hidden width |
| `--P` | Training set size |
| `--T` | Training iterations |
| `--scaling_config` | Parameterization (see `scaling_configs.py`) |
| `--router_fn` | Router activation (`sigmoid`, `softmax`, `linear`) |
| `--topk` | Routing: 0=soft, k=top-k sparse |
| `--router_init` | Router init: `zero`, `mup`, `ntp` |
| `--num_gpus` | Parallel GPU workers (-1=auto, 0=CPU) |
| `--use_optimal` | Load tuned LR multipliers from `tuned_multipliers.py` |
| `--last_layer_init` | Output layer init: `zero` (recommended), `mup` |

## Output Format

Each run saves results under `results/{config}/stats/`:

- **`nn_N{N}_M{M}_...{config}.npz`** — training statistics array with keys:
  - `train_loss`, `val_loss`, `val_acc`, `val_top5` — loss and accuracy over time
  - `effective_updates_{layer}`, `propagating_updates_{layer}` — coordinate check metrics
  - `gradient_norms_{layer}`, `activation_rms_{layer}` — gradient and activation scales
  - `router_concentration`, `router_entropy` — routing statistics
- **`config_N{N}_...{config}.json`** — full configuration for that run (useful for inspecting exact hyperparameters)

Load results with:
```python
import numpy as np
data = np.load('results/.../stats/nn_N256_....npz', allow_pickle=True)
print(data['val_loss'])
```

## Plotting

All scripts in `plotting/` are run from the repo root and save figures to the results directory.

```bash
# Coordinate check (effective/propagating updates, scaling exponents)
python plotting/plot_coordinate_check.py --results_dir results/coord_check

# Training dynamics (loss, accuracy, router concentration, entropy)
python plotting/plot_training_dynamics.py --results_dir results/coord_check

# Scaling exponent bar charts (RCC plots)
python plotting/plot_rcc_exponents.py --results_dir results/coord_check

# LR sweep curves
python plotting/plot_lr_sweep.py --results_dir results/lr_sweep

# 5D hyperparameter sweep visualization
python plotting/plot_5d_sweep.py --results_dir results/5d_sweep

# Extract sweep summary statistics to compressed JSON
python plotting/extract_sweep_summary.py --results_dir results/5d_sweep
```

### LLM Experiment Plots (WandB)

Scripts in `plotting/llm/` reproduce the LLM-scale figures from the paper. They fetch data directly from WandB and require a `wandb login`. Set your WandB entity via the `WANDB_ENTITY` environment variable (or pass `--entity` to `plot_lr_transfer_joint_4panel.py`).

```bash
export WANDB_ENTITY=your_entity

# LR transfer figure (joint 4-panel: μP vs MSSP, Regime II and III)
python plotting/llm/plot_lr_transfer_joint_4panel.py

# Coordinate check RCC plots for LLM experiments
python plotting/llm/wandb_rcc_plots.py --project <wandb_project> --config_type <bottleneck|allscale|fixedE>
```

Fetched data is cached under `results/wandb_cache/` to avoid repeated API calls.

## Datasets

- **CIFAR-10** (default): downloaded automatically on first run
- **TinyImageNet**: `--dataset tinyimagenet` — downloaded automatically to `data/`

## Multi-GPU Execution

Experiment runners distribute jobs across GPUs:
```bash
--num_gpus 8    # Use 8 GPUs
--num_gpus -1   # Auto-detect all available GPUs
--num_gpus 0    # CPU only
```
