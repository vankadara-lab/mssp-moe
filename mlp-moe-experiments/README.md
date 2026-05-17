# MLP MoE Experiments

MLP MoEs on CIFAR-10 / TinyImageNet for coordinate checks and parameterization studies.

Run all commands below from this directory (`mlp-moe-experiments/`).

## Directory layout

```
mlp-moe-experiments/
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

### Single training run
```bash
python moe_training.py \
  --N 256 --P 5000 --T 500 \
  --scaling_config mup_allscaling_multfree \
  --nonlin gelu --router_fn sigmoid \
  --results_dir results/single_run
```

### Coordinate check
Verifies that key quantities stay width-independent — the defining property of a well-parameterized model.
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

### Learning rate sweep
```bash
python run_lr_sweep.py \
  --config fixed_E_mup_multfree \
  --widths 128 256 512 1024 2048 \
  --seeds 42 43 44 45 \
  --last_layer_init zero \
  --results_dir results/lr_sweep
```

### 5D hyperparameter sweep
```bash
python run_5d_sweep.py \
  --config mup_allscaling_multfree \
  --N 128 --P 5000 --T 500 \
  --init_std_values 0.0625 0.25 1.0 4.0 16.0 \
  --lr_in_values   0.0625 0.25 1.0 4.0 16.0 \
  --lr_out_values  0.0625 0.25 1.0 4.0 16.0 \
  --num_gpus 8 \
  --results_dir results/5d_sweep
```

## Coordinate Check Metrics

The coordinate check tracks whether key quantities remain **model-scale-independent**. A
configuration passes if all metrics are width-independent:

| Metric | Definition | Interpretation |
|--------|-----------|----------------|
| Effective updates | `‖(W(t)−W(0)) x‖_RMS` | How much the current layer's weight updates affect the layer's output |
| Propagating updates | `‖W(0) Δx‖_RMS` | How feature changes propagate forward through each layer's initial weights |
| Gradient norms | `‖∂L/∂W‖_RMS` | Per-layer gradient scale |
| Activation RMS | `‖h^l‖_RMS` | Per-layer activation scale |
| Router gradient | `‖∂L/∂φ‖_RMS` | Loss gradient w.r.t. router pre-activations |

For μP to hold, effective and propagating updates must both be width independent. Gradient norms
and activation RMS being width independent is a necessary but not sufficient condition.

## Scaling regimes

| Regime | Config prefix | N_expert | M (experts) |
|--------|--------------|----------|-------------|
| Fixed-E | `fixed_E_*` | N (scales) | 8 (fixed) |
| Bottleneck | `*_bottleneck_*` | 16 (fixed) | N/16 (scales) |
| All-Scaling | `*_allscaling_*` | N (scales) | N/16 (scales) |

## Key parameters

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

## Output format

Each run saves results under `results/{config}/stats/`:

- **`nn_N{N}_M{M}_…{config}.npz`** — training statistics with keys:
  - `train_loss`, `val_loss`, `val_acc`, `val_top5`
  - `effective_updates_{layer}`, `propagating_updates_{layer}`
  - `gradient_norms_{layer}`, `activation_rms_{layer}`
  - `router_concentration`, `router_entropy`
- **`config_N{N}_…{config}.json`** — full configuration for that run

```python
import numpy as np
data = np.load('results/.../stats/nn_N256_....npz', allow_pickle=True)
print(data['val_loss'])
```

## Plotting

All scripts in `plotting/` are run from this directory and save figures to the results directory.

```bash
python plotting/plot_coordinate_check.py  --results_dir results/coord_check
python plotting/plot_training_dynamics.py --results_dir results/coord_check
python plotting/plot_rcc_exponents.py     --results_dir results/coord_check
python plotting/plot_lr_sweep.py          --results_dir results/lr_sweep
python plotting/plot_5d_sweep.py          --results_dir results/5d_sweep
python plotting/extract_sweep_summary.py  --results_dir results/5d_sweep
```

## Datasets

- **CIFAR-10** (default): downloaded automatically on first run
- **TinyImageNet**: `--dataset tinyimagenet` — downloaded automatically to `data/`

## Multi-GPU execution

```bash
--num_gpus 8    # Use 8 GPUs
--num_gpus -1   # Auto-detect all available GPUs
--num_gpus 0    # CPU only
```
