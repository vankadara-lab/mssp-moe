# How to Scale Mixture-of-Experts: From muP to the Maximally Scale-Stable Parameterization

Code accompanying the paper:
**["How to Scale Mixture-of-Experts: From μP to the Maximally Scale-Stable Parameterization"](https://arxiv.org/abs/2605.14200)**

## Contents

- [`mlp-moe-experiments/`](mlp-moe-experiments) — MLP MoEs on CIFAR-10 / TinyImageNet:
  coordinate checks, parameterization studies, LR / 5D hyperparameter sweeps. See its
  [README](mlp-moe-experiments/README.md).
- [`transformer-moe-experiments/`](transformer-moe-experiments) — Transformer MoEs trained on
  OLMo-3 pretraining data (all-scale and bottleneck families, soft and sparse routing). See its
  [README](transformer-moe-experiments/README.md).

## Setup

```bash
pip install -r requirements.txt
```

The two subdirectories are independent: each has its own training entry point, configs, and
plotting scripts. Pick the one matching the experiments you want to reproduce and follow its
README from there.

## License

MIT — see [LICENSE](LICENSE).

## Citation
If you use this software, or any ideas from our code or paper, please cite the following publication:

```bib
@article{vankadara2026moescaling,
  title={How to Scale Mixture-of-Experts: From μP to the Maximally Scale-Stable Parameterization},
  author={Chennuru Vankadara, Leena and Haas, Moritz and Hayward, Luke and Bordt, Sebastian and Breccia, Alessandro},
  journal={arXiv:2605.14200},
  year={2026}
}
```
