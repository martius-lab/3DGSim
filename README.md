# 3DGSim: Learning 3D-Gaussian Simulators from RGB Videos

[![arXiv](https://img.shields.io/badge/arXiv-2503.24009-b31b1b.svg)](https://arxiv.org/abs/2503.24009)
[![Datasets](https://img.shields.io/badge/Datasets-Keeper-2C6E91.svg)](https://keeper.mpdl.mpg.de/d/a2c4e6f34ad24419947a/)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-mzhobro%2F3dgsim-FFD21E.svg)](https://huggingface.co/mzhobro/3dgsim)
[![Code](https://img.shields.io/badge/Code-synthetic__3dgsim-2ea44f.svg)](https://github.com/martius-lab/3DGSim/tree/synthetic_3dgsim)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

**Mikel Zhobro, Andreas René Geist, Georg Martius** · ICML 2026

**3DGSim** is a learned 3D-Gaussian simulator trained directly from multi-view RGB videos — no
ground-truth 3D supervision. It reconstructs a scene into 3D Gaussians and rolls out its dynamics
autoregressively, enabling photorealistic simulation and latent-space scene editing of elastic
objects and cloth.

## Code Release

The source code for the synthetic experiments of the paper (elastic objects and cloth, simulated
with Genesis) — including pretrained checkpoints, a scene-editing demo, and training/evaluation
pipelines — is available on the
[**`synthetic_3dgsim`**](https://github.com/martius-lab/3DGSim/tree/synthetic_3dgsim) branch:

```bash
git clone -b synthetic_3dgsim git@github.com:martius-lab/3DGSim.git
```

See the branch [README](https://github.com/martius-lab/3DGSim/blob/synthetic_3dgsim/README.md) for
installation, demo, and evaluation instructions.

## Resources

- **Paper**: [arXiv:2503.24009](https://arxiv.org/abs/2503.24009)
- **Datasets** (elastic & cloth, Genesis): [Keeper](https://keeper.mpdl.mpg.de/d/a2c4e6f34ad24419947a/)
- **Pretrained models & sample data**: [HuggingFace `mzhobro/3dgsim`](https://huggingface.co/mzhobro/3dgsim)

## Roadmap

- [x] Release datasets
- [x] Release inference code + checkpoints (synthetic experiments, [`synthetic_3dgsim`](https://github.com/martius-lab/3DGSim/tree/synthetic_3dgsim))
- [ ] Release refactored codebase (v2)

## Citation

```bibtex
@inproceedings{zhobro2026_3dgsim,
  title     = {3DGSim: Learning 3D-Gaussian Simulators from RGB Videos},
  author    = {Zhobro, Mikel and Geist, Andreas Ren{\'e} and Martius, Georg},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```
