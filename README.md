# 3DGSim: Learning 3D-Gaussian Simulators from RGB Videos (ICML 2026)

[![arXiv](https://img.shields.io/badge/arXiv-2503.24009-b31b1b.svg)](https://arxiv.org/abs/2503.24009)
[![Datasets](https://img.shields.io/badge/Datasets-Keeper-2C6E91.svg)](https://keeper.mpdl.mpg.de/d/a2c4e6f34ad24419947a/)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-mzhobro%2F3dgsim-FFD21E.svg)](https://huggingface.co/mzhobro/3dgsim)
[![License: MIT](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

Official source code release for **3DGSim** ([paper](https://arxiv.org/abs/2503.24009)), a learned 3D-Gaussian simulator trained directly from multi-view RGB videos. 3DGSim reconstructs a scene into 3D Gaussians and rolls out its dynamics autoregressively, enabling photorealistic simulation of elastic objects and cloth without ground-truth 3D supervision.

> **Note:** This branch (`synthetic_3dgsim`) contains the codebase for the synthetic experiments of the paper (elastic objects and cloth, simulated with Genesis). A refactored v2 codebase will be released separately — see the [TODO](#todo) below.

## Installation

### uv (recommended)

`uv` reads dependencies from `pyproject.toml`. A single `uv sync` handles torch (cu118),
torch-scatter (pre-built from pyg), spconv-cu118, lpips, and all other deps. Only
flash-attn and diff-gaussian-rasterization still need `uv pip install` because they
require torch at build time (`--no-build-isolation`).

```bash
# 1. Sync everything declarative (torch+cu118, torch-scatter, spconv, lpips, …)
uv sync

# 2. CUDA extensions that need torch at build time
export TORCH_CUDA_ARCH_LIST="6.1;7.0;7.5;8.0+PTX"
export CUDA_HOME=/path/to/cuda-11.8                     # e.g. /is/software/nvidia/cuda-11.8
export PATH=$CUDA_HOME/bin:$PATH
uv pip install psutil                                   # build-time requirement of flash-attn
uv pip install --no-build-isolation "flash-attn==2.6.3" # pinned: precompiled wheel exists for cu118+torch2.4
uv pip install --no-build-isolation git+https://github.com/dcharatan/diff-gaussian-rasterization-modified
```

### conda (alternative)

```bash
conda create -n 3dgsim python=3.10 --yes
conda activate 3dgsim

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
pip install ninja
pip install flash-attn==2.6.3 --no-build-isolation

export TORCH_CUDA_ARCH_LIST="6.1;7.0;7.5;8.0+PTX"
pip install -r requirements.txt
pip install git+https://github.com/dcharatan/diff-gaussian-rasterization-modified
pip install git+https://github.com/jotix16/PerceptualSimilarity.git
pip install spconv-cu118
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.0+cu118.html
```

## Demo

`demo.py` downloads the sample scenes and pretrained checkpoints on first run and rolls out the simulator:

```bash
uv run demo.py --model elastic --dataset elastic_unseen --object bunny --n_add 2
uv run demo.py --model elastic --dataset elastic --object spot --n_add 4
uv run demo.py --model elastic --dataset elastic_unseen --object bunny --n_add 10 --remove_ground
uv run demo.py --model elastic --dataset elastic --object worm --n_add 10 --remove_ground
uv run demo.py --model elastic --dataset elastic --object worm --n_add 10
uv run demo.py --model elastic_noseg --dataset elastic --object spot --n_add 2  # no segmentation mask used for training or inference
uv run demo.py --model cloth --dataset cloth --object duck --n_add 0
```

## Sample Data

Sample scenes are stored as zips in `assets/data/`. They are downloaded and unzipped by `demo.py` on first run.

The directory structure after unzipping looks like this:
- `assets/data/soft_genesis_elastic/{dragon,duck,kawaii_demon,pig,spot,worm}/0/`
- `assets/data/soft_genesis_cloth/{dragon,duck,kawaii_demon,pig,spot,worm}/0/`
- `assets/data/soft_genesis_elastic_unseen/bunny/{0,1,2,3,4}/`

## Full Datasets

The full synthetic datasets (elastic and cloth, simulated with Genesis) are available on
[Keeper](https://keeper.mpdl.mpg.de/d/a2c4e6f34ad24419947a/). Scenes follow the same layout as the
sample data (`<object>/<sample>/cam_<i>/` + `metadata.json` per sample).

## Evaluation

Evaluate a pretrained model with `main.py` in test mode, using the resolved config that ships with
each checkpoint. Both the config and the sample data used below are downloaded by the first
`demo.py` run (`assets/model/` and `assets/data/`; also available directly from
[HuggingFace](https://huggingface.co/mzhobro/3dgsim)):

```bash
# Use the pretrained config as the hydra root config
SNAP=$(ls -d assets/model/models--mzhobro--3dgsim/snapshots/*/)
mkdir -p eval_cfg && cp $SNAP/configs/elastic_latent_4_1_12_33.1487_psnr.yaml eval_cfg/elastic.yaml

ulimit -n 65536  # evaluation opens all cameras/timesteps of a window at once
uv run python main.py --config-path eval_cfg --config-name elastic \
    mode=test \
    checkpointing.load=${SNAP}models/elastic_latent_4_1_12_33.1487_psnr.ckpt \
    "dataset.roots=[assets/data/soft_genesis_elastic_unseen]" "dataset.root_depths=[2]" \
    test.compute_scores=true test.save_video=false test.save_image=false \
    wandb.mode=disabled hydra.run.dir=eval_unseen
```

This evaluates the elastic model on the held-out bunny sample scene (unseen object, ~30 s).
PSNR/SSIM/LPIPS (with per-horizon breakdown and a static-scene baseline) are printed and dumped to
`eval_unseen/test/<timestamp>/scores/`. To evaluate on the full datasets, point `dataset.roots` at
a [Keeper](https://keeper.mpdl.mpg.de/d/a2c4e6f34ad24419947a/) download instead (the dataset holds
out a test split internally); for a single object, use its subdirectory (e.g.
`.../soft_genesis_elastic/duck`) with `dataset.root_depths=[1]`.

## TODO

- [x] Release datasets
- [ ] Release the new codebase (v2)

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{zhobro2026_3dgsim,
  title     = {3DGSim: Learning 3D-Gaussian Simulators from RGB Videos},
  author    = {Zhobro, Mikel and Geist, Andreas Ren{\'e} and Martius, Georg},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

## Acknowledgements

This codebase builds on [MVSplat](https://github.com/donydchen/mvsplat) and uses a [modified diff-gaussian-rasterization](https://github.com/dcharatan/diff-gaussian-rasterization-modified). Licensed under the MIT License (see `LICENSE`).
