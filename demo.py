"""
Standalone demo for 3DGSim — Gaussian Particle World Model.

Downloads a pretrained model from HuggingFace and runs scene editing inference.
No Hydra or PyTorch Lightning required.

Usage:
    python demo.py --model elastic
    python demo.py --model cloth --data_path /path/to/scene
    python demo.py --model elastic --n_pred 20 --output_dir results/
"""

import argparse
import json
import warnings
import zipfile
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as tf
from einops import pack, unpack
from huggingface_hub import hf_hub_download, snapshot_download
from kornia import morphology as morph
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch import nn

# ── Model imports ────────────────────────────────────────────────────────────
# These rely on src being importable (i.e. run from the repo root).
from src.config import RootCfg, load_typed_root_config
from src.dataset.shims.crop_shim import apply_crop_shim
from src.dataset.types import BatchedTempExample, to_batched_temp_shim
from src.global_cfg import set_cfg
from src.model.decoder import get_decoder
from src.model.dynamic_model import get_dynamic_model
from src.model.encoder import get_encoder
from src.model.state_adapter import StateAdapter
from src.visualization.layout import add_border, hcat, vcat

# ── Constants ────────────────────────────────────────────────────────────────
REPO_ID = "mzhobro/3dgsim"

MODELS = {
    "elastic": "elastic_latent_4_1_12_33.1487_psnr",
    "cloth": "cloth_latent_4_1_8_26.98_psnr",
    "elastic_noseg": "noseg_ablation_elastic_explicit_4_1_12_32.66_psnr",
}

OBJECTS = ["dragon", "duck", "kawaii_demon", "pig", "spot", "worm", "bunny"]

DATASETS = {
    "elastic": "data/elastic",
    "elastic_unseen": "data/elastic_unseen",
    "cloth": "data/cloth",
}

# Maps dataset key -> directory name under datasets/
DATASET_DIRS = {
    "elastic": "soft_genesis_elastic",
    "elastic_unseen": "soft_genesis_elastic_unseen",
    "cloth": "soft_genesis_cloth",
}

ASSETS_DIR = Path("assets/data")


# ── DemoModel — lightweight nn.Module replacing ModelWrapper (LightningModule) ─
def _dilate(mask: torch.Tensor, kernel: torch.Tensor | None = None) -> torch.Tensor:
    if kernel is None:
        return mask
    mask, ps = pack([mask], "* c h w")
    mask = morph.dilation(mask, kernel)
    mask = unpack(mask, ps, "* c h w")[0]
    return mask


def _separate_state_masks(batch, keep_state_mask=False, keep_static_float=False, dilate_masks=True):
    """Split concatenated RGBA into RGB + masks (mirrors data_module.to_seperate_state_masks)."""
    batch["context"]["image"], state_mask = unpack(batch["context"]["image"], [[3], [-1]], "b t v * h w")
    batch["target"]["image"], state_mask_t = unpack(batch["target"]["image"], [[3], [-1]], "b t v * h w")

    kernel = torch.ones(3, 3).to(batch["context"]["image"].device) if dilate_masks else None
    if keep_state_mask:
        if torch.numel(state_mask) > 0:
            batch["context"]["state_mask"] = state_mask > 0.00
            batch["target"]["state_mask"] = state_mask_t > 0.00
        elif "state_mask" in batch["context"]:
            batch["context"]["state_mask"] = batch["context"]["state_mask"] > 0.00
            batch["target"]["state_mask"] = batch["target"]["state_mask"] > 0.00

    if keep_static_float and "static_float" in batch["context"]:
        batch["context"]["static_float"] = (_dilate(batch["context"]["static_float"], kernel) > 0.00).float()
        batch["target"]["static_float"] = (_dilate(batch["target"]["static_float"], kernel) > 0.00).float()
    return batch


def _concat_latent_dicts(*dicts):
    return {k: torch.cat([d[k] for d in dicts if k in d], dim=1) for k in dicts[0].keys()}


class DemoModel(nn.Module):
    """Minimal inference-only wrapper. Same attribute names as ModelWrapper for checkpoint compat."""

    def __init__(self, encoder, dynamic_model, decoder, state_adapter):
        super().__init__()
        self.encoder = encoder
        self.dynamic_model = dynamic_model
        self.decoder = decoder
        self.state_adapter = state_adapter

        # Build the data shim pipeline (mirrors get_data_shim in data_module.py)
        shims = [to_batched_temp_shim]
        if hasattr(encoder, "get_data_shim"):
            shims.append(encoder.get_data_shim())

        keep_static_float = not encoder.encoder_info.learn_static_mask
        keep_state_mask = not encoder.encoder_info.learn_background_mask
        shims.append(partial(_separate_state_masks, keep_state_mask=keep_state_mask, keep_static_float=keep_static_float))
        self._shims = shims

    @property
    def n_state(self) -> int:
        return self.dynamic_model.n_step_state if self.dynamic_model is not None else 1

    def data_shim(self, batch):
        for shim in self._shims:
            batch = shim(batch)
        return batch

    def encode(self, context):
        return self.encoder.encode(context, deterministic=False, global_step=0)

    def predict(self, past_latent, n_steps):
        if n_steps <= 0 or self.dynamic_model is None:
            return past_latent, 0.0
        future_latent, reg_loss = self.dynamic_model.rollout(past_latent, n_steps)
        return _concat_latent_dicts(past_latent, future_latent), reg_loss


# ── Local data helpers ───────────────────────────────────────────────────────
def _get_zip_name(dataset: str, obj: str, sample: int) -> str:
    """Return the zip filename for a given dataset/object/sample."""
    if dataset == "elastic_unseen":
        return f"elastic_unseen_{obj}_{sample}.zip"
    return f"{dataset}_{obj}.zip"


def get_scene_path(dataset: str, obj: str, sample: int) -> Path:
    """Return the scene path from assets/data/, downloading + unzipping from HF if needed."""
    if dataset == "elastic" and obj == "worm":
        sample = 1  # artifact :(

    dir_name = DATASET_DIRS[dataset]
    scene_path = ASSETS_DIR / dir_name / obj / str(sample)

    if scene_path.exists():
        return scene_path

    # Download zip from HF and unzip
    zip_name = _get_zip_name(dataset, obj, sample)
    zip_path = ASSETS_DIR / zip_name

    if not zip_path.exists():
        print(f"  Downloading {zip_name} from HuggingFace...")
        zip_path = Path(hf_hub_download(REPO_ID, f"data/{zip_name}", cache_dir=str(ASSETS_DIR)))

    print(f"  Unzipping {zip_name}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(ASSETS_DIR)

    if not scene_path.exists():
        raise FileNotFoundError(f"Unzipped {zip_name} but {scene_path} still not found.")

    return scene_path


# ── HuggingFace download helpers ─────────────────────────────────────────────
def download_model(name: str, cache_dir: str | None = None) -> tuple[Path, Path]:
    """Download checkpoint + config from HuggingFace. Returns (ckpt_path, cfg_path)."""
    full_name = MODELS[name]
    ckpt_path = hf_hub_download(
        REPO_ID,
        f"models/{full_name}.ckpt",
        cache_dir=cache_dir,
    )
    cfg_path = hf_hub_download(
        REPO_ID,
        f"configs/{full_name}.yaml",
        cache_dir=cache_dir,
    )
    return Path(ckpt_path), Path(cfg_path)


# ── Config loading (replaces Hydra) ─────────────────────────────────────────
def load_config(config_path: Path) -> tuple[RootCfg, DictConfig]:
    """Load a flat resolved YAML config without Hydra."""
    cfg_dict = OmegaConf.load(config_path)
    assert isinstance(cfg_dict, DictConfig)

    # Force test mode
    cfg_dict.mode = "test"

    # The global config singleton must be set before model construction
    # (encoder and PTv3 read from it during __init__).
    set_cfg(cfg_dict)

    cfg = load_typed_root_config(cfg_dict)
    return cfg, cfg_dict


# ── Model construction (replaces Hydra + Lightning trainer) ──────────────────
def build_model(cfg: RootCfg, device: torch.device) -> DemoModel:
    """Build the full model from config (same component wiring as scene_editing.py)."""
    state_adapter = StateAdapter(cfg.model.state_adapter)
    state_info = state_adapter.get_state_info()
    encoder_info = state_adapter.get_encoder_info()
    decoder_info = state_adapter.get_decoder_info()

    encoder, encoder_vis = get_encoder(cfg.model.encoder, cfg.dataset, state_info, encoder_info)
    dynamic_model = get_dynamic_model(cfg.model.dynamic_model, cfg.dataset, state_info) if cfg.dataset.n_step_predict > 0 else None
    decoder = get_decoder(cfg.model.decoder, cfg.dataset, decoder_info)

    model = DemoModel(
        encoder=encoder,
        dynamic_model=dynamic_model,
        decoder=decoder,
        state_adapter=state_adapter,
    )
    model = model.eval().to(device)
    return model


def load_checkpoint(model: DemoModel, ckpt_path: Path) -> None:
    """Load a Lightning checkpoint's state_dict into the model."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"  Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")


# ── Minimal data loader (replaces DataModule / DatasetGenesis / ViewSampler) ─
def load_scene_data(
    scene_path: Path,
    n_state: int,
    n_pred: int,
    context_views: list[int] | list[list[int]],  # camera groups rotate per timestep if nested
    target_views: list[int],
    near: float,
    far: float,
    speedup: int,
    with_seg: bool,
    image_shape: tuple[int, int],
    device: torch.device,
) -> BatchedTempExample:
    """Load a scene directly from disk — replaces the entire DataModule stack."""
    metadata = json.loads((scene_path / "metadata.json").read_text())
    n_cams = metadata["n_cams"]
    n_steps_total = metadata["n_steps"]
    to_tensor = tf.ToTensor()
    to_tensor_uint = tf.PILToTensor()

    # Camera parameters (static across timesteps)
    def get_normalized_intrinsics(intrinsics, width, height):
        K = np.array(intrinsics, dtype=np.float32)
        K[0, 0] /= width
        K[1, 1] /= height
        K[0, 2] /= width
        K[1, 2] /= height
        return K

    def get_w2c(c2w):
        return np.linalg.inv(np.array(c2w, dtype=np.float32))

    cams = list(range(n_cams))
    w2cs = [get_w2c(metadata[f"cam_{j}"]["extrinsics"]) for j in cams]
    Ks = [get_normalized_intrinsics(metadata[f"cam_{j}"]["intrinsics"], *metadata[f"cam_{j}"]["resolution"]) for j in cams]
    extrinsics_all = torch.tensor(np.array(w2cs), dtype=torch.float32)
    intrinsics_all = torch.tensor(np.array(Ks), dtype=torch.float32)

    file_tpl = str(scene_path / metadata["file_path_template"])

    # Determine timesteps
    n_total = n_state + n_pred
    if n_total * speedup > n_steps_total:
        n_pred = n_steps_total // speedup - n_state
        n_total = n_state + n_pred
        print(f"  Clamped n_pred to {n_pred} (scene has {n_steps_total} frames, speedup={speedup})")

    # If target_views is "all", use all cameras
    if not target_views or target_views == "all":
        target_views = list(range(n_cams))

    # Build context and target for each timestep
    contexts = []
    targets = []
    for t in range(n_total):
        step_i = t * speedup
        cam_indices = list(range(n_cams))

        # Load RGB images for all cameras
        rgbs = [to_tensor(Image.open(file_tpl.format(cam_ix=j, n_step=step_i, img_type="rgb") + ".png")) for j in cam_indices]
        rgbs = torch.stack(rgbs)  # (V, 3, H, W)

        # Load seg masks
        segs = [to_tensor_uint(Image.open(file_tpl.format(cam_ix=j, n_step=step_i, img_type="seg") + ".png")) for j in cam_indices]
        segs = torch.stack(segs)  # (V, 1, H, W)
        # Fix: if seg labels don't include 0, shift
        if 0 not in set(segs.flatten().tolist()):
            segs = segs - 1

        state_mask = (segs > 0).float()  # (V, 1, H, W)
        static_float = (segs == 3).float()

        n_views = len(target_views)
        target_idx = torch.tensor(target_views, dtype=torch.long)

        target_entry = {
            "extrinsics": extrinsics_all[target_idx],
            "intrinsics": intrinsics_all[target_idx],
            "near": torch.full((n_views,), near),
            "far": torch.full((n_views,), far),
            "index": target_idx,
            "image": rgbs[target_idx],
            "state_mask": state_mask[target_idx],
        }
        if with_seg:
            target_entry["static_float"] = static_float[target_idx]
        targets.append(target_entry)

        if t < n_state:
            # Rotate through the camera groups per timestep (as during training)
            # when context_views is a list of groups; otherwise use it as-is.
            if isinstance(context_views[0], list):
                views_t = context_views[t % len(context_views)]
            else:
                views_t = context_views
            ctx_idx = torch.tensor(views_t, dtype=torch.long)
            n_ctx = len(views_t)
            context_entry = {
                "extrinsics": extrinsics_all[ctx_idx],
                "intrinsics": intrinsics_all[ctx_idx],
                "near": torch.full((n_ctx,), near),
                "far": torch.full((n_ctx,), far),
                "index": ctx_idx,
                "image": rgbs[ctx_idx],
                "state_mask": state_mask[ctx_idx],
            }
            if with_seg:
                context_entry["static_float"] = static_float[ctx_idx]
            contexts.append(context_entry)

    # Stack across time: each value becomes (T, V, ...)
    keys = contexts[0].keys()
    context = {k: torch.stack([c[k] for c in contexts]) for k in keys}
    target = {k: torch.stack([t[k] for t in targets]) for k in targets[0].keys()}

    scene_name = f"{scene_path.name}_0:{n_total}"
    batch = {"scene": scene_name, "context": context, "target": target}

    # Apply crop shim (resize to target resolution + adjust intrinsics)
    batch = apply_crop_shim(batch, image_shape)

    # Add batch dim and move to device
    for group in ["context", "target"]:
        for k, v in batch[group].items():
            batch[group][k] = v.unsqueeze(0).to(device)  # (1, T, V, ...)
    batch["scene"] = [scene_name]

    return batch


# ── Video saving ─────────────────────────────────────────────────────────────
def save_video(frames: list[torch.Tensor], path: Path, fps: int = 12) -> None:
    """Save a list of (C, H, W) tensors as an mp4 video."""
    import moviepy.editor as mpy

    video = torch.stack(frames)
    video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
    # moviepy expects (T, H, W, C)
    video = np.transpose(video, (0, 2, 3, 1))
    clip = mpy.ImageSequenceClip(list(video), fps=fps)
    path.parent.mkdir(parents=True, exist_ok=True)
    clip.write_videofile(str(path), logger=None)
    print(f"  Saved: {path}")


def save_comparison_video(output, batch: BatchedTempExample, n_state: int, output_dir: Path, name: str = "comparison") -> None:
    """Save a GT vs prediction side-by-side video."""
    rgb_pred = output.color[0]  # (T, V, C, H, W)

    rgb_gt = batch["target"]["image"][0]
    frames = [add_border(vcat(hcat(*gt), hcat(*pred))) for gt, pred in zip(rgb_gt.cpu(), rgb_pred.cpu())]
    # frames = [add_border(hcat(*pred)) for pred in rgb_pred.cpu()]
    save_video(frames, output_dir / f"{name}.mp4")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="3DGSim demo — run from pretrained")
    parser.add_argument("--n_pred", type=int, default=None, help="Number of future steps to predict")
    parser.add_argument("--model", choices=list(MODELS.keys()), default="elastic", help="Which pretrained model to use")
    parser.add_argument("--dataset", choices=list(DATASETS.keys()), default="elastic_unseen", help="Dataset to download (ignored if --data_path is set)")
    parser.add_argument("--object", choices=OBJECTS, default="spot", help="Which object to use (ignored if --data_path is set)")
    parser.add_argument("--sample", type=int, default=0, help="Sample index (ignored if --data_path is set)")
    parser.add_argument("--output_dir", type=str, default="demo_output", help="Where to save videos")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    # local data and model options (skip HuggingFace download if set):
    parser.add_argument("--data_path", type=str, default=None, help="Path to a scene directory. If omitted, downloads sample data from HuggingFace.")
    parser.add_argument("--ckpt_path", type=str, default=None, help="Local checkpoint path (skips HuggingFace download)")
    parser.add_argument("--cfg_path", type=str, default=None, help="Local config path (skips HuggingFace download)")
    # Editing options:
    parser.add_argument("--n_add", type=int, default=0, help="Number of duplicate objects to add via scene editing (0 = no editing)")
    parser.add_argument("--remove_ground", action="store_true", help="Remove static/ground Gaussians from the scene")
    args = parser.parse_args()
    if args.object == "bunny" and args.dataset != "elastic_unseen":
        print(f"Warning: 'bunny' object is only in 'elastic_unseen' dataset, but got dataset='{args.dataset}'")
        exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Get model checkpoint + config (local or from HuggingFace)
    if args.ckpt_path and args.cfg_path:
        ckpt_path, cfg_path = Path(args.ckpt_path), Path(args.cfg_path)
        print("Using local model files:")
    else:
        print(f"Downloading model '{args.model}'...")
        ckpt_path, cfg_path = download_model(args.model, cache_dir="assets/model")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Config:     {cfg_path}")

    # 2. Locate scene data (unzips from assets/data/ if needed)
    if args.data_path is None:
        print(f"Locating {args.dataset}/{args.object}/{args.sample}...")
        scene_path = get_scene_path(args.dataset, args.object, args.sample)
    else:
        scene_path = Path(args.data_path)
    print(f"  Scene: {scene_path}")

    # 3. Load config
    print("Loading config...")
    cfg, cfg_dict = load_config(cfg_path)

    # 4. Build model
    print("Building model...")
    model = build_model(cfg, device)

    # 5. Load checkpoint
    print("Loading checkpoint...")
    load_checkpoint(model, ckpt_path)

    # 6. Determine prediction parameters from config
    n_state = cfg.dataset.n_step_state  # typically 4
    speedup = cfg.dataset.speedup  # typically 2
    near = cfg.dataset.near
    far = cfg.dataset.far
    # Match training-time camera sampling: the training view sampler rotates
    # through the camera groups in `context_views` (group t % n_groups at
    # timestep t, see camera_group_ix in the no_replacement sampler).
    context_views = cfg_dict.dataset.view_sampler.context_views
    if context_views is None:
        context_views = cfg_dict.dataset.view_sampler.test_context_views
    context_views = OmegaConf.to_container(context_views)
    with_seg = cfg_dict.dataset.with_seg
    image_shape = tuple(OmegaConf.to_container(cfg_dict.dataset.image_shape))

    # All cameras as target views
    target_views = list(range(12))

    # n_pred: from CLI or from the scene (all remaining frames)
    metadata = json.loads((scene_path / "metadata.json").read_text())
    max_pred = metadata["n_steps"] // speedup - n_state
    n_pred = args.n_pred if args.n_pred is not None else max_pred
    n_pred = min(n_pred, max_pred)

    print(f"  n_state={n_state}, n_pred={n_pred}, speedup={speedup}")
    print(f"  context_views={context_views}, near={near}, far={far}")

    # 7. Load scene data
    print("Loading scene data...")
    batch = load_scene_data(
        scene_path=scene_path,
        n_state=n_state,
        n_pred=n_pred,
        context_views=context_views,
        target_views=target_views,
        near=near,
        far=far,
        speedup=speedup,
        with_seg=with_seg,
        image_shape=image_shape,
        device=device,
    )

    # 8. Run inference
    print("Running inference...")
    batch = model.data_shim(batch)

    n_state = model.n_state
    n_pred = batch["target"]["image"].shape[1] - n_state

    with torch.no_grad():
        # 1. Encode
        past_latent = model.encode(batch["context"])
        print(f"  Encoded ({n_state} frames)")

        # 2. Scene Editing
        # Duplicate the segmented object in latent space by cloning Gaussians.
        extra_shift_if_z = 0.2  # to avoid complete overlap when shifting along z-axis
        shift = 0.3  # how much to shift the duplicated object (in world units)
        axis = 2  # which axis to shift along (0=x, 1=y, 2=z)
        n_add = args.n_add  # how many duplicates to add

        # If no static_float, objects are segmented using z coordinate of Gaussian means
        mask = past_latent.get("static_float", past_latent["means"][..., 2:] > 0.1)

        new_latent: dict = {}
        delta = shift + (extra_shift_if_z if axis == 2 else 0.0)

        for key, value in past_latent.items():
            if not isinstance(value, torch.Tensor):
                new_latent[key] = value
                continue

            if key == "means":
                added = []
                for i in range(n_add):
                    shifted = value.clone()
                    shifted[..., axis : axis + 1] += (i + 1) * delta * mask
                    added.append(shifted)
                new_latent[key] = torch.cat([value, *added], dim=2)
            else:
                new_latent[key] = torch.cat([value, *([value] * n_add)], dim=2)

        past_latent = new_latent
        print(f"  Edited: added {n_add} duplicate(s) of segmented object, shifted by {delta} along axis {axis})")

        # Remove ground/static Gaussians
        if args.remove_ground:
            ground_mask = past_latent.get("static_float", past_latent["means"][..., 2:] > 0.1)
            keep = ground_mask.squeeze(-1) > 0.5  # (B, T, G) — True for dynamic
            past_latent = {k: v[:, :, keep[0, 0]] if isinstance(v, torch.Tensor) and v.ndim >= 3 else v for k, v in past_latent.items()}
            print(f"  Removed ground: kept {keep.sum().item()} / {keep.numel()} Gaussians")

        # 3. Predict
        past_future_latent, _ = model.predict(past_latent, n_pred)
        print(f"  Predicted ({n_pred} future frames)")

        # 4. Decode
        extr = batch["target"]["extrinsics"]
        intr = batch["target"]["intrinsics"]
        near = batch["target"]["near"]
        far = batch["target"]["far"]
        h, w = batch["target"]["image"].shape[-2:]
        past_future_gaussians = model.decoder.prepare_gaussians(past_future_latent)
        output = model.decoder.forward(past_future_gaussians, extr, intr, near, far, (h, w))
        print("  Decoded")

    # 9. Save videos
    print("Saving comparison video...")
    name = f"{args.model}_{args.dataset}_{args.object}_{args.sample}"
    save_comparison_video(output, batch, n_state, output_dir, name=name)

    print(f"\nDone! Videos saved to {output_dir}/")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision("high")
    main()
