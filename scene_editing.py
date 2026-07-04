import os
import random
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import hydra
import numpy as np
import torch
from colorama import Fore
from einops import rearrange, repeat
from jaxtyping import install_import_hook
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from src.dataset.types import AnyExample, BatchedTempExample, to_batched_example
from src.model.types import Gaussians
from src.visualization.camera_trajectory.interpolation import interpolate_extrinsics, interpolate_intrinsics


def set_deterministic(seed=0):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


import wandb

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import RootCfg, load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.dataset.shims.crop_shim import apply_crop_shim
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.ema import EMA, EMAModelCheckpoint
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.dynamic_model import get_dynamic_model
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import AnyExample, BatchedTempExample, ModelWrapper
    from src.model.state_adapter import StateAdapter
    from src.visualization.annotation import add_label
    from src.visualization.layout import add_border, hcat, vcat


class ModelWrapperVis(ModelWrapper):
    def test_step(self, batch1: BatchedTempExample, batch_idx):
        # torch.cuda.empty_cache()
        batch = apply_crop_shim(batch1, (128, 128))
        batch: AnyExample = self.data_shim(batch)
        assert batch["context"]["image"].shape[1] == self.n_state
        assert batch["context"]["image"].shape[0] == 1

        scene = batch["scene"][0]
        n_state = self.n_state
        n_pred = batch["target"]["image"].shape[1] - n_state
        print(f"{batch_idx} scene: {scene}, n_state: {n_state}, n_pred: {n_pred}")

        # Encode Predict Decode
        past_latent = self.encode(batch["context"], test=True)
        print("\t 1. encoded")

        # scene editing
        # offcenter obj
        # past_latent["means"][..., 0:1] += 0.2 * past_latent["static_float"]

        # lower ground
        # past_latent["means"][..., 2:3] -= 1.2 * (past_latent["static_float"] < 0.5).float()

        # Duplicate object
        new_latent = {}

        # Retrieve the static_float mask, which indicates segmentation.
        mask = past_latent.get("static_float", past_latent["means"][..., 2:] > 0.1)

        for key, value in past_latent.items():
            # If the value is None, simply pass it along.
            if value is None:
                new_latent[key] = None
                continue

            # For tensor values, create a shifted version if needed.
            N_add = 2
            I = 2
            shift = 0.3
            N_added = []
            if isinstance(value, torch.Tensor):
                if key == "means":
                    # Create a shifted version of the "means"
                    added = []
                    for i in range(N_add):
                        shifted = value.clone()  # T, G, 3
                        # This only affects the spatial locations where mask > 0.
                        shifted[..., I : I + 1] += (i + 1) * (shift + (0.15 if I == 2 else 0.0)) * mask

                        # T = shifted.shape[1]
                        # print(f"T={T}, G={shifted.shape[1]}, I={I}, N_add={N_add}")
                        # direction = [1.0, -1.0]
                        # for t in range(T):
                        #     shifted[:, t] += (
                        #         # t * value.new_tensor([0.0, 0.0, 0.04]) * mask[:, t] * direction[i % 2]
                        #         t
                        #         * value.new_tensor([0.04, 0.04, 0.00])
                        #         * mask[:, t]
                        #         * direction[i % 2]
                        #     )
                        added.append(shifted)
                    # value[..., I : I + 1] -= shift * mask
                    # Concatenate original and shifted versions along the feature dimension (last dim).
                    new_val = torch.cat([value, *added], dim=2)
                # elif key == "harmonics":
                #     # make the new color blue
                #     new_value = value.clone()  # G, 3, d_sh
                #     new_value[..., :, 0] = new_value.new_tensor([0.0, 0.0, 1.0])
                #     # new_value[..., :, 1:] = 0.0
                #     new_val = torch.cat([value, new_value], dim=2)
                else:
                    # For all other features, simply duplicate the tensor.
                    new_val = torch.cat([value, *([value] * N_add)], dim=2)

                new_latent[key] = new_val
            else:
                # For non-tensor values, just copy over.
                new_latent[key] = value
            N_added.append(new_latent[key].shape[2] - past_latent[key].shape[2])

        # create two objects
        N_added = new_latent["means"].shape[2] - past_latent["means"].shape[2]
        past_latent = new_latent

        print("\t 2. Duplicated")

        # 2. Predict
        past_future_latent, reg_loss = self.predict(past_latent, n_pred, test=True)
        print("\t 3. Simulated")

        # DECODE
        # past_future_output, past_future_gaussians = self.decode(
        #     past_future_latent, batch["target"], test=False, with_extras=True, return_constant_pred=False
        # )
        # custom decode
        # Decode
        extr = batch["target"]["extrinsics"]
        intr = batch["target"]["intrinsics"]
        near = batch["target"]["near"]
        far = batch["target"]["far"]
        past_future_gaussians = self.decoder.prepare_gaussians(past_future_latent)
        # change color of the added ones
        # if sum(N_added) > 0:
        #     mask = past_future_gaussians.static_float
        #     for N_ in range(N_added):
        #     past_future_gaussians.harmonics[:, :, -N_added:, :, 0] = extr.new_tensor([0.0, 0.0, 1.0]) * mask

        past_future_output = self.decoder.forward(past_future_gaussians, extr, intr, near, far, (128, 128))
        print("\t 4. Decoded")
        print("\t 5. Saving..")

        # Mask to white the ground truth (mask in higher resolution then scale down to avoid aliasing)
        def mask_to_white(img, mask):
            # img (B, T, V, C, H, W)
            # mask (B, T, V, 1, H, W)
            img = img.clone()
            img[~repeat(mask.squeeze(3), "... h w -> ... 3 h w").bool()] = 1.0
            return img

        batch1["target"]["image"] = mask_to_white(
            batch1["target"]["image"], batch1["target"].get("state_mask")
        )
        batch1 = apply_crop_shim(batch1, (128, 128))

        # GT and Prediction
        # Create video
        rgb_pred = past_future_output.color[0]  # (T, V, C, H, W)
        rgb_gt = batch1["target"]["image"][0]

        comparison = [
            add_border(
                vcat(
                    hcat(*img_gt),
                    hcat(*img_pred),
                )
            )
            for img_gt, img_pred in zip(rgb_gt, rgb_pred)
        ]
        self.log_video(
            f"test_videos/{str(self.test_cfg.output_path).split('/')[-1]}",
            comparison,
            loop_reverse=False,
            caption=f"{scene}  N_state={n_state}, N_pred={self.n_pred}, N_future={len(rgb_pred)-n_state}",
            fps=12,
        )

        self.render_interpolation(batch1, past_future_gaussians, smooth=False)

    def render_interpolation(self, batch: BatchedTempExample, past_future_gaussians: Gaussians, smooth=True):
        num_frames = 100
        scene, n_fut = (batch["scene"][0], batch["target"]["image"].shape[1] - self.n_state)

        batch = to_batched_example(batch)
        gaussians = past_future_gaussians.to_batched_gaussians()

        _, v, _, _ = batch["context"]["extrinsics"].shape
        ix1, ix2 = 0, 1

        # Start end extrinsics and intrinsics
        ex0, ex1 = batch["context"]["extrinsics"][0, ix1], batch["context"]["extrinsics"][0, ix2]
        # ex1[..., 2] += 2.0  # ex1 should be 1 m higher
        int0, int1 = batch["context"]["intrinsics"][0, ix1], batch["context"]["intrinsics"][0, ix2]

        # TODO: Interpolate near and far planes?
        _, _, _, h, w = batch["context"]["image"].shape
        near = repeat(batch["context"]["near"][0, 0], " -> b 1", b=num_frames)
        far = repeat(batch["context"]["far"][0, 0], " -> b 1", b=num_frames)

        ex11 = ex0.new_tensor(
            [
                [-0.0, 0.2228248, -0.9748585, 3.5],
                [1.0, 0.0, 0.0, 0.0],
                [-0.0, -0.9748585, -0.2228248, 0.99999994],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        ex00 = ex0.new_tensor(
            [
                # [-1.0, -0.0, -0.0, -0.0],
                # [-0.0, 0.457348, -0.88928777, 3.5000002],
                # [-0.0, -0.88928777, -0.457348, 2.0],
                # [0.0, 0.0, 0.0, 1.0],
                [-1.0, -0.0, -0.0, -0.0],
                [0.0, 0.76822126, -0.6401844, 1.5],
                [-0.0, -0.6401844, -0.76822126, 2.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        ex0[..., :, :] = ex00
        ex1[..., :, :] = ex11

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(ex0, ex1, t)
            intrinsics = interpolate_intrinsics(int0, int1, t)
            return extrinsics[None], intrinsics[None]

        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2
            # t = -1 * (torch.cos(-np.pi / 2.0 + 0.5 * torch.pi * (t + 1))) / 2

        extrinsics, intrinsics = trajectory_fn(t)
        extrinsics = rearrange(extrinsics[:1], "b v i j -> v b i j")
        intrinsics = rearrange(intrinsics[:1], "b v i j -> v b i j")

        # Interpolate the Gaussians (assume first dim is time)
        t_g = torch.linspace(0, 1, gaussians.means.shape[0], dtype=torch.float32, device=self.device)
        corresponding_ix = torch.abs(t.unsqueeze(1) - t_g).argmin(dim=1)  # gaussian index for each time in t
        gaussians = gaussians[corresponding_ix]

        output_prob = self.decoder.forward(gaussians, extrinsics, intrinsics, near, far, (h, w), "depth")

        images = [*output_prob.color[:, 0]]
        self.log_video(
            f"test_videos/teaser_{str(self.test_cfg.output_path).split('/')[-1]}",
            images,
            loop_reverse=False,
            caption=f"{scene}  N_state={self.n_state}, N_pred={self.n_pred}, N_future={n_fut}",
            fps=20,
        )


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


def green(text: str) -> str:
    return f"{Fore.GREEN}{text}{Fore.RESET}"


def load_logger(cfg_dict: DictConfig, output_dir: Path, prev_exp_dir: Path = None, test_path: Path = None):
    """
    In case output_dir has a wandb directory, we either resume or fork the experiment from there.
    If we fork, we also create a new directory with the current timestamp, to keep the wandb directory clean.
    """

    def get_wandb_id_for_resume(prev_exp_dir: Path):
        latest_wandb_id = None
        wandb_dir = prev_exp_dir / "wandb"
        if wandb_dir.exists():
            list_of_files = list(wandb_dir.glob("run-*"))
            if list_of_files:
                latest_wandb_id = str(max(list_of_files, key=os.path.getctime)).split("-")[-1]
        return latest_wandb_id

    with_wandb = cfg_dict.wandb.mode != "disabled"
    if with_wandb:
        # Defaults for new exp
        wandb_id = wandb.util.generate_id()
        group_id = f"{wandb_id}_group"
        resume = None
        dscp = "New exp"
        wandb_dir = output_dir

        _wandb_id = get_wandb_id_for_resume(prev_exp_dir)
        if _wandb_id is not None:
            # resume from the previous run if not in test mode and if asked for resume, otherwise fork
            resume = "must" if (cfg_dict.mode != "test" and cfg_dict.checkpointing.resume) else False
            wandb_id = _wandb_id if resume else wandb_id
            group_id = f"{wandb_id}_group" if resume else f"{_wandb_id}_group"
            dscp = "Resume" if resume else ("Fork" if cfg_dict.mode != "test" else "Test")
            wandb_dir = (
                output_dir / "wandb" / f"{test_path.name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
                if cfg_dict.mode == "test"
                else wandb_dir
            )
            wandb_dir.mkdir(parents=True, exist_ok=True)

        wandb_extra_kwargs = {"id": wandb_id, "group": f"{cfg_dict.wandb.name}_{group_id}", "resume": resume}
        print(green(f"\n\n{dscp}: wandb-id: {wandb_id}, group-id: {cfg_dict.wandb.name}_{group_id}\n\n"))

        extra_name = (
            # f"{output_dir.parent.name}/{output_dir.name}"
            ""
            if cfg_dict.mode != "test"
            else f" (test_{test_path.name}_{time.strftime('%Y-%m-%d_%H-%M-%S')})"
        )
        logger = WandbLogger(
            entity=cfg_dict.wandb.entity,
            project=cfg_dict.wandb.project,
            mode=cfg_dict.wandb.mode,
            name=f"{cfg_dict.wandb.name}{extra_name}",
            tags=cfg_dict.wandb.get("tags", None),
            log_model=cfg_dict.wandb.get("log_model", False),
            save_dir=wandb_dir,
            config=OmegaConf.to_container(cfg_dict),
            **wandb_extra_kwargs,
        )

        # On rank != 0, wandb.run is None.
        if wandb.run is not None:
            wandb.run.log_code("src")
    else:
        logger = LocalLogger()

    return logger, with_wandb


def prepare_outdir(cfg_dict: DictConfig):
    """
    In case the output_dir/checkpoints exists, we either resume or fork the experiment from there.
    If we fork, we also create a new directory with the current timestamp, to keep the checkpoints directory clean.

    """
    if cfg_dict.output_dir is None:
        # to override it: [python main.py ... hydra.run.dir="/path/to/dir"]
        output_dir = Path(hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"])
    else:  # for resuming
        output_dir = Path(cfg_dict.output_dir)

    prev_exp_dir = output_dir
    if (output_dir / "checkpoints").exists():
        if cfg_dict.mode == "test":
            print(cyan(f"Testing: "))
        elif not cfg_dict.checkpointing.resume:
            print(cyan("Forking:"))
            output_dir = output_dir / f"fork/{time.strftime('%Y-%m-%d_%H-%M-%S')}"
        else:
            print(cyan("Resuming:\n Checkpoints will be saved in the same directory."))
        print(cyan(f" Previous experiment directory: {prev_exp_dir}"))
    else:
        print(cyan("Starting a new experiment:"))
    print(cyan(f" Output directory: {output_dir}"))

    os.makedirs(output_dir, exist_ok=True)

    return output_dir, prev_exp_dir


def get_test_output_path(output_dir, output_path: Path | None):
    output_path = str(output_path) if output_path is not None else None
    if output_path is None:
        output_path = output_dir / f"test/{time.strftime('%Y-%m-%d_%H')}"
    elif output_path.startswith("/"):
        output_path = output_path  # absolute path provided
    else:
        output_path = output_dir / output_path  # local paths are relative to output_dir
    print(cyan(f" Test output path: {output_path}"))
    return output_path


def get_checkpoint_from_prev_exp(prev_exp_dir: Path):
    latest_checkpoint = None
    checkpoint_dir = prev_exp_dir / "checkpoints"
    if checkpoint_dir.exists():
        list_of_files = [p for p in checkpoint_dir.glob("*.ckpt") if "last" not in p.stem]
        latest_checkpoint = max(list_of_files, key=os.path.getctime)
    return latest_checkpoint


def get_checkpoint(path: str | None, wandb_cfg: dict, prev_exp_dir: Path):
    checkpoint_path = get_checkpoint_from_prev_exp(prev_exp_dir) or update_checkpoint_path(path, wandb_cfg)
    if checkpoint_path is not None:
        print(cyan(f" Initialize from checkpoint: {checkpoint_path}.\n"))
    return checkpoint_path


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="main",
)
def train(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)

    # Set up the output directory and check for previous experiment
    cfg.output_dir, prev_exp_dir = prepare_outdir(cfg_dict)
    cfg.test.output_path = get_test_output_path(cfg.output_dir, cfg.test.output_path)

    # Prepare the checkpoint for loading.
    checkpoint_path = get_checkpoint(cfg.checkpointing.load, cfg.wandb, prev_exp_dir)

    # Set up logging with wandb.
    logger, with_wandb = load_logger(cfg_dict, cfg.output_dir, prev_exp_dir, cfg.test.output_path)

    # Set up checkpointing.

    # This allows the current step to be shared with the data loader processes.
    step_tracker = StepTracker()

    ddp_strategy = DDPStrategy(find_unused_parameters=True)
    trainer = Trainer(
        detect_anomaly=cfg.trainer.detect_anomaly,
        max_epochs=cfg.trainer.max_epochs,
        accelerator="gpu",
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        logger=logger,
        devices="auto",
        num_nodes=cfg.trainer.num_nodes,
        strategy=ddp_strategy if torch.cuda.device_count() > 1 else "auto",
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=1 if cfg.trainer.val_check_interval < 1.0 else None,
        enable_progress_bar=cfg.mode == "test",
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        # enable_checkpointing=False,
    )
    trainer.logger.log_hyperparams(asdict(cfg))
    torch.manual_seed(cfg_dict.seed + trainer.global_rank)
    # set_deterministic(cfg_dict.seed + trainer.global_rank)

    state_adapter = StateAdapter(cfg.model.state_adapter)
    state_info = state_adapter.get_state_info()
    encoder_info = state_adapter.get_encoder_info()
    decoder_info = state_adapter.get_decoder_info()

    encoder, encoder_vis = get_encoder(cfg.model.encoder, cfg.dataset, state_info, encoder_info)
    dynamic_model = (
        get_dynamic_model(cfg.model.dynamic_model, cfg.dataset, state_info)
        if cfg.dataset.n_step_predict > 0
        else None
    )
    decoder = get_decoder(cfg.model.decoder, cfg.dataset, decoder_info)

    model_kwargs = {
        "optimizer_cfg": cfg.optimizer,
        "test_cfg": cfg.test,
        "train_cfg": cfg.train,
        "encoder": encoder,
        "encoder_visualizer": encoder_vis,
        "dynamic_model": dynamic_model,
        "decoder": decoder,
        "state_adapter": state_adapter,
        "losses": get_losses(cfg.loss),
        "step_tracker": step_tracker,
        "detect_anomaly": cfg.trainer.detect_anomaly,
    }

    model_wrapper = ModelWrapperVis(**model_kwargs)
    # model_wrapper.load_state_dict(torch.load("checkpoints/tgs_weights.pth"), strict=False)

    # # TODO: REQUIRED because we plan to remove part of the model in encoder_costvolume/depth_predictor
    # # but still want to be able to load the old checkpoints for baseline comparison
    # model_wrapper.strict_loading = False

    # Update for multi-step training curriculum
    if cfg.train.dyn_model_schedule.n_max_pred > 1 and cfg.dataset.n_step_predict > 0:
        cfg.dataset.n_step_predict *= cfg.train.dyn_model_schedule.n_max_pred

    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader,
        step_tracker,
        global_rank=trainer.global_rank,
    )
    if cfg.dataset.test_validity:
        data_module.check_validity()
        return

    assert cfg.mode == "test"
    trainer.test(
        model_wrapper,
        datamodule=data_module,
        ckpt_path=checkpoint_path,
    )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision("high")
    # torch.autograd.set_detect_anomaly(True)
    # set_deterministic()

    train()
