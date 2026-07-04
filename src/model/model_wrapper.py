import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import Optional, Protocol, Union, runtime_checkable

import matplotlib
import matplotlib.pyplot as plt
import moviepy.editor as mpy
import numpy as np
import torch
import wandb
from einops import pack, rearrange, reduce, repeat, unpack  # noqa
from imageio import imread
from jaxtyping import Bool, Float
from plyfile import PlyData, PlyElement
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, nn, optim
from torch.utils.checkpoint import checkpoint

matplotlib.use("Agg")


from ..dataset.data_module import get_data_shim
from ..dataset.types import (
    AnyExample,
    BatchedExample,
    BatchedTempExample,
    BatchedTempViews,
    to_batched_example,
)
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..geometry.projection import get_means_from_depth
from ..global_cfg import get_cfg
from ..loss import Loss
from ..misc.benchmarker import Benchmarker
from ..misc.image_io import prep_image, save_image, save_video  # noqa
from ..misc.LocalLogger import get_log_path
from ..misc.step_tracker import StepTracker
from ..model.types import Gaussians, SpatialLatentsDict
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import interpolate_extrinsics, interpolate_intrinsics
from ..visualization.camera_trajectory.wobble import generate_wobble, generate_wobble_transformation
from ..visualization.color_map import apply_color_map_to_image
from ..visualization.layout import add_border, hcat, vcat
from ..visualization.validation_in_3d import render_cameras, render_projections
from .decoder.decoder import Decoder, DepthRenderingMode
from .dynamic_model import DynamicModel
from .encoder import Encoder
from .encoder.visualization.encoder_visualizer import EncoderVisualizer
from .state_adapter import StateAdapter


@dataclass
class OptimizerCfg:
    lr: float
    warm_up_steps: int
    cosine_lr: bool
    cosine_annealing_warmup: bool


@dataclass
class TestCfg:
    output_path: Path | None
    compute_scores: bool
    save_image: bool
    save_video: bool
    eval_time_skip_steps: int


@dataclass(kw_only=True)
class DynModelTrainingScheduleCfg:
    # Static Prediction regularizer
    reg_weight_dict: dict[str, float] = field(
        default_factory=lambda: {
            "means": 0.0,
            "rotations": 0.0,
            "harmonics": 0.0,
            "opacities": 0.0,
            "scales": 0.0,
            "features": 0.0,
        }
    )
    N0: int = 0  # 500  # where to end static dynamics regularizer

    # Multi-Step Prediction Scheduler
    N1: int = 1  # 8000  # from where to start linear warmup for probability
    N2: int = 2  # 20000  # where to end linear warmup for probability
    n_max_pred: int = 4  # max multiple of steps to predict
    n_max_grad_pred: int = 2  # max multiple of steps to predict with grad_recording
    probability: float = (
        0.8  # probability of uniformly predicting n_pred in [n_max_grad_pred, n_max_step_pred]
    )

    # 0 ------------------ N0 ------------------ N1 ------------------ N2 ------------------ max_train_steps
    #  static_regulizer       1-n_max_grad_pred  | uniformly sample n_pred in [n_max_grad_pred, n_max_step_pred] with probability δ
    #  l_reg=λ * reg * Δ_f |  linear_warmup      | δ: 0->1              |     δ = 1.

    def __post_init__(self):
        print("\nDynModelTrainingScheduleCfg:")
        assert 0 <= self.probability <= 1.001, "Probability must be between 0 and 1"

        if self.n_max_pred < self.n_max_grad_pred:
            self.n_max_pred = self.n_max_grad_pred
            print(f" n_max_pred < n_max_grad_pred, setting n_max_pred = {self.n_max_grad_pred}")

        self.N0 = 0  # TODO: remove this, no regularizer
        if self.N0 < 0:
            self.N0 = 0
            print(" Skip static prediction regularizer")

        if self.N1 < 0 and self.N2 < 0:
            self.N1, self.N2 = self.N0, self.N0
            self.n_max_pred = self.n_max_grad_pred
            print(f" Default {self.n_max_grad_pred}-step pred from global-step={self.N0}.")
        else:
            if self.N1 < 0:
                self.N1 = self.N0
                print(" Skip linear warmup to multi-step predictions with grad recording")

            if self.N2 < 0:
                self.N2 = self.N1
                self.n_max_pred = self.n_max_grad_pred
                print(" Skip linear warmup to  multi-step prediction with grad cutting.\n")

        assert self.N0 <= self.N1 <= self.N2, "N0 < N1 < N2"
        assert self.N0 >= 0, "N0 must be positive"
        assert self.N1 >= 0, "N1 must be positive"
        assert self.N2 >= 0, "N2 must be positive"
        print(f" N0={self.N0}, N1={self.N1}, N2={self.N2}")
        print(
            f" n_max_pred={self.n_max_pred}, n_max_grad_pred={self.n_max_grad_pred}, probability={self.probability}\n"
        )

    def get_staic_reg_weight_dict(self, global_step: int):
        if global_step >= self.N0:
            # catches also cases where N0 is 0 or -1
            alpha = 0.0
        else:
            #  Linear warmup
            # alpha = 1.0 - (global_step / self.N0)

            # Cosine warmup
            alpha = 0.5 * (1.0 + np.cos(np.pi * global_step / self.N0))

        return {k: v * alpha for k, v in self.reg_weight_dict.items()}

    def get_nstep_pred(self, global_step: int):
        if global_step < self.N0:
            # is ignored for now
            n_full_pred = 0

        elif global_step >= self.N0 and global_step < self.N1:
            # encoder warmup
            n_full_pred = 0

        elif global_step >= self.N1 and global_step < self.N2:
            # nstep prediction warmup
            n_full_pred = 1 + (self.n_max_grad_pred - 1) * min(
                max((global_step - self.N1) / (self.N2 - self.N1), 0), 1
            )
        else:
            n_full_pred = self.n_max_pred

        n_full_pred = int(n_full_pred)
        if n_full_pred > 0:

            if not np.random.rand() > self.probability:
                n_full_pred = np.random.randint(1, n_full_pred + 1)

            # randomly choose the number of steps to predict
            n_grad_predict = min(self.n_max_grad_pred, n_full_pred)
            n_nograd_predict = n_full_pred - n_grad_predict
        else:
            # force one step prediction
            n_grad_predict = 0
            n_nograd_predict = 0

        needs_update = n_grad_predict != self.n_max_pred
        return n_grad_predict, n_nograd_predict, needs_update

    def batch_update(self, n_state: int, n_pred: int, batch: BatchedTempExample, global_step: int):
        n_grad_predict, n_nograd_predict, needs_update = self.get_nstep_pred(global_step)

        n_grad_steps = n_grad_predict * n_pred
        n_nograd_steps = n_nograd_predict * n_pred
        n_full_steps = n_grad_steps + n_nograd_steps

        if needs_update:
            # choose random start index (np.random.randint is exclusive for the upper bound -> +1)
            B = batch["context"]["image"].shape[0]
            # start_idx = torch.randint(0, n_pred * self.n_max_pred - n_full_steps + 1, (B,))
            start_idx = np.random.randint(0, n_pred * self.n_max_pred - n_full_steps + 1)
            state_end_idx = start_idx + n_state

            start_future_grad_idx = state_end_idx + n_nograd_steps
            end_future_grad_idx = start_future_grad_idx + n_grad_steps

            # remove no grad targets
            new_batch = dict(
                context=batch["context"],
                target={
                    k: torch.cat(
                        [
                            v[:, start_idx:state_end_idx],
                            v[:, start_future_grad_idx:end_future_grad_idx],
                        ],
                        dim=1,
                    )
                    for k, v in batch["target"].items()
                },
                scene=[
                    self.get_slice_scene(s, start_idx, n_state, n_pred, n_nograd_predict)
                    for s in batch["scene"]
                ],
            )
        else:
            new_batch = batch
        return new_batch, n_grad_steps, n_nograd_steps

    def get_slice_scene(self, scene: str, start_idx: int, st_end: int, pred_start: int, pred_end: int):
        try:
            if "_" in scene and ":" in scene:
                st_start = int(scene.split("_")[-1].split(":")[0]) + start_idx

                scene = "_".join(scene.split("_")[:2]) + f"_{st_start:05}-{st_end}"
                if pred_start != pred_end:
                    scene += f":{pred_start:05}-{pred_end}"
            return scene
        except Exception as e:
            print(e)
            return scene

    def get_reg_loss_dict(self, past_future_latent: SpatialLatentsDict, n_state: int, global_step: int):
        weights = self.get_staic_reg_weight_dict(global_step)
        reg_loss_dict = {}
        for k, v in past_future_latent.items():
            weight = weights.get(k, 0.0)
            if weight > 1e-8:
                v0 = v[:, n_state - 1 : n_state]
                v1 = v[:, n_state:]
                reg_loss_dict[k] = weight * torch.mean((v1 - v0.detach()) ** 2)

        return reg_loss_dict


@dataclass
class TrainCfg:
    depth_mode: DepthRenderingMode | None
    extended_visualization: bool
    print_log_every_n_steps: int
    dyn_model_schedule: DynModelTrainingScheduleCfg
    gradient_checkpointing: bool
    freeze_encoder: bool = False


@runtime_checkable
class TrajectoryFn(Protocol):
    def __call__(
        self,
        t: Float[Tensor, " t"],
    ) -> tuple[
        Float[Tensor, "batch view 4 4"],  # extrinsics
        Float[Tensor, "batch view 3 3"],  # intrinsics
    ]:
        pass


class ModelWrapper(LightningModule):
    logger: Optional[WandbLogger]
    encoder: Encoder
    encoder_visualizer: Optional[EncoderVisualizer]
    dynamic_model: DynamicModel | None
    decoder: Decoder
    state_adapter: StateAdapter
    losses: nn.ModuleList
    optimizer_cfg: OptimizerCfg
    test_cfg: TestCfg
    train_cfg: TrainCfg
    step_tracker: StepTracker | None
    detect_anomaly: bool

    def __init__(
        self,
        optimizer_cfg: OptimizerCfg,
        test_cfg: TestCfg,
        train_cfg: TrainCfg,
        encoder: Encoder,
        encoder_visualizer: Optional[EncoderVisualizer],
        dynamic_model: DynamicModel | None,
        decoder: Decoder,
        state_adapter: StateAdapter,
        losses: list[Loss],
        step_tracker: StepTracker | None,
        detect_anomaly: bool = False,
    ) -> None:
        super().__init__()
        self.optimizer_cfg = optimizer_cfg
        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        self.step_tracker = step_tracker
        self.detect_anomaly = detect_anomaly

        # Set up the model.
        self.encoder = encoder
        if self.train_cfg.freeze_encoder:
            print("Freezing encoder weights.")
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        self.encoder_visualizer = encoder_visualizer
        self.dynamic_model = dynamic_model
        self.decoder = decoder
        self.state_adapter = state_adapter

        self.data_shim = get_data_shim(self.encoder)
        self.losses = nn.ModuleList(losses)

        # This is used for testing.
        self.benchmarker = Benchmarker()
        self.eval_cnt = 0

        if self.test_cfg.compute_scores:
            self.test_step_outputs = {}
            self.time_skip_steps_dict = {
                "encoder": 0,
                "decoder": 0,
                "dyn_sys": 0,
            }
        print(f"Gradient Checkpointing: {self.train_cfg.gradient_checkpointing}")

    def get_reg_loss(self, past_future_latent: SpatialLatentsDict, n_state: int):
        # models: Encoder, DynamicModel, Decoder, GaussianAdapter
        # check if they have the method
        reg_loss = past_future_latent["means"].new_zeros(1)
        if hasattr(self.state_adapter, "get_reg_loss"):
            reg_loss += self.state_adapter.get_reg_loss()

        # In case we are training the dynamic model with reg-loss
        dyn_reg_loss = 0.0
        if self.dynamic_model is not None:
            dyn_model_reg_loss_dict = self.train_cfg.dyn_model_schedule.get_reg_loss_dict(
                past_future_latent, n_state, self.global_step
            )
            if len(dyn_model_reg_loss_dict) > 0:
                dyn_reg_loss = sum(dyn_model_reg_loss_dict.values())

                # Log the reg-losses
                for k, v in dyn_model_reg_loss_dict.items():
                    self.log(f"reg_loss/{k}", v.item())
                self.log("reg_loss/dyn_reg_loss", dyn_reg_loss.item())

        reg_loss += dyn_reg_loss
        self.log("reg_loss/loss_regularize", reg_loss.item())

        return reg_loss

    def dynamic_model_forward(
        self,
        past_latent_dict: SpatialLatentsDict,
        n_steps: int,
        n_pred_nograd=0,
        test=False,
    ):
        # Simulate the future.
        with self.benchmarker.time("dyn_sys", num_calls=n_steps, test=test):

            if n_pred_nograd > 0:
                with torch.no_grad():
                    past_latent_dict, _ = self.dynamic_model.rollout(
                        past_latent_dict,
                        n_pred_nograd,
                        with_past=True,
                    )
            future_latent_dict, reg_loss = self.dynamic_model.rollout(past_latent_dict, n_steps)

        if self.detect_anomaly:
            assert all(
                [torch.isfinite(v).all() for k, v in future_latent_dict.items()]
            ), "NaNs in Dynamic Model Outputs"
        return future_latent_dict, reg_loss

    def encode(self, context: BatchedTempViews, test: bool):
        with self.benchmarker.time("encoder", test=test):

            past_latents = self.encoder.encode(context, deterministic=False, global_step=self.global_step)
        return past_latents

    def predict(
        self,
        past_latents: SpatialLatentsDict,
        n_steps: int,
        n_pred_nograd=0,
        test=False,
    ):
        # Concatenate past and future. B,T+Tnew, Gaussian.. -> (B,T+Tnew), Gaussian..
        reg_loss = 0.0
        past_future_latent = past_latents
        if n_steps > 0:
            future_latents, reg_loss = self.dynamic_model_forward(past_latents, n_steps, n_pred_nograd, test)
            past_future_latent = concat_latent_dicts(past_latents, future_latents)

        return past_future_latent, reg_loss

    def decode(
        self,
        past_future_latent: SpatialLatentsDict,
        target: BatchedTempViews,
        test: bool,
        with_extras: bool = False,
        return_constant_pred: bool = False,
    ):
        T = past_future_latent["means"].shape[1]
        n_step, v, _, h, w = target["image"].shape[1:]

        # Decode
        depth_mode = self.train_cfg.depth_mode if not test else "depth"
        extr = target["extrinsics"]
        intr = target["intrinsics"]
        near = target["near"]
        far = target["far"]

        with self.benchmarker.time("decoder", num_calls=v * T, test=test):
            past_future_gaussians = self.decoder.prepare_gaussians(past_future_latent)

            # If we predicted longer than we need (e.g. for validation - dont render the extra steps but still return them)
            past_future_gaussians_ = past_future_gaussians
            if n_step < past_future_gaussians.means.shape[1]:
                past_future_gaussians_ = Gaussians(
                    **{k: v[:, :n_step] for k, v in past_future_gaussians.items()}
                )

            extras = None
            if with_extras:
                extras: dict[str, Tensor] = {
                    k: past_future_latent[k].float()
                    for k in ["static_float", "background_float", "state_mask"]
                    if k in past_future_latent
                }

            past_future_decoder_output = self.decoder.forward(
                past_future_gaussians_, extr, intr, near, far, (h, w), depth_mode, extras
            )

            if return_constant_pred and T - self.n_state > 0:
                past_future_gaussians_constant = Gaussians(
                    **{
                        k: torch.cat(
                            (
                                v[:, : self.n_state],
                                repeat(v[:, self.n_state], "b ... -> b t ...", t=T - self.n_state),
                            ),
                            dim=1,
                        )
                        for k, v in past_future_gaussians_.items()
                    }
                )
                past_future_decoder_output_constant = self.decoder.forward(
                    past_future_gaussians_constant, extr, intr, near, far, (h, w)
                )
                return past_future_decoder_output, past_future_gaussians, past_future_decoder_output_constant

        return past_future_decoder_output, past_future_gaussians

    # ----------------------------------------------------
    # TRAIN
    # ----------------------------------------------------
    @property
    def n_state(self):
        return self.dynamic_model.n_step_state if self.dynamic_model is not None else 1

    @property
    def n_pred(self):
        return self.dynamic_model.n_step_predict if self.dynamic_model is not None else 0

    def on_train_start(self):
        self.start_time = time.monotonic()

    def get_remaining_time(self):
        elapsed_time = time.monotonic() - self.start_time

        time_per_step = elapsed_time / (self.global_step + 1)
        remaining_time = time_per_step * (self.trainer.max_steps - self.global_step)

        ret_dict = {
            "time_per_step(s)": time_per_step,
            "elapsed_time(h)": elapsed_time / 3600,
            "remaining_time(h)": remaining_time / 3600,
        }
        return ret_dict

    def training_step(self, batch: BatchedTempExample, batch_idx):
        # torch.cuda.empty_cache()
        batch: AnyExample = self.data_shim(batch)
        assert batch["context"]["image"].shape[1] == self.n_state

        n_state = self.n_state
        n_pred = self.n_pred

        # Prepare weights for the loss: L = (1-lambda)/n_state * l_past + lambda/n_pred * l_future
        lambda_ = 0.0
        future_weighs = []
        n_grad_predict, n_pred_nograd = 0, 0
        if n_pred > 0:
            # 1. Update batch if we are training with multiple steps curriculum
            batch, n_grad_predict, n_pred_nograd = self.train_cfg.dyn_model_schedule.batch_update(
                n_state, n_pred, batch, self.global_step
            )

            # 2. Prepare weights for the future part of the loss
            lambda_ = 0.5  # if not self.train_cfg.freeze_encoder else 1.0
            gamma = 0.87  # future weight decay
            # old
            # for j in range(int(n_grad_predict / n_pred)):
            #     future_weighs.extend([gamma**j * lambda_ / n_pred] * n_pred)
            # new 1.0
            # future_weighs = [gamma**j for j in range(n_grad_predict)]
            # sum_future_weighs = sum(future_weighs)
            # future_weighs = [lambda_ * w / sum_future_weighs for w in future_weighs]
            # new 2.0
            future_weighs = [lambda_ * gamma**j for j in range(n_grad_predict)]

        weights = batch["context"]["image"].new_tensor([(1 - lambda_) / n_state] * n_state + future_weighs)

        # Encode Predict Decode
        past_latent = self.encode(batch["context"], test=False)
        past_future_latent, reg_loss = self.predict(past_latent, n_grad_predict, n_pred_nograd, test=False)
        past_future_output, past_future_gaussians = self.decode(
            past_future_latent, batch["target"], test=False
        )

        # Calc losses
        reg_loss = reg_loss + self.get_reg_loss(past_future_latent, n_state)

        if "learned_nonstatic_sigmoid" in past_future_latent.keys():
            static_reg = 1e-3 * torch.mean(past_future_latent["learned_nonstatic_sigmoid"][:, n_state:])
            reg_loss += static_reg
            self.log("reg_loss/static_reg", static_reg)

        if "learned_background_sigmoid" in past_future_latent.keys():
            bckg_reg = 1e-5 * torch.mean(past_future_latent["learned_background_sigmoid"][:, n_state:])
            reg_loss += bckg_reg
            self.log("reg_loss/background_reg", bckg_reg)

        # 1. Compute Losses
        losses_dict = {}  # losses have shape (T,)
        for loss_fn in self.losses:
            loss = loss_fn.forward(past_future_output, batch, past_future_gaussians, self.global_step)
            losses_dict[f"train/loss_{loss_fn.name}"] = loss

        # Compute Training Loss as weighted average of all losses
        train_loss_traj = sum([loss * weights for loss in losses_dict.values()])
        train_loss = torch.sum(train_loss_traj)

        # LOGGING
        with torch.no_grad():
            num_particles_masked = int(
                past_future_gaussians.means.shape[2]
                if past_latent.get("state_mask", None) is None
                else past_latent.get("state_mask").float().sum(2).mean().item()
            )
            self.log("info/num_particles", num_particles_masked)
            self.log("info/n_grad_step", n_grad_predict)
            self.log("info/n_nograd_step", n_pred_nograd)

            # 1.b Compute PSNR
            target_state_mask = batch["target"].get("state_mask", None)
            psnr = compute_psnr(batch["target"]["image"], past_future_output.color)
            psnr = reduce(psnr, "b t ... -> t", reduction="mean")

            # 1.a Log training losses
            self.log("train/loss_encoder", torch.sum(train_loss_traj[:n_state]).item())
            self.log("train/psnr_avg_past", torch.mean(psnr[:n_state]).item())
            if n_pred > 0:
                self.log("train/loss_future", torch.sum(train_loss_traj[n_state:]).item())
                self.log("train/psnr_avg_future", torch.mean(psnr[n_state:]).item())

            psnr_mean = psnr.mean().item()
            psnr_unmasked = (
                compute_psnr(batch["target"]["image"], past_future_output.color, target_state_mask)
                .mean()
                .item()
            )
            self.log("train/psnr_avg", psnr_mean)
            self.log("train/psnr_masked", psnr_unmasked)
            [self.log(f"train/{k.split('/')[-1]}_avg", v.mean().item()) for k, v in losses_dict.items()]

            # 1.b infos (glob_step needed for checkpointing)
            self.log("info/near", batch["context"]["near"].detach().cpu().numpy().mean())
            self.log("info/far", batch["context"]["far"].detach().cpu().numpy().mean())
            self.log("info/global_step", self.global_step)  # hack for ckpt monitor
            time_info = self.get_remaining_time()
            [self.log(f"info/{k}", v) for k, v in time_info.items()]

            # Log directly to wandb (TODO)
            if self.global_step % self.trainer.val_check_batch == 0 and n_grad_predict > 0:
                [
                    self.log_traj(f"train_traj/{k}", v, n_state - 1, n_pred_nograd)
                    for k, v in losses_dict.items()
                ]
                self.log_traj("train_traj/psnr", psnr, ix_vertical=n_state - 1, n_pred_nograd=n_pred_nograd)

            # Print info
            if self.global_rank == 0 and self.global_step % self.train_cfg.print_log_every_n_steps == 0:
                B, T, N = past_future_latent["opacities"].shape[:3]
                num_static = int(
                    (
                        int(torch.sum(1.0 - past_future_latent["static_float"]).item())
                        if "static_float" in past_future_latent
                        else 0
                    )
                    / B
                    / T
                )
                num_bckg = int(
                    (
                        int(torch.sum(1.0 - past_future_latent["background_float"]).item())
                        if "background_float" in past_future_latent
                        else 0
                    )
                    / B
                    / T
                )
                self.log("info/num_static_perframe", num_static)
                self.log("info/num_bckg_perframe", num_bckg)
                self.log("info/num_all_perframe", num_particles_masked / B / T)
                num_all = past_future_latent["opacities"].numel() / B / T
                print(
                    f"\n[epoch {self.trainer.current_epoch}]\n"
                    f"train step {self.global_step}; "
                    f"[B={B} T={T} N={N}: {num_particles_masked}m/{num_static}s/{num_bckg}b/{num_all}]; "
                    f"(State->Pred: {n_state}->{n_pred_nograd}+{n_grad_predict}); "
                    f"scene = {[x for x in batch['scene']]}; "
                    f"context = {batch['context']['index'][0].tolist()[:2]} ...; "  # log only first batch
                    f"target = {batch['target']['index'][0].tolist()[:2]} ...; "
                    f"bound = [{batch['context']['near'].detach().cpu().numpy().mean():.2f} "
                    f"{batch['context']['far'].detach().cpu().numpy().mean():.2f}]; "
                    f"weights = {weights.tolist()}; "
                    f"learn_state_mask: {self.encoder.encoder_info.learn_background_mask}; "
                    f"learn_static_mask: {self.encoder.encoder_info.learn_static_mask}; "
                    f"with_state_mask: {'state_mask' in batch['context']}; "
                    f"with_static_mask: {'static_float' in batch['context']}; "
                    f"reg_loss = {reg_loss.item():.6f}; "
                    f"loss = {train_loss.item():.6f}, psnr = {psnr_unmasked:.2f}/{psnr_mean:.2f}; "
                    f"time = {time_info['time_per_step(s)']:.2f}s/step, remaining = {time_info['remaining_time(h)']:.2f}h"
                )

            # Tell the data loader processes about the current step.
            if self.step_tracker is not None:
                self.step_tracker.set_step(self.global_step)

            if self.detect_anomaly:
                assert torch.isfinite(train_loss).all(), "NaNs in Train Loss"

        return train_loss + reg_loss

    # ----------------------------------------------------
    # VALIDATION
    # ----------------------------------------------------
    @rank_zero_only
    @torch.no_grad()
    def validation_step(self, batch: BatchedTempExample, batch_idx):
        batch: AnyExample = self.data_shim(batch)
        assert batch["context"]["image"].shape[0] == 1
        assert batch["context"]["image"].shape[1] == self.n_state
        scene = batch["scene"][0]

        if self.global_rank == 0:
            print(
                f"\n[validation step {self.global_step}; "
                f"scene = {scene}; "
                f"context = {batch['context']['index'].tolist()}\n"
                f" running encode-predict-decode {self.training}"
            )

        n_state = self.n_state
        n_pred = batch["target"]["image"].shape[1] - n_state

        # Encode Predict Decode
        n_pred_val = int(n_pred * 2)
        past_latent = self.encode(batch["context"], test=False)
        past_future_latent, reg_loss = self.predict(past_latent, n_pred_val, test=False)
        past_future_output, past_future_gaussians = self.decode(
            past_future_latent, batch["target"], test=False
        )

        # A. Compute Validation Metrics.
        print(" computing validation metrics")
        target_state_mask = batch["target"].get("state_mask", None)
        rgb_softmax = past_future_output.color
        rgb_gt = batch["target"]["image"]
        metrics = {
            # B, T, V
            "psnr_val": compute_psnr(rgb_gt, rgb_softmax, target_state_mask),
            "lpips_val": compute_lpips(rgb_gt, rgb_softmax),
            "ssim_val": compute_ssim(rgb_gt, rgb_softmax),
        }

        print(" logging validation metrics and others")
        for metric_name, metric in metrics.items():
            if n_pred > 0:
                avg_metric_traj = reduce(metric, "b t ... -> t", reduction="mean")
                self.log_traj(f"val_traj/{metric_name}", avg_metric_traj, ix_vertical=n_state - 1)
                self.log(f"val/encoder_{metric_name}_avg", avg_metric_traj[:n_state].mean().item())
                self.log(f"val/future_{metric_name}_avg", avg_metric_traj[n_state:].mean().item())

            self.log(f"val/{metric_name}_avg", metric.mean().item())

        # B. RECONSTRUCTION VISUALIZATIONS
        # unbatch: (B, T, View.. -> T, View..
        ctx_rgb = batch["context"]["image"][0]
        gt_rgb = batch["target"]["image"][0]
        pred_rgb = past_future_output.color[0]

        kkeys = {
            "state_mask": "Context Mask",
            "static_float": "Context DynMask",
        }
        extra_context = {}
        for kk, vv in kkeys.items():
            if kk in batch["context"]:
                vvv = batch["context"][kk][0][:n_state].float()
                extra_context[vv] = torch.cat([vvv, torch.zeros_like(vvv), 1 - vvv], dim=2)

        # if "state_mask" in batch["context"]:
        #     ctx_state_mask = batch["context"]["state_mask"][0][:n_state].float()
        #     print("MASK shape", ctx_state_mask.shape)
        #     ctx_state_mask = torch.cat(
        #         [ctx_state_mask, torch.zeros_like(ctx_state_mask), 1 - ctx_state_mask], dim=2
        #     )
        # else:
        #     ctx_state_mask = None

        # B.1 PAST: log Encoder - Reconstructions (only the first n_step_state)
        self.encoder_reconstruction(
            "val_video/Encoder - Reconstruction",
            scene,
            ctx_rgb[:n_state],
            gt_rgb[:n_state],
            pred_rgb[:n_state],
            extra_context,
        )

        # B.2 FUTURE: log Dynamic System - Prediction (future -> the last n_step_predict steps)
        self.dynamic_system_reconstruction(
            "val_video/Dynamic System - Prediction",
            scene,
            gt_rgb[n_state:],
            pred_rgb[n_state:],
        )

        # C. Render projections and construct projection image.
        self.visualize_projections("val_video/Projection", past_future_gaussians)

        # ------------------- FROM NOW ON --------------------
        # the batch dim is now the time dim (T+Tnew)
        squeezed_batch = to_batched_example(batch)
        squeezed_gaussians = past_future_gaussians.to_batched_gaussians()
        squeezed_extras: dict[str, Tensor] = {
            k: pack([past_future_latent[k]], "* gaussian one")[0]
            for k in ["static_float", "background_float", "state_mask"]
            if k in past_future_latent
        }
        if "state_mask" in squeezed_extras:
            squeezed_extras["state_mask"] = squeezed_extras["state_mask"].float()
        # -----------------------------------------------------

        # C.2 Log PCD
        # self.log_pcd("val_info/Encoder-PCD-Reconstruction (t=0)", squeezed_gaussians)

        # C.3 Log statistics
        self.log_statistics(
            past_future_gaussians.opacities, scales=past_future_latent.get("scales", None), supkey="val_info"
        )

        # D. Draw cameras.
        self.visualize_cameras("val_info/Cameras", squeezed_batch)

        # E. Visualize encoder (short-circuited for now)
        self.visualize_encoder("val_video/Encoder-Vis", squeezed_batch)

        # F. Run video validation step. (interpolates over the whole gaussian sequence between two cameras)
        if n_pred > 0:
            self.render_video_interpolation(
                "val_video/static", squeezed_gaussians, squeezed_batch, ix1=0, ix2=0, extras=squeezed_extras
            )
        self.render_video_interpolation(
            "val_video/interpolation", squeezed_gaussians, squeezed_batch, extras=squeezed_extras
        )
        self.render_video_wobble(
            "val_video/wobble", squeezed_gaussians, squeezed_batch, extras=squeezed_extras
        )
        if self.train_cfg.extended_visualization:
            self.render_video_interpolation_exaggerated(
                "val_video/exagerrated", squeezed_gaussians, squeezed_batch
            )

    # ----------------------------------------------------
    # TEST
    # ----------------------------------------------------
    def test_step(self, batch: BatchedTempExample, batch_idx):
        # torch.cuda.empty_cache()
        batch: AnyExample = self.data_shim(batch)
        assert batch["context"]["image"].shape[1] == self.n_state
        assert batch["context"]["image"].shape[0] == 1

        scene = batch["scene"][0]
        n_state = self.n_state
        n_pred = batch["target"]["image"].shape[1] - n_state

        # Encode Predict Decode
        past_latent = self.encode(batch["context"], test=True)
        past_future_latent, reg_loss = self.predict(past_latent, n_pred, test=True)
        past_future_output, past_future_gaussians, past_future_output_constant = self.decode(
            past_future_latent, batch["target"], test=True, with_extras=True, return_constant_pred=True
        )

        # GT and Prediction
        target_indices = batch["target"]["index"][0][0].tolist()  # test: same context for all timesteps
        context_indices = batch["context"]["index"][0][0].tolist()
        context_str = f"[{'_'.join([str(x) for x in context_indices])}]"
        target_str = f"[{'_'.join([str(x) for x in target_indices])}]"

        rgb_pred = past_future_output.color[0]
        depth_pred = past_future_output.depth[0]
        if past_future_output.extras:
            other_imgs, keys = zip(*[(v[0], k) for k, v in past_future_output.extras.items()])
        else:
            other_imgs, keys = [], []
        # print(f"shapes: {rgb_pred.shape}, {depth_pred.shape}, {other_imgs[0].shape} ")
        # exit()
        rgb_gt = batch["target"]["image"][0]
        rgb_context = batch["context"]["image"][0]
        T, V = rgb_pred.shape[:2]

        # Save images.
        if self.test_cfg.save_image and n_pred == 0:
            img = add_border(
                vcat(
                    add_label(hcat(*rgb_context[0]), f"Context: {context_str}"),
                    add_label(hcat(*rgb_gt[0]), f"Target: {target_str}"),
                    add_label(hcat(*rgb_pred[0]), "Predicted"),
                    add_label(hcat(*depth_map(depth_pred[0])), "Predicted Depth"),
                )
            )
            self.log_image(
                f"test/{str(self.test_cfg.output_path).split('/')[-1]}_image",
                img,
                caption=f"{scene.split(':')[0]}_{context_str}_{target_str}",
            )

        # save video
        if self.test_cfg.save_video and n_pred > 0:
            comparison = [
                add_border(
                    vcat(
                        add_label(hcat(*rgb_context[i % self.n_state]), f"Context: {context_str}"),
                        add_label(hcat(*img_gt), f"Target: {target_str}"),
                        add_label(hcat(*img_pred), "Predicted"),
                        add_label(hcat(*depth_map(depth_pred)), "Predicted Depth"),
                        *[add_label(hcat(*im), k) for im, k in zip(ims, keys)],
                    ),
                )
                for i, (img_gt, img_pred, depth_pred, *ims) in enumerate(
                    zip(rgb_gt, rgb_pred, depth_pred, *other_imgs)
                )
            ]
            if len(comparison) > 1:
                self.log_video(
                    f"test_videos/{str(self.test_cfg.output_path).split('/')[-1]}",
                    comparison,
                    loop_reverse=False,
                    caption=f"{scene}  N_state={n_state}, N_pred={self.n_pred}, N_future={len(rgb_pred)-n_state}",
                    fps=12,
                )

        # compute scores
        if self.test_cfg.compute_scores:
            if batch_idx < self.test_cfg.eval_time_skip_steps:
                self.time_skip_steps_dict["encoder"] += 1
                self.time_skip_steps_dict["decoder"] += T * V
                self.time_skip_steps_dict["dyn_sys"] += n_pred

            # Compute batched metrics
            target_state_mask = batch["target"].get("state_mask", None)
            rgb_gt = batch["target"]["image"]
            rgb_context = batch["context"]["image"]
            rgb_pred = past_future_output.color
            rgb_pred_constant = past_future_output_constant.color
            metrics = {
                # T, V
                f"psnr": compute_psnr(rgb_gt, rgb_pred, target_state_mask),
                f"psnr_constant": compute_psnr(rgb_gt, rgb_pred_constant, target_state_mask),
                f"lpips": compute_lpips(rgb_gt, rgb_pred),
                f"lpips_constant": compute_lpips(rgb_gt, rgb_pred_constant),
                f"ssim": compute_ssim(rgb_gt, rgb_pred),
                f"ssim_constant": compute_ssim(rgb_gt, rgb_pred_constant),
            }
            if target_state_mask is not None:
                metrics[f"psnr_unmasked"] = compute_psnr(rgb_gt, rgb_pred)
                metrics[f"psnr_unmasked_constant"] = compute_psnr(rgb_gt, rgb_pred_constant)

            # log for each test case, so that we can hand pick best trajs
            name = f"{str(self.test_cfg.output_path).split('/')[-1]}"
            my_dict = {}
            for k, v in metrics.items():
                if "constant" in k:
                    continue
                v = reduce(v, "b t ... -> t", reduction="mean")
                my_dict.update(
                    {
                        f"test_scores/{k}_past_{name}": v[:n_state].mean().item(),
                        f"test_scores/{k}_future_{name}": v[n_state:].mean().item(),
                        f"test_scores/{k}_{name}": v.mean().item(),
                    }
                )
            self.logger.log_metrics(my_dict, step=self.global_step)

            for metric_name, metric in metrics.items():
                metric = reduce(metric, "b t ... -> t", reduction="mean")

                if metric_name not in self.test_step_outputs:
                    self.test_step_outputs[metric_name] = []
                self.test_step_outputs[metric_name].append(metric.mean().item())

                for i in range(n_pred + n_state):
                    _t = "past" if i < n_state else "future"
                    _i = i if i < n_state else i - n_state

                    test_metric_name = f"{metric_name}_{_t}_{_i}"
                    if test_metric_name not in self.test_step_outputs:
                        self.test_step_outputs[test_metric_name] = []

                    self.test_step_outputs[test_metric_name].append(metric[i].mean().item())

    def on_test_end(self) -> None:
        out_dir = self.test_cfg.output_path / "scores"
        saved_scores = {}
        if self.test_cfg.compute_scores:
            self.benchmarker.dump_memory(out_dir / "peak_memory.json")
            self.benchmarker.dump(out_dir / "benchmark.json")

            for metric_name, metric_scores in self.test_step_outputs.items():
                # avg_scores = sum(metric_scores) / len(metric_scores)
                avg_scores = np.mean(metric_scores)
                std_scores = np.std(metric_scores, ddof=1)
                saved_scores[metric_name] = (avg_scores, std_scores)
                print(metric_name, avg_scores, std_scores)
                with (out_dir / f"scores_{metric_name}_all.json").open("w") as f:
                    json.dump(metric_scores, f)
                metric_scores.clear()
            self.log_test_scores(f"{str(self.test_cfg.output_path).split('/')[-1]}", saved_scores)

            for tag, times in self.benchmarker.execution_times.items():
                times = times[int(self.time_skip_steps_dict[tag]) :]
                saved_scores[tag] = [len(times), np.mean(times)]
                print(f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call")
                self.time_skip_steps_dict[tag] = 0

            with (out_dir / "scores_all_avg.json").open("w") as f:
                json.dump(saved_scores, f)

            self.benchmarker.clear_history()
        else:
            self.benchmarker.dump(self.test_cfg.output_path / "benchmark.json")
            self.benchmarker.dump_memory(self.test_cfg.output_path / "peak_memory.json")
            self.benchmarker.summarize()

    # ----------------------------------------------------
    # VISUALIZATION UTILITIES used in validation_step
    # ----------------------------------------------------

    @rank_zero_only
    @torch.no_grad()
    def log_statistics(
        self,
        opacities: Tensor,
        scales: Tensor | None = None,
        supkey: str = "",
    ):
        data_dict = dict(
            opacities=opacities.detach().cpu().numpy().reshape(-1),
        )
        if scales is not None:
            data_dict["scales"] = scales.norm(dim=-1).detach().cpu().numpy().reshape(-1)

        kkey = "info" if supkey == "" else supkey
        metrics = {f"{kkey}/{k}_statistics": wandb.Histogram(v) for k, v in data_dict.items()}
        self.logger.log_metrics(metrics, step=self.global_step)

    @rank_zero_only
    @torch.no_grad()
    def log_traj(self, key: str, traj: Tensor, ix_vertical: int = None, n_pred_nograd=0):
        if self.n_pred == 0:
            return

        plt.figure()
        # t = traj.cpu().numpy()
        t = traj.detach().cpu().numpy()

        x = np.arange(t.shape[0])

        if n_pred_nograd == 0:
            plt.scatter(x[: x.shape[0]], t, color="b", alpha=0.8)
        else:
            n_state = self.n_state
            plt.scatter(x[:n_state], t[:n_state], color="b", alpha=0.8)
            plt.scatter(
                np.arange(n_state, n_state + n_pred_nograd),
                [0] * n_pred_nograd,
                color="orange",
                alpha=0.8,
            )
            plt.scatter(
                n_pred_nograd + x[n_state:],
                t[n_state:],
                color="g",
                alpha=0.8,
            )

        if ix_vertical is not None:
            plt.axvline(ix_vertical, color="r", linestyle="--")
        plt.xlabel("Time")
        plt.ylabel(key)
        self.log_figure(key, plt.gcf())

    def log_test_scores(self, name: str, scores: dict[str, tuple[float, float]]):
        # keys = [k for k in ["psnr", "psnr_unmasked", "lpips", "ssim"] if k in scores]
        keys = [k for k in scores.keys() if "past" not in k and "future" not in k]
        avg_vals = {key: scores.pop(key) for key in keys if key in scores}
        self.logger.log_metrics(
            {f"test_{k}/full_avg_{name}": v[0] for k, v in avg_vals.items() if "constant" not in k},
            step=self.global_step,
        )
        self.logger.log_metrics(
            {f"test_{k}/full_std_{name}": v[1] for k, v in avg_vals.items() if "constant" not in k},
            step=self.global_step,
        )
        # if self.n_pred == 0:
        if "psnr_future_0" not in scores:
            return

        avgs_past = {key: [] for key in keys}
        stds_past = {key: [] for key in keys}

        avgs_future = {key: [] for key in keys}
        stds_future = {key: [] for key in keys}

        for key in keys:
            i = 0
            # past
            while f"{key}_past_{i}" in scores:
                v_avg, v_std = scores[f"{key}_past_{i}"]
                avgs_past[key].append(v_avg)
                stds_past[key].append(v_std)
                i += 1

            # future
            j = 0
            while f"{key}_future_{j}" in scores:
                v_avg, v_std = scores[f"{key}_future_{j}"]
                avgs_future[key].append(v_avg)
                stds_future[key].append(v_std)
                j += 1

        # remove constant keys only used for supportive curves in the plots
        keys = [k for k in keys if "constant" not in k]

        self.logger.log_metrics(
            {f"test_{k}/past_avg_{name}": np.mean(avgs_past[k]) for k in keys}, step=self.global_step
        )
        self.logger.log_metrics(
            {f"test_{k}/past_std_{name}": np.mean(stds_past[k]) for k in keys}, step=self.global_step
        )
        self.logger.log_metrics(
            {f"test_{k}/future_avg_{name}": np.mean(avgs_future[k]) for k in keys if k in avgs_future},
            step=self.global_step,
        )
        self.logger.log_metrics(
            {f"test_{k}/future_std_{name}": np.mean(stds_future[k]) for k in keys if k in stds_future},
            step=self.global_step,
        )
        for i, key in enumerate(keys):
            # Create a new figure for each key
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plot the data
            avg_traj = np.array(avgs_past[key] + avgs_future[key])
            std_traj = np.array(stds_past[key] + stds_future[key])
            n_steps = len(avg_traj)
            ax.plot(avg_traj, label=key)

            # Plot standard deviation area
            ax.fill_between(
                np.arange(n_steps), avg_traj - std_traj, avg_traj + std_traj, alpha=0.3, label="±1 STD"
            )

            # Add horizontal and vertical lines
            # ax.axhline(avg_vals[key], color="r", linestyle="--", label=f"{key} avg")
            key_constant = f"{key}_constant"
            avg_traj = np.array(avgs_past[key_constant] + avgs_future[key_constant])
            std_traj = np.array(stds_past[key_constant] + stds_future[key_constant])
            n_steps = len(avg_traj)
            ax.plot(avg_traj, label=key_constant, color="r", linestyle="--")

            # Plot standard deviation area
            ax.fill_between(
                np.arange(n_steps),
                avg_traj - std_traj,
                avg_traj + std_traj,
                alpha=0.1,
                label=f"±1 STD (constant)",
            )

            # add vertical line for future start
            ax.axvline(len(avgs_past[key]), color="g", linestyle="--", label="future start")

            # Add legend and title
            ax.legend()
            ax.set_title(f"{key} (avg = {avg_vals[key][0]:.2f})")

            # Add a super title for each individual plot
            full_avg = avg_vals[key][0]
            full_std = avg_vals[key][1]
            past_avg = np.mean(avgs_past[key])
            past_std = np.mean(stds_past[key])
            future_avg = np.mean(avgs_future[key])
            future_std = np.mean(stds_future[key])
            title = f"{key} (avg = {full_avg:.2f} ± {full_std:.2f})\n"
            title += f"Past: {past_avg:.2f} ± {past_std:.2f}, Future: {future_avg:.2f} ± {future_std:.2f}"
            fig.suptitle(title)

            # Log the individual figure
            self.log_figure(f"test_plots/{key}_{name}", fig)

    def log_figure(self, name: str, fig):
        with SpooledTemporaryFile(max_size=1024**2) as fp:  # 1 MB
            fig.savefig(fp, format="png")
            plt.close(fig)
            fp.seek(0)
            img = imread(fp, format="png")
        self.logger.log_image(name, [img], step=self.global_step)

    @rank_zero_only
    def encoder_reconstruction(
        self, key, scene: str, temp_rgb_context, temp_rgb_gt, temp_rgb_softmax, ctx_state_mask=None
    ):
        ctx_state_mask = {} if ctx_state_mask is None else ctx_state_mask
        comparison = [
            add_border(
                hcat(
                    # *(
                    #     [add_label(vcat(*img_ctx), "Context")]+ [add_label(vcat(*v[i]), k)for k, v in ctx_state_mask]
                    # ),
                    *[add_label(vcat(*img_ctx), "Context")],
                    *[add_label(vcat(*v[i]), k) for k, v in ctx_state_mask.items()],
                    add_label(vcat(*img_gt), "GT-Target"),
                    add_label(vcat(*img_softmax), "Reconst-Target"),
                )
            )
            for i, (img_ctx, img_gt, img_softmax) in enumerate(
                zip(temp_rgb_context, temp_rgb_gt, temp_rgb_softmax)
            )
        ]
        if len(comparison) > 1:
            self.log_video(key, comparison, loop_reverse=False, caption=scene, fps=4)

        else:
            self.logger.log_image(
                key,
                [prep_image(comparison[0])],
                step=self.global_step,
                caption=[scene],
            )

        # B.0 AS HORIZONTALLY-CONCATENATED IMAGE (log Encoder-Reconstruction)
        # without border using einops
        # img_ctx = rearrange(temp_rgb_context, "t v c h w -> v c h (t w)")
        # img_gt = rearrange(temp_rgb_gt[-n_pred:], "t v c h w -> v c h (t w)")
        # img_softmax = rearrange(temp_rgb_softmax[-n_pred:], "t v c h w -> v c h (t w)")
        # comparison = add_border(
        #     hcat(
        #         add_label(vcat(*img_ctx), "Context"),
        #         add_label(vcat(*img_gt), "Target (Ground Truth)"),
        #         add_label(vcat(*img_softmax), "Reconst-Target (Softmax)"),
        #     )
        # )

    @rank_zero_only
    def dynamic_system_reconstruction(self, key, scene: str, temp_rgb_gt, temp_rgb_softmax):
        n_pred = self.n_pred
        n_state = self.n_state

        if n_pred > 0:
            comparison = [
                add_label(
                    add_border(
                        hcat(
                            add_label(vcat(*img_gt), "GT-Future"),
                            add_label(vcat(*img_softmax), "Predicted-Future"),
                        ),
                    ),
                    f"N_state={n_state}, N_pred={self.n_pred}, N_future={len(temp_rgb_gt)-n_state}",
                )
                for img_gt, img_softmax in zip(temp_rgb_gt, temp_rgb_softmax)
            ]
            if len(comparison) > 1:
                self.log_video(key, comparison, loop_reverse=False, caption=scene, fps=4)

            else:
                self.logger.log_image(
                    key,
                    [prep_image(comparison[0])],
                    step=self.global_step,
                    caption=[scene],
                )

    @rank_zero_only
    def visualize_projections(self, key, past_future_gaussians):
        temp_projections = render_projections(
            past_future_gaussians[0],  # unbatch (batch=1)
            256,
            extra_label="",
        )

        projections = [add_border(hcat(*img)) for img in temp_projections]
        if len(projections) > 1:
            self.log_video(key, projections, loop_reverse=False, fps=4)

        else:
            self.logger.log_image(key, [prep_image(projections[0])], step=self.global_step)

    @rank_zero_only
    def visualize_cameras(self, key, squeezed_batch):
        cameras = hcat(*render_cameras(squeezed_batch, 256))  # renders only the first timestep camera
        self.logger.log_image(key, [prep_image(add_border(cameras))], step=self.global_step)

    @rank_zero_only
    def visualize_encoder(self, key, squeezed_batch):
        return
        if self.encoder_visualizer is not None:
            for k, image in self.encoder_visualizer.visualize(
                squeezed_batch["context"], self.global_step
            ).items():
                self.logger.log_image(f"{key}/k", [prep_image(image)], step=self.global_step)

    @rank_zero_only
    def render_video_wobble(self, key, gaussians: Gaussians, batch: BatchedExample, extras={}):
        # Only first Ttwo views are needed to get the wobble radius.
        # _, v, _, _ = batch["context"]["extrinsics"].shape
        # if v != 2:
        #     return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            extrinsics = generate_wobble(
                batch["context"]["extrinsics"][:, 0],
                delta * 0.25,
                t,
            )
            intrinsics = repeat(
                batch["context"]["intrinsics"][:, 0],
                "b i j -> b v i j",
                v=t.shape[0],
            )
            return extrinsics, intrinsics

        return self.render_video_generic(gaussians, batch, trajectory_fn, key, num_frames=60, extras=extras)

    @rank_zero_only
    def render_video_interpolation(
        self, key, gaussians: Gaussians, batch: BatchedExample, ix1=0, ix2=1, extras={}
    ):
        _, v, _, _ = batch["context"]["extrinsics"].shape

        def trajectory_fn(t):
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, ix1],
                (batch["context"]["extrinsics"][0, ix2] if v > 1 else batch["target"]["extrinsics"][0, 0]),
                t,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, ix1],
                (batch["context"]["intrinsics"][0, ix2] if v > 1 else batch["target"]["intrinsics"][0, 0]),
                t,
            )
            return extrinsics[None], intrinsics[None]

        return self.render_video_generic(gaussians, batch, trajectory_fn, key, extras=extras)

    @rank_zero_only
    def render_video_interpolation_exaggerated(self, key, gaussians: Gaussians, batch: BatchedExample):
        # Two views are needed to get the wobble radius.
        _, v, _, _ = batch["context"]["extrinsics"].shape
        if v != 2:
            return

        def trajectory_fn(t):
            origin_a = batch["context"]["extrinsics"][:, 0, :3, 3]
            origin_b = batch["context"]["extrinsics"][:, 1, :3, 3]
            delta = (origin_a - origin_b).norm(dim=-1)
            tf = generate_wobble_transformation(
                delta * 0.5,
                t,
                5,
                scale_radius_with_t=False,
            )
            extrinsics = interpolate_extrinsics(
                batch["context"]["extrinsics"][0, 0],
                (batch["context"]["extrinsics"][0, 1] if v == 2 else batch["target"]["extrinsics"][0, 0]),
                t * 5 - 2,
            )
            intrinsics = interpolate_intrinsics(
                batch["context"]["intrinsics"][0, 0],
                (batch["context"]["intrinsics"][0, 1] if v == 2 else batch["target"]["intrinsics"][0, 0]),
                t * 5 - 2,
            )
            return extrinsics @ tf, intrinsics[None]

        return self.render_video_generic(
            gaussians,
            batch,
            trajectory_fn,
            key,
            num_frames=300,
            smooth=False,
            loop_reverse=False,
        )

    @rank_zero_only
    def render_video_generic(
        self,
        gaussians: Gaussians,  # first dim is time
        batch: BatchedExample,
        trajectory_fn: TrajectoryFn,
        name: str,
        num_frames: int = 30,
        smooth: bool = True,
        loop_reverse: bool = False,
        extras={},
    ) -> None:
        t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=self.device)
        if smooth:
            t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

        extrinsics, intrinsics = trajectory_fn(t)
        extrinsics = rearrange(extrinsics[:1], "b v i j -> v b i j")
        intrinsics = rearrange(intrinsics[:1], "b v i j -> v b i j")

        # Interpolate the Gaussians (assume first dim is time)
        t_g = torch.linspace(0, 1, gaussians.means.shape[0], dtype=torch.float32, device=self.device)
        corresponding_ix = torch.abs(t.unsqueeze(1) - t_g).argmin(dim=1)  # gaussian index for each time in t
        gaussians = gaussians[corresponding_ix]
        extras_traj = {k: v[corresponding_ix] for k, v in extras.items()}

        _, _, _, h, w = batch["context"]["image"].shape

        # TODO: Interpolate near and far planes?
        near = repeat(batch["context"]["near"][0, 0], " -> b 1", b=num_frames)
        far = repeat(batch["context"]["far"][0, 0], " -> b 1", b=num_frames)
        output_prob = self.decoder.forward(
            gaussians, extrinsics, intrinsics, near, far, (h, w), "depth", extras=extras_traj
        )
        rgbs, depths = output_prob.color[:, 0], depth_map(output_prob.depth[:, 0])
        other_imgs, keys = zip(*[(v[:, 0], k) for k, v in output_prob.extras.items()]) if extras else ([], [])
        n_state, n_pred, n_fut = self.n_state, self.n_pred, len(t_g) - self.n_state
        desc = f"[Dyn-Model: {n_state}->{n_pred}, N={n_fut}]" if n_pred > 0 else ""
        images_prob = [
            vcat(
                add_label(rgb, f"RGB {desc}"),
                add_label(depth, "Depth"),
                *[add_label(im, k) for im, k in zip(ims, keys)],
            )
            for rgb, depth, *ims in zip(rgbs, depths, *other_imgs)
        ]

        images = [add_border(image_prob) for image_prob, _ in zip(images_prob, images_prob)]
        self.log_video(name, images, loop_reverse)

    def log_video(
        self,
        name: str,
        images: list[Tensor],
        loop_reverse: bool = False,
        caption=None,
        fps=30,
        save_path: Path = None,
    ) -> None:
        video = torch.stack(images)
        video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
        if loop_reverse:
            video = pack([video, video[::-1][1:-1]], "* c h w")[0]

        name = name if "/" in name else f"video/{name}"
        visualizations = {
            name: wandb.Video(video[None], fps=fps, format="mp4", caption=caption),
        }
        if wandb.run is not None:
            self.logger.log_metrics(visualizations, step=self.global_step)
        else:
            for key, value in visualizations.items():
                tensor = value._prepare_video(value.data)
                clip = mpy.ImageSequenceClip(list(tensor), fps=fps)
                if save_path is None:
                    caption = f"_{caption}" if caption is not None else ""
                    save_path = get_log_path()

                save_path = save_path / key
                save_path = save_path / f"{name.split('/')[-1]}{caption}_{self.global_step:0>6}.mp4"
                save_path.parent.mkdir(exist_ok=True, parents=True)
                clip.write_videofile(str(save_path), logger=None)

    def log_image(
        self,
        key: str,
        image: Tensor,
        caption: str = None,
        save_path: Path = None,
    ):
        if wandb.run is not None:
            self.logger.log_image(key, [image], step=self.global_step, caption=[caption])
        else:
            caption = f"_{caption}" if caption is not None else ""
            save_image(
                image,
                (
                    get_log_path() / f"{key}{caption}_{self.global_step:0>6}.png"
                    if save_path is None
                    else save_path
                ),
            )

    def log_pcd(
        self,
        key: str,
        gaussians: Gaussians,
    ):
        pcd_means = gaussians.means[: self.n_state].view(-1, 3).cpu().numpy()
        pcd_colors = gaussians.harmonics[: self.n_state, ..., 0].view(-1, 3).cpu().numpy() * 255.0
        pcd = np.concatenate([pcd_means, pcd_colors], axis=-1)
        if gaussians.state_mask is not None:
            pcd = pcd[gaussians.state_mask[: self.n_state].cpu().numpy().reshape(-1)]

        dtype = [("x", "f4"), ("y", "f4"), ("z", "f4"), ("red", "u1"), ("green", "u1"), ("blue", "u1")]
        vertex = np.array([tuple([*p[:3], *p[3:].astype(np.uint8)]) for p in pcd], dtype=dtype)
        if wandb.run is not None:
            self.logger.log_metrics(
                {key: wandb.Object3D({"type": "lidar/beta", "points": vertex})}, step=self.global_step
            )
        else:
            # Fallback to saving as PLY file
            el = PlyElement.describe(vertex, "vertex")
            PlyData([el]).write(f"{get_log_path()}/point_cloud.ply")

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.optimizer_cfg.lr, weight_decay=1e-3)
        if self.optimizer_cfg.cosine_annealing_warmup:
            from cosine_annealing_warmup import CosineAnnealingWarmupRestarts

            num_cycles = 3
            gamma = 0.5
            min_lr = min(
                self.optimizer_cfg.lr / self.optimizer_cfg.lr * gamma**num_cycles,
                self.optimizer_cfg.lr / 10.0,
            )
            warm_up = CosineAnnealingWarmupRestarts(
                optimizer,
                first_cycle_steps=self.trainer.max_steps // num_cycles,
                cycle_mult=1.0,
                max_lr=self.optimizer_cfg.lr,
                min_lr=min_lr,
                warmup_steps=1000,
                gamma=gamma,
            )
        elif self.optimizer_cfg.cosine_lr:
            warm_up = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                self.optimizer_cfg.lr,
                self.trainer.max_steps + 10,
                pct_start=0.01,
                cycle_momentum=False,
                anneal_strategy="cos",
            )

        else:
            warm_up_steps = self.optimizer_cfg.warm_up_steps
            warm_up = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                1 / warm_up_steps,
                1,
                total_iters=warm_up_steps,
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": warm_up,
                "interval": "step",
                "frequency": 1,
            },
        }


# Color-map the result.
def depth_map(result: Tensor):
    try:
        near = result[result > 0][:16_000_000].quantile(0.01).log()
    except Exception as e:
        print(f"Exception when computing near for depth: {e}")
        near = result.min().clip(0.0).log()
    try:
        far = result.view(-1)[:16_000_000].quantile(0.99).log()
    except Exception as e:
        print(f"Exception when computing far for depth: {e}")
        far = result.max().log()

    result = result.log()
    result = 1 - (result - near) / (far - near)
    return apply_color_map_to_image(result, "turbo")


def concat_latent_dicts(*past_latent_dict: SpatialLatentsDict) -> SpatialLatentsDict:
    return {
        k: torch.cat([v[k] for v in past_latent_dict if k in v], dim=1) for k in past_latent_dict[0].keys()
    }
