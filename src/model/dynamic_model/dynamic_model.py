import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Optional, TypeVar

import torch
from einops import pack, rearrange, repeat, unpack
from jaxtyping import Bool, Float
from torch import Tensor
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from ...dataset import DatasetCfg
from ..state_adapter import CX, DX, ModalityT, StateInfo, X, Z


@dataclass(kw_only=True)
class DynamicModelCfg:
    no_xyz: Optional[bool] = False


T = TypeVar("T", bound=DynamicModelCfg)


class DynamicModel(torch.nn.Module, ABC, Generic[T]):
    cfg: T
    dataset_cfg: DatasetCfg
    state_info: StateInfo

    n_step_state: int
    n_step_state_predict: int
    n_step_predict_multiple: int

    keys: list[str]
    static_keys: list[str]
    pred_modality: dict[str, ModalityT]
    prediction_ps: dict[str, tuple[int]]

    def __init__(
        self,
        cfg: T,
        dataset_cfg: DatasetCfg,
        state_info: StateInfo,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.dataset_cfg = dataset_cfg
        self.state_info = state_info

        self.no_xyz = cfg.no_xyz

        # Input output shapes
        # n_step_state: 1  # nr of steps for the state
        # n_step_predict: 0  # nr of steps for the prediction (n_step_predict_multiple * n_step_state_predict)
        # n_step_state_predict: 1  # nr of steps from the step that predict forward (starting from the back)
        # n_step_predict_multiple:  1 # multiples that each of n_step_state_predict predicts forward in time
        self.n_step_state = dataset_cfg.n_step_state
        self.n_step_state_predict = dataset_cfg.n_step_state_predict
        self.n_step_predict_multiple = dataset_cfg.n_step_predict_multiple

        # make sure key=means is the first in the list
        self.keys = sorted([k for k in state_info.dynamic_state_keys if k in self.state_info.state_shapes])
        self.static_keys = sorted([k for k in self.state_info.state_shapes if k not in self.keys])

        # prepare the shapes for the prediction and state (to be used with einops)
        pred_shapes = {
            f"{Z}_{X}": self.state_info.state_shapes,
            f"{CX}_{X}": self.state_info.state_increment_shapes,
            f"{CX}_{DX}": self.state_info.state_increment_shapes,
        }
        self.pred_modality = {k: state_info.pred_modality.get(k, f"{Z}_{X}") for k in self.keys}
        self.prediction_ps = {k: pred_shapes[self.pred_modality[k]][k] for k in self.keys}

        # print INFO
        dyn_state = {k: self.state_info.state_shapes[k] for k in self.keys}
        static_state = {k: self.state_info.state_shapes[k] for k in self.static_keys}
        print(f"\nDynamicModel:  dynamic_state: {dyn_state}, static_keys: {static_state}\n")
        assert "means" in self.keys, "Key 'means' must be present in the state shapes"

    @property
    def n_step_predict(self) -> int:
        return self.n_step_predict_multiple * self.n_step_state_predict

    def state_dim(self) -> int:
        state_dim = sum([math.prod(s) for s in self.state_info.state_shapes.values()])
        if self.no_xyz:
            state_dim -= 3
        return state_dim

    def pred_dim(self) -> int:
        return sum([math.prod(s) for s in self.prediction_ps.values()])

    def allowed_key(self, k) -> bool:
        return False if (k == "means" and self.no_xyz) else True

    # ---------------------
    # ----- Interface -----
    # ---------------------

    def get_regularization_loss(self, pred: dict[str, Tensor]) -> Tensor | float:
        return 0.0

    @abstractmethod
    def forward(
        self,
        xyz: Float[Tensor, "batch time gaussian 3"],
        feature: Float[Tensor, "batch time gaussian d_3_p_feat"],
        state_mask: Optional[Bool[Tensor, "batch time gaussian 1"]] = None,
        cond: Optional[dict[str, Tensor]] = None,
    ) -> tuple[Float[Tensor, "batch time_pred gaussian d_3_p_feat"], dict[str, Tensor]]:
        # output has shape [batch, n_step_predict, xyz.size(-1)+feature.size(-1)]
        pass

    # ---------------------
    def default_increment(
        self,
        past: Float[Tensor, "batch time gaussian *#state"],
        pred: Float[Tensor, "batch n_pred time gaussian *#d_pred"],
        static_float: Optional[Float[Tensor, "batch n_pred time gaussian 1"]],
        k: str,
    ) -> Float[Tensor, "batch t_pred gaussian_pred *#d_state"]:
        # k is one of z_x: absolute future_predictions
        #             cx_x: relative future_predictions
        #             cx_dx relative future_predictions increments
        n_pred = pred.size(1)
        if n_pred > 1:
            past = repeat(past, "b t ... -> b n_pred t ...", n_pred=n_pred)
        if k == "z_x":
            delta_future = pred - past
        elif k == "cx_x":
            delta_future = pred
        elif k == "cx_dx":
            delta_future = pred.cumsum(dim=1)
        else:
            raise ValueError(f"Unknown modality: {k}")

        if static_float is not None:
            delta_future = static_float * delta_future

        future = rearrange(past + delta_future, "b n_pred t g ... -> b (n_pred t) g ...")
        return future

    def step(
        self,
        past_dict: dict[str, Float[Tensor, "batch time gaussian *#state"]],
        static_dict: dict[
            str, Float[Tensor, "batch time gaussian *#state1"] | Bool[Tensor, "batch time gaussian 1"]
        ],
        cond: Optional[dict[str, Tensor]] = None,
    ) -> tuple[dict[str, Float[Tensor, "batch time_pred gaussian *#state"]], Tensor | float]:

        # 0. Masks
        state_mask = static_dict.pop("state_mask", None)
        static_float = static_dict.pop("static_float", None)

        # 1. Prepare xyz, and pack the features
        xyz = past_dict["means"]
        features, ps_feat = pack(
            [past_dict[k] for k in self.keys if self.allowed_key(k)]
            + [static_dict[kk] for kk in self.static_keys],
            "b t g *",
        )

        # 2. Forward
        prediction, reg_dict = self(xyz, features, state_mask=state_mask, cond=cond)
        reg_loss = self.get_regularization_loss(reg_dict)

        # 3. Increment the future_init with the prediction to get the future
        pred_list = unpack(prediction, [self.prediction_ps[k] for k in self.keys], "b n_pred t g *")

        n_pred = pred_list[0].size(1)
        if static_float is not None:
            static_float = repeat(
                static_float[:, -self.n_step_state_predict :], "b t ... -> b n_pred t ...", n_pred=n_pred
            )

        future_dict = {
            k: self.state_info.state_update_funcs.get(k, self.default_increment)(
                past=(past_dict[k][:, -self.n_step_state_predict :]),
                pred=pred_list[i],
                static_float=static_float if k not in self.state_info.immune_to_static else None,
                k=self.pred_modality[k],
            )
            for i, k in enumerate(self.keys)
        }
        return future_dict, reg_loss

    def rollout(
        self,
        spatial_latent_vec: dict[str, torch.Tensor],
        n_steps: int = -1,
        n_considered: int = -1,
        cond: Optional[dict[str, Tensor]] = None,
        with_past: bool = False,
        verbose: bool = False,
    ):
        """Rollout dynamics model for n_steps.

        Args:
            model: dynamics model
            past: (B, n_past, *elt_size) past states
            cond: None or (B, *cond_size) extra conditioning
            n_steps: number of steps to rollout
            stride: stride for rolling out
            verbose: use tqdm for progress bar

        Returns:
            (B, n_steps, *elt_size) rolled out states
        """
        # EDGE-CASES: deal with n_steps=0 or to short of a past
        reg_loss = spatial_latent_vec["means"].new_tensor(0)
        if n_steps == 0:
            return (
                {
                    k: repeat(v, "b t ... -> b (t z)...", z=0) if v is not None else None
                    for k, v in spatial_latent_vec.items()
                },
                reg_loss,
            )
        assert self.n_step_state <= spatial_latent_vec["means"].size(
            1
        ), f"provided past with T={spatial_latent_vec['means'].size(1)} does not match the model's n_step_state={self.n_step_state}"

        # START
        n_state = self.n_step_state
        B, n_past = spatial_latent_vec["means"].shape[:2]
        n_considered = n_considered if n_considered > 0 else self.n_step_predict

        # pre-simulation to get indexes for static part
        sim_indexes = list(range(n_past))
        for i in range(n_state, n_state + n_steps, n_considered):
            past_ix = sim_indexes[i - self.n_step_state_predict : i] * self.n_step_predict_multiple
            e = min(i + n_considered, n_state + n_steps)
            sim_indexes.extend(past_ix[: e - i])

        # Non changing features (leave mask unchanged for the output, multiply it with workspace mask for rollout)
        static_keys = [
            k
            for k in spatial_latent_vec.keys()
            if k in self.static_keys + ["state_mask", "static_float", "background_float"]
        ]

        static_dict = {k: spatial_latent_vec[k][:, sim_indexes] for k in static_keys}
        static_dict = self.update_with_state_mask(static_dict, spatial_latent_vec, sim_indexes)

        rest_dict = {
            k: spatial_latent_vec[k][:, sim_indexes]
            for k in spatial_latent_vec.keys()
            if k not in (static_keys + self.keys)
        }

        # Dict for the rollouts
        rollout_dict = {
            k: torch.cat(
                [
                    spatial_latent_vec[k],
                    reg_loss.new_zeros(B, n_steps, *spatial_latent_vec[k].shape[2:]),
                ],
                dim=1,
            )
            for k in self.keys
        }

        for i in tqdm(range(n_state, n_state + n_steps, n_considered), desc="Rollout", disable=not verbose):
            # Static part
            _static_dict = {k: v[:, i - n_state : i] for k, v in static_dict.items()}

            # State for the current step
            past_dict = {k: v[:, i - n_state : i] for k, v in rollout_dict.items()}

            # Forward
            pred_dict, _reg_loss = self.step(past_dict, _static_dict, cond=cond)

            reg_loss += _reg_loss

            # e: progress index (steps taken sofar)
            # i: previus progress index
            e = min(i + n_considered, n_state + n_steps)

            rollout_dict = {
                k: torch.cat(
                    (
                        rollout_dict[k][:, :i],  # before the current step
                        pred[:, : e - i],  # the considered in the current step
                        rollout_dict[k][:, e:],  # after the current step
                    ),
                    dim=1,
                )
                for k, pred in pred_dict.items()
            }

        # add static part
        rollout_dict.update({k: static_dict[k] for k in static_keys})
        rollout_dict.update(rest_dict)

        if not with_past:
            rollout_dict = {k: v[:, n_past:] for k, v in rollout_dict.items()}

        return rollout_dict, reg_loss

    @torch.no_grad()
    def update_with_state_mask(
        self,
        static_dict: dict[str, torch.Tensor],
        spatial_latent_vec: dict[str, torch.Tensor],
        sim_indexes: list[int],
    ) -> dict[str, torch.Tensor]:
        # This is the mask of the state that is considered.
        # consist of mask (can be provided with alpha masks) and whether points are inside the workspace

        # State_mask provided
        state_mask = static_dict.get("state_mask", None)

        # Workspace limits
        if self.dataset_cfg.workspace_limits is not None:
            xyz = spatial_latent_vec["means"][:, sim_indexes]
            workspace_mask = torch.logical_and(
                (xyz > xyz.new_tensor(self.dataset_cfg.workspace_limits[0])).all(dim=-1, keepdim=True),
                (xyz < xyz.new_tensor(self.dataset_cfg.workspace_limits[1])).all(dim=-1, keepdim=True),
            )
            state_mask = workspace_mask if state_mask is None else state_mask * workspace_mask

        # Background mask
        if "background_float" in static_dict:
            non_background_mask = static_dict["background_float"] > 0.5
            state_mask = non_background_mask if state_mask is None else state_mask * non_background_mask

        if state_mask is not None:
            static_dict["state_mask"] = state_mask
        return static_dict
