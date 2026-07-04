import math
from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Literal, Optional, get_args

import einops as eo
import torch
from jaxtyping import Float
from torch import Tensor, nn

from ..geometry.rotation_conversions import increment_increments, increment_rot, normalize_repr, to_rot_mat
from ..misc.sh_rotation import rotate_sh

CX, Z, X, DX = "cx", "z", "x", "dx"
ModalityT = Literal[f"{Z}_{X}", f"{CX}_{X}", f"{CX}_{DX}"]
GS_PARAM_T = Literal["means", "rotations", "scales", "opacities", "harmonics"]
GS_PARAM = list(get_args(GS_PARAM_T))


@dataclass
class GaussiansCfg:
    sh_degree: int = 4
    rotations: int = 4  # 4 for quaternion, 6 for gramm matrix
    scale_min: float = 0.004
    scale_max: float = 0.14
    cam_scale_min: float = 0.5
    cam_scale_max: float = 15.0


@dataclass
class StateCfg:
    metric: float = 0.1  # the unit is: 0.1*meter
    pred_inductive_bias: bool = False  # whether to use inductive bias on how big deltas can be

    fps_to_num_points: int = -1  # -1: use all points
    gaussians: GaussiansCfg = field(default_factory=lambda: GaussiansCfg())
    # which 3dgs param should be regressed by encoder
    state_3dgs_params: list[GS_PARAM_T] = field(default_factory=lambda: GS_PARAM)

    # 1. encoder related
    # masks
    learn_background_mask: bool = False
    learn_static_mask: bool = False
    # dimension of the latent features regressed by encoder
    static_latent_feature_dim: int = 0
    dynamic_latent_feature_dim: int = 0

    # 2. dynamic model related
    pred_modality: dict[str, ModalityT] = field(default_factory=lambda: {})
    dynamic_state_keys: list[str] = field(default_factory=lambda: [])
    immune_to_static: list[str] = field(default_factory=lambda: [])

    def __post_init__(self):
        self.static_latent_feature_dim = max(0, self.static_latent_feature_dim)
        self.dynamic_latent_feature_dim = max(0, self.dynamic_latent_feature_dim)

        print(
            f"\nStateAdapter:\n",
            f" 3dgs params: {self.state_3dgs_params}\n",
            f" learn_background_mask: {self.learn_background_mask} learn_static_mask: {self.learn_static_mask}\n",
            f" static_latent_feature_dim: {self.static_latent_feature_dim} dynamic_latent_feature_dim: {self.dynamic_latent_feature_dim}\n",
            f" pred_modality: {self.pred_modality}\n",
            f" dynamic_state_keys: {self.dynamic_state_keys}\n",
            f" immune_to_static: {self.immune_to_static}\n",
            f" fps_to_num_points: {self.fps_to_num_points}\n",
        )


# ----------------------------------------------------------------------------------
# Informations shared from the StateAdapter to the Encoder, DynamicModel and Decoder
# ----------------------------------------------------------------------------------
@dataclass
class StateInfo:
    # State related information (mostly for the dynamic model)
    state_shapes: dict[str, tuple[int, ...]]  # shapes of all(dynamic+static) state-features
    state_increment_shapes: dict[str, tuple[int, ...]]
    state_update_funcs: dict[str, Callable]  # used in dynamic model to update the state-features

    pred_modality: dict[str, ModalityT]
    dynamic_state_keys: list[str]
    immune_to_static: list[str]


@dataclass
class EncoderInfo:
    gaussians: GaussiansCfg
    fps_to_num_points: int
    raw_gaussians_dim: int
    static_latent_feature_dim: int
    dynamic_latent_feature_dim: int

    learn_background_mask: bool
    learn_static_mask: bool
    workspace_limits: list[list[float]] | None


@dataclass
class DecoderInfo:
    missing_3dgs_param_shapes: dict[str, tuple[int, ...]]
    raw_gaussians_dim: int
    input_feature_dim: int
    gaussian_regress_funcs: dict[str, Callable]  # used in decoder to regress the missing 3dgs params
    workspace_limits: list[list[float]] | None


class StateAdapter(nn.Module):
    cfg: StateCfg
    workspace_limits: list[list[float]] | None

    sh_mask: Tensor

    def __init__(self, cfg: StateCfg, workspace_limits: list[list[float]] | None = None) -> None:
        super().__init__()
        self.cfg = cfg
        self.workspace_limits = workspace_limits

        # Create a mask for the spherical harmonics coefficients. This ensures that at
        # initialization, the coefficients are biased towards having a large DC
        # component and small view-dependent components.
        self.register_buffer(
            "sh_mask",
            torch.ones((self.d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, self.cfg.gaussians.sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

    @property
    def d_sh(self) -> int:
        return (self.cfg.gaussians.sh_degree + 1) ** 2

    # ----------------------------------------------------------------------------
    # Encoder related methods
    # ----------------------------------------------------------------------------
    def get_encoder_info(self) -> EncoderInfo:
        """
        Information about the 3dgs-parameters and optionally the latent feature
        that are regressed by the encoder

        returns dict with dimension of parameters regressed by encoder: 3dgs_params, latent_features
        """
        gaussian_shapes = self.get_3dgs_shapes()

        encoder_state_dim = EncoderInfo(
            gaussians=self.cfg.gaussians,
            fps_to_num_points=self.cfg.fps_to_num_points,
            raw_gaussians_dim=sum([math.prod(gaussian_shapes[k]) for k in self.cfg.state_3dgs_params]),
            dynamic_latent_feature_dim=self.cfg.dynamic_latent_feature_dim,
            static_latent_feature_dim=self.cfg.static_latent_feature_dim,
            learn_background_mask=self.cfg.learn_background_mask,
            learn_static_mask=self.cfg.learn_static_mask,
            workspace_limits=self.workspace_limits,
        )
        return encoder_state_dim

    # ----------------------------------------------------------------------------
    # Decoder related methods
    # ----------------------------------------------------------------------------
    def get_decoder_info(self) -> DecoderInfo:
        """
        Information about 3dgs-parameters that are not regressed by the encoder
        and should be regressed by the decoder.
        """
        gaussian_shapes = self.get_3dgs_shapes()
        missing_3dgs_keys = list(set(GS_PARAM) - set(self.cfg.state_3dgs_params))

        decoder_state_dim = DecoderInfo(
            missing_3dgs_param_shapes={k: gaussian_shapes[k] for k in missing_3dgs_keys},
            raw_gaussians_dim=sum([math.prod(gaussian_shapes[k]) for k in missing_3dgs_keys]),
            input_feature_dim=self.cfg.dynamic_latent_feature_dim + self.cfg.static_latent_feature_dim,
            gaussian_regress_funcs={
                "rotations": self.regress_rotations,
                "scales": self.regress_scale,
                "opacities": self.regress_opacity,
                "harmonics": self.regress_harmonics,
            },
            workspace_limits=self.workspace_limits,
        )
        return decoder_state_dim

    # ----------------------------------------------------------------------------
    # State related methods (params regressed by the encoder)
    # ----------------------------------------------------------------------------
    def get_state_info(self) -> StateInfo:
        """
        StateInfo includes information about all keys in the state regressed by the encoder:
        - shapes of all state-features [state_shapes]
        - shapes of the increments [state_increment_shapes] - relevant only for params on manifold, e.g. rotations, harmonics
        - update functions for the state-features [state_update_funcs]

        State features that are updated by the dynamic model
        - the changing state keys [dynamic_state_keys]
        - the prediction modality for each changing state key [pred_modality]

        This is used primarly in the dynamic_model

        Returns the shapes of the state-features, the increments and the update functions
        """
        state_info = StateInfo(
            state_shapes=self._get_state_shapes(),
            # the changing state-features' keys
            state_increment_shapes=self._get_state_increment_shapes(),
            state_update_funcs=self._get_state_update_funcs(),
            pred_modality=self._get_pred_modality(),
            dynamic_state_keys=self._get_dynamic_state_keys(),
            immune_to_static=list(set(self.cfg.immune_to_static + ["dynamic_latent_features"])),
        )
        return state_info

    def _get_state_shapes(self) -> dict[str, tuple[int, ...]]:
        # Shapes of the Gaussian state-features (means, (optional)3dgs_params, (optional)dynamic/static_latent_features)
        gaussian_shapes = self.get_3dgs_shapes()
        state_shapes = {k: gaussian_shapes[k] for k in self.cfg.state_3dgs_params}
        if self.cfg.dynamic_latent_feature_dim > 0:
            state_shapes["dynamic_latent_features"] = (self.cfg.dynamic_latent_feature_dim,)
        if self.cfg.static_latent_feature_dim > 0:
            state_shapes["static_latent_features"] = (self.cfg.static_latent_feature_dim,)
        return state_shapes

    def _get_state_increment_shapes(self) -> dict[str, tuple[int, ...]]:
        # Shapes of the Gaussian state-feature increments
        # (only relevant for params on manifold, e.g. rotations, harmonics)
        # We provide it for all state-features, but normally only needed for changing state-features
        gaussian_increment_shapes = self.get_3dgs_increment_shapes()
        state_increment_shapes = {k: gaussian_increment_shapes[k] for k in self.cfg.state_3dgs_params}
        if self.cfg.dynamic_latent_feature_dim > 0:
            state_increment_shapes["dynamic_latent_features"] = (self.cfg.dynamic_latent_feature_dim,)
        return state_increment_shapes

    def _get_state_update_funcs(self) -> dict[str, Callable]:
        # Functions to increment the state-features
        # (use default update_func for latent_features,)
        # We provide it for all state-features, but normally only needed for changing state-features
        gaussian_update_funcs = self.get_3dgs_update_functions()
        state_update_funcs = {k: gaussian_update_funcs[k] for k in self.cfg.state_3dgs_params}
        return state_update_funcs

    def _get_dynamic_state_keys(self) -> list[str]:
        # Keys of the changing state-features that are updated by the dynamic model
        dynamic_state_keys = self.cfg.dynamic_state_keys
        dynamic_state_keys.append("means")

        if self.cfg.dynamic_latent_feature_dim > 0:
            dynamic_state_keys.append("dynamic_latent_features")
        if "static_latent_features" in dynamic_state_keys:
            dynamic_state_keys.remove("static_latent_features")

        return list(set(dynamic_state_keys))

    def _get_pred_modality(self) -> dict[str, ModalityT]:
        # Prediction modality for each changing state-feature
        pred_modality = self.cfg.pred_modality
        for k in self._get_dynamic_state_keys():
            if k not in pred_modality:
                pred_modality[k] = f"{Z}_{X}"  # absolute future_predictions by default
        return pred_modality

    # --------------------------------------------------------------------------
    # 3DGS related methods
    # --------------------------------------------------------------------------
    def get_3dgs_dim(self) -> int:
        # Total dimension of the Gaussian features
        feat_dim = sum([math.prod(s) for s in self.get_3dgs_shapes().values()])
        return feat_dim

    def get_3dgs_shapes(self) -> dict[str, tuple[int, ...]]:
        # Shapes of the Gaussian features
        return {
            "means": (3,),
            "rotations": (self.cfg.gaussians.rotations,),
            "scales": (3,),
            "opacities": (1,),
            "harmonics": (3, self.d_sh),
        }

    def get_3dgs_increment_shapes(self) -> dict[str, tuple[int, ...]]:
        # Shapes of the Gaussian feature-increments
        return {
            "means": (3,),
            "rotations": (3,),
            "scales": (3,),
            "opacities": (1,),
            "harmonics": (3, self.d_sh),
        }

    def get_3dgs_update_functions(self):
        return {
            "means": self.update_mean,
            "rotations": self.update_rotation,
            "scales": self.update_scale,
            "opacities": self.update_opacity,
            "harmonics": self.update_harmonics,
        }

    # --------------------------------------------------------------------------
    # State regress functions
    # --------------------------------------------------------------------------
    def regress_rotations(self, feat: Float[Tensor, "*#batch d"]) -> Float[Tensor, "*#batch d"]:
        return normalize_repr(feat)

    def regress_scale(self, feat: Float[Tensor, "*#batch 3"]) -> Float[Tensor, "*#batch 3"]:
        scale_min = self.cfg.gaussians.scale_min
        scale_max = self.cfg.gaussians.scale_max
        scale = scale_min + (scale_max - scale_min) * feat.sigmoid()
        return scale

    def regress_opacity(self, feat: Float[Tensor, "*#batch 1"]) -> Float[Tensor, "*#batch 1"]:
        return feat.sigmoid()

    def regress_harmonics(
        self,
        feat: Float[Tensor, "*#batch xyz d_sh"],
        rotations: Float[Tensor, "*#batch d_rot"],
    ) -> Float[Tensor, "*#batch xyz d_sh"]:
        sh = feat * self.sh_mask
        # transform the harmonics to the world coordinate system defined by rotations
        rot_mat = to_rot_mat(rotations, normalize=True)
        harmonics = rotate_sh(sh, rot_mat.unsqueeze(-3))
        return harmonics

    # --------------------------------------------------------------------------
    # State update functions
    # --------------------------------------------------------------------------
    def update_mean(
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
            past = eo.repeat(past, "b t ... -> b n_pred t ...", n_pred=n_pred)

        if k == "z_x":
            delta_future = pred - past
        else:
            if self.cfg.pred_inductive_bias:
                pred = self.cfg.metric * 0.5 * pred.tanh()

            if k == "cx_x":
                delta_future = pred
            elif k == "cx_dx":
                delta_future = pred.cumsum(dim=1)
            else:
                raise ValueError(f"Unknown modality: {k}")

        if static_float is not None:
            delta_future = static_float * delta_future

        future = eo.rearrange(past + delta_future, "b n_pred t g ... -> b (n_pred t) g ...")
        return future

    def update_rotation(
        self,
        past: Float[Tensor, "batch time gaussian d"],
        pred: Float[Tensor, "batch n_pred time gaussian one"],  # c is 3 or 9
        static_float: Optional[Float[Tensor, "batch n_pred time gaussian 1"]],
        k: str,
    ) -> Float[Tensor, "batch time_pred gaussian d"]:
        # k is one of z_x: absolute future_predictions
        #             cx_x: relative future_predictions
        #             cx_dx relative future_predictions increments
        n_pred = pred.size(1)

        if n_pred > 1:
            past = eo.repeat(past, "b t ... -> b n_pred t ...", n_pred=n_pred)
            if static_float is not None:
                pred = static_float * pred
        else:
            static_float = 1.0

        if k == "z_x":
            past = past * (1.0 - static_float)
            future = past + pred
        elif k == "cx_x":
            future = increment_rot(past, pred)
        elif k == "cx_dx":
            future = increment_rot(past, increment_increments(pred))
        else:
            raise ValueError(f"Unknown modality: {k}")

        future = eo.rearrange(future, "b n_pred t g ... -> b (n_pred t) g ...")
        return future

    def update_scale(
        self,
        past: Float[Tensor, "batch time gaussian xyz"],
        pred: Float[Tensor, "batch n_pred time gaussian xyz"],
        static_float: Optional[Float[Tensor, "batch n_pred time gaussian 1"]],
        k: str,
    ) -> Float[Tensor, "batch time_pred gaussian xyz"]:
        # k is one of z_x: absolute future_predictions
        #             cx_x: relative future_predictions
        #             cx_dx relative future_predictions increments
        scale_min = self.cfg.gaussians.scale_min
        scale_max = self.cfg.gaussians.scale_max

        n_pred = pred.size(1)
        if n_pred > 1:
            past = eo.repeat(past, "b t ... -> b n_pred t ...", n_pred=n_pred)

        if k == "z_x":
            delta_future = scale_min + (scale_max - scale_min) * pred.sigmoid() - past
        else:
            if self.cfg.pred_inductive_bias:
                max_delta = (scale_max - scale_min) / 6.0
                pred = max_delta * pred.tanh()

            if k == "cx_x":
                delta_future = pred
            elif k == "cx_dx":
                delta_future = pred.cumsum(dim=1)
            else:
                raise ValueError(f"Unknown modality: {k}")

        if static_float is not None:
            delta_future = static_float * delta_future

        future = eo.rearrange(past + delta_future, "b n_pred t g ... -> b (n_pred t) g ...")
        return future.clip(min=scale_min, max=scale_max)

    def update_opacity(
        self,
        past: Float[Tensor, "batch time gaussian one"],
        pred: Float[Tensor, "batch n_pred time gaussian one"],
        static_float: Optional[Float[Tensor, "batch n_pred time gaussian 1"]],
        k: str,
    ) -> Float[Tensor, "batch time_pred gaussian one"]:
        # k is one of z_x: absolute future_predictions
        #             cx_x: relative future_predictions
        #             cx_dx relative future_predictions increments
        n_pred = pred.size(1)
        if n_pred > 1:
            past = eo.repeat(past, "b t ... -> b n_pred t ...", n_pred=n_pred)

        if k == "z_x":
            delta_future = pred.sigmoid() - past
        elif k == "cx_x":
            delta_future = pred
        elif k == "cx_dx":
            delta_future = torch.cumsum(pred, dim=1)
        else:
            raise ValueError(f"Unknown modality: {k}")

        if static_float is not None:
            delta_future = static_float * delta_future

        future = eo.rearrange(past + delta_future, "b n_pred t g ... -> b (n_pred t) g ...")
        return future.clip(min=0.0)

    def update_harmonics(
        self,
        past: Float[Tensor, "batch time gaussian xyz d_sh"],
        pred: Float[Tensor, "batch n_pred time gaussian xyz d_sh"],
        static_float: Optional[Float[Tensor, "batch n_pred time gaussian 1"]],
        k: str,
    ) -> Float[Tensor, "batch t_pred gaussian xyz d_sh"]:
        # k is one of z_x: absolute future_predictions
        #             cx_x: relative future_predictions
        #             cx_dx relative future_predictions increments
        # print("sh_mask", sh_mask)
        n_pred = pred.size(1)
        if n_pred > 1:
            past = eo.repeat(past, "b t ... -> b n_pred t ...", n_pred=n_pred)

        # Scale the harmonics by the mask
        scaled_pred = pred * self.sh_mask
        if k == "z_x":
            delta_future = scaled_pred - past
        elif k == "cx_x":
            delta_future = scaled_pred
        elif k == "cx_dx":
            delta_future = torch.cumsum(scaled_pred, dim=1)
        else:
            raise ValueError(f"Unknown modality: {k}")

        if static_float is not None:
            delta_future = static_float.unsqueeze(-1) * delta_future

        future = eo.rearrange(past + delta_future, "b n_pred t g ... -> b (n_pred t) g ...")
        return future
