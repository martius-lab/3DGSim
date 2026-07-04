from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import einops as eo
import numpy as np
import torch
from jaxtyping import Bool, Float
from torch import Tensor

from ...dataset import DatasetCfg
from ...dataset.types import AnyViews, BatchedTempViews, DataShim, to_batched_views
from ..state_adapter import EncoderInfo, StateInfo
from ..types import SpatialLatentsDict

T = TypeVar("T")


class Encoder(torch.nn.Module, ABC, Generic[T]):
    cfg: T
    state_info: StateInfo
    encoder_info: EncoderInfo

    def __init__(
        self, cfg: T, dataset_cfg: DatasetCfg, state_info: StateInfo, encoder_info: EncoderInfo
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.dataset_cfg = dataset_cfg
        self.state_info = state_info
        self.encoder_info = encoder_info

    def get_latent_feature_dims(self) -> dict[str, int]:
        latent_feature_dims = {}
        if self.encoder_info.dynamic_latent_feature_dim > 0:
            latent_feature_dims["dynamic_latent_features"] = self.encoder_info.dynamic_latent_feature_dim
        if self.encoder_info.static_latent_feature_dim > 0:
            latent_feature_dims["static_latent_features"] = self.encoder_info.static_latent_feature_dim
        if self.encoder_info.learn_static_mask:
            latent_feature_dims["static_float"] = 1
        if self.encoder_info.learn_background_mask:
            latent_feature_dims["state_mask"] = 1
        return latent_feature_dims

    def encode(
        self,
        context: BatchedTempViews,
        deterministic: bool,
        global_step: int = -1,
    ) -> SpatialLatentsDict:
        """
        Encode the context views.
        """
        b, t = context["image"].shape[:2]

        # provided ground truth masks are overriden by the learned masks
        optionals = {
            k: context[k]
            for k in ["state_mask", "static_float"]
            if k in context and k not in self.get_latent_feature_dims()
        }

        if not self.is_temporal():
            context = to_batched_views(context)
            optionals = {k: eo.pack([v], pattern="* view c h w")[0] for k, v in optionals.items()}

        spatial_latent = self.encode_context(
            context=context,
            optionals=optionals,
            global_step=global_step,
        )
        spatial_latent = self.update_with_latent_features(spatial_latent)

        spatial_latent = masked_furthest_point_sampling(spatial_latent, self.encoder_info.fps_to_num_points)

        if not self.is_temporal():
            spatial_latent = SpatialLatentsDict(
                **{k: eo.rearrange(v, "(b t) ... -> b t ...", b=b, t=t) for k, v in spatial_latent.items()},
            )
        return spatial_latent

    def update_with_latent_features(self, spatial_latent: SpatialLatentsDict) -> SpatialLatentsDict:
        if self.encoder_info.learn_static_mask:
            nonstatic_sigmoid = spatial_latent.pop("static_float").sigmoid()
            nonstatic_mask = (
                (nonstatic_sigmoid > 0.3).float() - nonstatic_sigmoid
            ).detach() + nonstatic_sigmoid
            # static_float: used in the dynamic model to multiply delta_updates of params
            spatial_latent["static_float"] = nonstatic_mask
            spatial_latent["learned_nonstatic_sigmoid"] = nonstatic_sigmoid

        if self.encoder_info.learn_background_mask:
            # remove regressed state_mask anad add background_float instead (no state_mask in the dict)
            background_sigmoid = spatial_latent.pop("state_mask").sigmoid()
            background_mask = (
                (background_sigmoid > 0.01).float() - background_sigmoid
            ).detach() + background_sigmoid
            # background_float: used in the decoder to multiply opacities, scales
            spatial_latent["background_float"] = background_mask
            spatial_latent["learned_background_sigmoid"] = background_sigmoid

        return spatial_latent

    # ------------------- Methods to be implemented by child classes ------------------- #

    def is_temporal(self) -> bool:
        """
        Weather the encoder wants to handle the temporal dimension itself or not.
        If True, the encoder will be passed a TempViews object,
        otherwise a Views object (where we flatten the time dimension into the batch-dim).
        """
        return False

    def get_data_shim(self) -> DataShim:
        """The default shim doesn't modify the batch."""
        return lambda x: x

    @abstractmethod
    def encode_context(
        self,
        context: AnyViews,
        optionals: dict[str, Float[Tensor, "*#batch _"] | Bool[Tensor, "*#batch view c h w"]] = {},
        global_step: int = -1,
    ) -> SpatialLatentsDict:
        """
        Encode the context views.
        """
        raise NotImplementedError


def masked_furthest_point_sampling(
    spatial_latent: SpatialLatentsDict,  # B N C
    num_points: int,
) -> SpatialLatentsDict:  # B num_points C
    """
    Furthest point sampling.
    """
    current_num_points = spatial_latent["means"].shape[1]

    # 1. The dense form is decided by the biggest mask in (B)
    mask: Bool[Tensor, "B N 1"] | None = spatial_latent.pop("state_mask", None)
    if mask is not None:
        mask = mask.squeeze(-1)  # B N

        with torch.no_grad():
            batch_mask_sizes = mask.sum(dim=1)
            max_mask_n = batch_mask_sizes.max()
            new_masks = mask.new_tensor(False).repeat(mask.shape[0], max_mask_n, 1)
            for b, bms in enumerate(batch_mask_sizes):
                new_masks[b, :bms] = True
        new_spatial_dict = {"state_mask": new_masks}  # B max_mask_size C
        v: Tensor
        for k, v in spatial_latent.items():
            shape = v.shape
            new_v = v.new_tensor(np.zeros((shape[0], max_mask_n, *shape[2:])))
            new_v[new_masks.squeeze(-1)] = v[mask]
            new_spatial_dict[k] = new_v

        current_num_points = max_mask_n.item()
    else:
        new_spatial_dict = spatial_latent
        new_masks = None

    # 2. FPS
    if current_num_points > num_points and num_points > 0:
        from pytorch3d.ops.sample_farthest_points import masked_gather, sample_farthest_points

        means, fps_idx = sample_farthest_points(new_spatial_dict.pop("means"), K=num_points)

        for k, v in new_spatial_dict.items():
            v_packed, ps = eo.pack([v], pattern="b n *")
            v_new = masked_gather(v_packed, fps_idx)
            new_spatial_dict[k] = eo.unpack(v_new, ps, pattern="b n *")[0]
        new_spatial_dict["means"] = means

    return new_spatial_dict
