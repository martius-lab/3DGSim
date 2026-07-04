import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Literal, Optional

import torch
from einops import rearrange, repeat
from jaxtyping import Bool, Float
from torch import Tensor, nn

from ...dataset import DatasetCfg
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import AnyExample, DataShim
from ...geometry.projection import get_means_from_depth, get_world_pixel_size
from ...geometry.rotation_conversions import rotmat_to_repr, to_rot_mat
from ...global_cfg import get_cfg
from ...misc.sh_rotation import rotate_sh
from ..encodings.positional_encoding import PositionalEncoding
from ..state_adapter import EncoderInfo, StateInfo
from .backbone import BackboneMultiview
from .costvolume.depth_predictor_multiview import DepthPredictorMultiView
from .encoder import BatchedTempViews, Encoder, SpatialLatentsDict
from .epipolar.epipolar_sampler import EpipolarSampler
from .visualization.encoder_visualizer_costvolume_cfg import EncoderVisualizerCostVolumeCfg


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderCostVolumeCfg:
    name: Literal["costvolume"]
    d_feature: int
    num_depth_candidates: int
    num_surfaces: int
    visualizer: EncoderVisualizerCostVolumeCfg
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    unimatch_weights_path: str | None
    downscale_factor: int
    shim_patch_size: int
    multiview_trans_attn_split: int
    costvolume_unet_feat_dim: int
    costvolume_unet_channel_mult: List[int]
    costvolume_unet_attn_res: List[int]
    depth_unet_feat_dim: int
    depth_unet_attn_res: List[int]
    depth_unet_channel_mult: List[int]
    wo_depth_refine: bool
    wo_cost_volume: bool
    wo_backbone_cross_attn: bool
    wo_cost_volume_refine: bool
    use_epipolar_trans: bool


class EncoderCostVolume(Encoder[EncoderCostVolumeCfg]):
    backbone: BackboneMultiview
    depth_predictor: DepthPredictorMultiView
    sh_mask: Tensor

    def __init__(
        self,
        cfg: EncoderCostVolumeCfg,
        dataset_cfg: DatasetCfg,
        state_info: StateInfo,
        encoder_info: EncoderInfo,
    ) -> None:
        super().__init__(cfg, dataset_cfg, state_info, encoder_info)
        assert {"means", "opacities"}.issubset(
            state_info.state_shapes.keys()
        ), "EncoderCostVolume needs opacities and means in state shapes"

        self.gradient_checkpointing = get_cfg().train.gradient_checkpointing
        print("CostVolume Gradient Checkpointing:", self.gradient_checkpointing)

        # multi-view Transformer backbone
        if cfg.use_epipolar_trans:
            self.epipolar_sampler = EpipolarSampler(
                num_views=get_cfg().dataset.view_sampler.num_context_views,
                num_samples=32,
            )
            self.depth_encoding = nn.Sequential(
                (pe := PositionalEncoding(10)),
                nn.Linear(pe.d_out(1), cfg.d_feature),
            )
        self.backbone = BackboneMultiview(
            feature_channels=cfg.d_feature,
            downscale_factor=cfg.downscale_factor,
            no_cross_attn=cfg.wo_backbone_cross_attn,
            use_epipolar_trans=cfg.use_epipolar_trans,
        )
        ckpt_path = cfg.unimatch_weights_path
        if get_cfg().mode == "train":
            if cfg.unimatch_weights_path is None:
                print("==> Init multi-view transformer backbone from scratch")
            else:
                print("==> Load multi-view transformer backbone checkpoint: %s" % ckpt_path)
                unimatch_pretrained_model = torch.load(ckpt_path)["model"]
                updated_state_dict = OrderedDict(
                    {k: v for k, v in unimatch_pretrained_model.items() if k in self.backbone.state_dict()}
                )
                # NOTE: when wo cross attn, we added ffns into self-attn, but they have no pretrained weight
                is_strict_loading = not cfg.wo_backbone_cross_attn
                self.backbone.load_state_dict(updated_state_dict, strict=is_strict_loading)

        # cost volume based depth predictor
        self.depth_predictor = DepthPredictorMultiView(
            feature_channels=cfg.d_feature,
            upscale_factor=cfg.downscale_factor,
            num_depth_candidates=cfg.num_depth_candidates,
            costvolume_unet_feat_dim=cfg.costvolume_unet_feat_dim,
            costvolume_unet_channel_mult=tuple(cfg.costvolume_unet_channel_mult),
            costvolume_unet_attn_res=tuple(cfg.costvolume_unet_attn_res),
            gaussian_raw_channels=encoder_info.raw_gaussians_dim - 2,
            monoc_feat_dims_dict=self.get_latent_feature_dims(),
            num_surfaces=cfg.num_surfaces,
            gaussians_per_pixel=cfg.gaussians_per_pixel,
            num_views=get_cfg().dataset.view_sampler.num_context_views,
            depth_unet_feat_dim=cfg.depth_unet_feat_dim,
            depth_unet_attn_res=cfg.depth_unet_attn_res,
            depth_unet_channel_mult=cfg.depth_unet_channel_mult,
            wo_depth_refine=cfg.wo_depth_refine,
            wo_cost_volume=cfg.wo_cost_volume,
            wo_cost_volume_refine=cfg.wo_cost_volume_refine,
        )
        # Create a mask for the spherical harmonics coefficients. This ensures that at
        # initialization, the coefficients are biased towards having a large DC
        # component and small view-dependent components.
        sh_degree = self.encoder_info.gaussians.sh_degree
        d_sh = (sh_degree + 1) ** 2
        self.register_buffer(
            "sh_mask",
            torch.ones((d_sh,), dtype=torch.float32),
            persistent=False,
        )
        for degree in range(1, sh_degree + 1):
            self.sh_mask[degree**2 : (degree + 1) ** 2] = 0.1 * 0.25**degree

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: AnyExample) -> AnyExample:
            batch = apply_patch_shim(
                batch,
                patch_size=self.cfg.shim_patch_size * self.cfg.downscale_factor,
            )

            # if self.cfg.apply_bounds_shim:
            #     from ...dataset.shims.bounds_shim import apply_bounds_shim
            #     _, _, _, h, w = batch["context"]["image"].shape
            #     near_disparity = self.cfg.near_disparity * min(h, w)
            #     batch = apply_bounds_shim(batch, near_disparity, self.cfg.far_disparity)

            return batch

        return data_shim

    @property
    def sampler(self):
        # hack to make the visualizer work
        return None

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def forward(
        self,
        context: dict,
        global_step: int,
        deterministic: bool = False,
        visualization_dump: Optional[dict] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, dict[str, Tensor]]:
        # Encode the context images.
        if self.cfg.use_epipolar_trans:
            epipolar_kwargs = {
                "epipolar_sampler": self.epipolar_sampler,
                "depth_encoding": self.depth_encoding,
                "extrinsics": context["extrinsics"],
                "intrinsics": context["intrinsics"],
                "near": context["near"],
                "far": context["far"],
            }
        else:
            epipolar_kwargs = None
        trans_features, cnn_features = self.backbone(
            context["image"],
            attn_splits=self.cfg.multiview_trans_attn_split,
            return_cnn_features=True,
            epipolar_kwargs=epipolar_kwargs,
        )

        # Sample depths from the resulting features.
        in_feats = trans_features
        extra_info = {}
        extra_info["images"] = rearrange(context["image"], "b v c h w -> (v b) c h w")

        # b v hw srf c
        if self.gradient_checkpointing:
            raw_gaussians, xy_offsets, depths, densities, learned_latents_dict = (
                torch.utils.checkpoint.checkpoint(
                    self.depth_predictor,
                    in_feats,
                    context["intrinsics"],
                    context["extrinsics"],
                    context["near"],
                    context["far"],
                    deterministic,
                    extra_info,
                    cnn_features,
                )
            )
        else:
            raw_gaussians, xy_offsets, depths, densities, learned_latents_dict = self.depth_predictor(
                in_feats,
                context["intrinsics"],
                context["extrinsics"],
                context["near"],
                context["far"],
                deterministic=deterministic,
                extra_info=extra_info,
                cnn_features=cnn_features,
            )

        # prepare shapes (b, v, hw, srf, dpt, #)
        raw_gaussians = rearrange(raw_gaussians, "b v hw srf c -> b v hw srf () c")
        xy_offsets = rearrange(xy_offsets, "b v hw srf c -> b v hw srf () c")
        depths = rearrange(depths, "b v hw srf dpt -> b v hw srf dpt ()")
        densities = rearrange(densities, "b v hw srf dpt -> b v hw srf dpt ()")
        learned_latents_dict = {
            k: rearrange(v, "b v hw srf c -> b v hw srf () c") for k, v in learned_latents_dict.items()
        }

        opacities = self.map_pdf_to_opacity(densities, global_step) / self.cfg.gaussians_per_pixel

        return raw_gaussians, xy_offsets, depths, opacities, learned_latents_dict

    def get_state_dict(
        self,
        extrinsics: Float[Tensor, "*#batch 4 4"],
        intrinsics: Float[Tensor, "*#batch 3 3"],
        offset_xy: Float[Tensor, "*#batch 2"],
        depths: Float[Tensor, "*#batch 1"],
        opacities: Float[Tensor, "*#batch 1"],
        raw_gaussians: Float[Tensor, "*#batch _"],
        image_shape_hw: tuple[int, int],
        other_features: dict[str, Float[Tensor, "*#batch _"] | Bool[Tensor, "*#batch _"]] = {},
        eps: float = 1e-8,
    ) -> SpatialLatentsDict:

        split_keys = [k for k in ["scales", "rotations", "harmonics"] if k in self.state_info.state_shapes]
        split_dims = [math.prod(self.state_info.state_shapes[k]) for k in split_keys]
        splits = torch.split(raw_gaussians, split_dims, dim=-1)

        # Prepare the parameters for the Gaussians.
        gs_params = dict(**other_features)
        gs_params["opacities"] = opacities
        gs_params["means"] = get_means_from_depth(depths, offset_xy, extrinsics, intrinsics, image_shape_hw)

        if "scales" in split_keys:
            scales = splits[split_keys.index("scales")]
            # Map scale features to valid scale range.
            scale_min = self.encoder_info.gaussians.cam_scale_min
            scale_max = self.encoder_info.gaussians.cam_scale_max
            scales = scale_min + (scale_max - scale_min) * scales.sigmoid()
            # Get scales in world frame (orientation still locally)
            multiplier = 0.1
            pixel_size_w = get_world_pixel_size(intrinsics, image_shape_hw)
            gs_params["scales"] = scales * depths * multiplier * pixel_size_w[..., None]

        if "rotations" in split_keys:
            rotations = splits[split_keys.index("rotations")]
            # Create world-space rotataion
            c2w_rotations = extrinsics[..., :3, :3]
            rotations = to_rot_mat(rotations, normalize=True, eps=eps)
            gs_params["rotations"] = rotmat_to_repr(
                c2w_rotations @ rotations, self.encoder_info.gaussians.rotations
            )

        if "harmonics" in split_keys:
            sh = splits[split_keys.index("harmonics")]
            # Apply sigmoid to get valid colors.
            # harmonics in world space (use c2w because harmonics are in camera space)
            c2w_rotations = extrinsics[..., :3, :3]
            sh = rearrange(sh, "... (xyz d_sh) -> ... xyz d_sh", xyz=3)
            sh = sh * self.sh_mask
            gs_params["harmonics"] = rotate_sh(sh, c2w_rotations.unsqueeze(-3))

            # in case we want local harmonics w.r.t rotation matrix
            # rot_T = rearrange(rotations, "... i j -> ... j i").unsqueeze(-3)
            # harmonics2 = rotate_sh(rotate_sh(sh, rot_T), new_rotations.unsqueeze(-3))
            # print(harmonics.abs().max(), harmonics.abs().min())
            # print("1", torch.allclose(harmonics, harmonics2, rtol=1e2, atol=1e-4))

        spatial_latent = SpatialLatentsDict(
            **{k: rearrange(v, "b v hw srf spp ... -> b (v hw srf spp) ...") for k, v in gs_params.items()}
        )
        return spatial_latent

    def encode_context(
        self,
        context: BatchedTempViews,
        global_step: int,
        optionals: dict[str, Float[Tensor, "b view c h w"] | Bool[Tensor, "b view c h w"]] = {},
    ):
        """
        Encode the context views.
        """

        # Run the Encoder. B,View.. -> B,View,Gaussian..
        # learned_latents_dict includes [dynamic_latent_features, static_latent_features, static_float, state_mask]
        raw_gaussians, xy_offsets, depths, opacities, learned_latents_dict = self(
            context=context,
            global_step=global_step,
            deterministic=False,
        )
        # Per pixel optional features to force on the splats
        srf, dpt = raw_gaussians.shape[-3:-1]
        other_features = {
            k: repeat(v, "... v c h w -> ... v (h w) srf dpt c", srf=srf, dpt=dpt)
            for k, v in optionals.items()
        }
        other_features.update(learned_latents_dict)

        # State dict
        spatial_dict = self.get_state_dict(
            rearrange(context["extrinsics"], "... v i j -> ... v () () () i j"),
            rearrange(context["intrinsics"], "... v i j -> ... v () () () i j"),
            xy_offsets.sigmoid(),
            depths,
            opacities,
            raw_gaussians,
            context["image"].shape[-2:],
            other_features=other_features,
        )

        return spatial_dict
