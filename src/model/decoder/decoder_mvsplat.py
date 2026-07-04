from dataclasses import dataclass
from typing import Literal

import torch
from einops import unpack
from jaxtyping import Float
from torch import Tensor, nn

from ...dataset import DatasetCfg
from ..state_adapter import DecoderInfo
from ..types import Gaussians, SpatialLatentsDict
from .decoder import Decoder
from .decoder_splatting_cuda import DecoderSplattingCUDA


@dataclass
class DecoderMVSplatCfg:
    name: Literal["mvsplat"]


class DecoderMVSplat(Decoder[DecoderMVSplatCfg]):
    background_color: Float[Tensor, "3"]

    def __init__(
        self,
        cfg: DecoderMVSplatCfg,
        dataset_cfg: DatasetCfg,
        decoder_info: DecoderInfo,
    ) -> None:
        super().__init__(cfg, dataset_cfg, decoder_info)

        self.splatting_cuda = DecoderSplattingCUDA(dataset_cfg.background_color)

        if self.decoder_info.missing_3dgs_param_shapes:
            dim_expansion = 4
            input_dim = decoder_info.input_feature_dim
            out_dim = decoder_info.raw_gaussians_dim
            self.to_missing_3dgs = nn.Sequential(
                nn.Linear(input_dim, input_dim * dim_expansion),
                nn.GELU(),
                nn.Linear(input_dim * dim_expansion, out_dim, bias=False),
                # nn.LayerNorm(out_dim),
            )

        self.missing_keys = sorted(list(self.decoder_info.missing_3dgs_param_shapes.keys()))
        self.missing_ps = [self.decoder_info.missing_3dgs_param_shapes[k] for k in self.missing_keys]

    # -------------------------------------------------------------------------------
    # The following 2 methods are just an example and can be overridden in the subclass
    # -------------------------------------------------------------------------------
    def prepare_gaussians(
        self,
        spatial_latent: SpatialLatentsDict,
    ) -> Gaussians:
        """
        1. Update the spatial_latent dictionary with the missing 3dgs parameters.
        2. If "background_float" is in the spatial_latent, multiply "opacities" and "scales" by "background_float".
        3. Add "state_mask" to Gaussian object if it exists (renderer only renders the masked splats)
        4. Add "background_float", "static_float" to Gaussian object if they exist (only used for visualization)
        r. Build the covariance matrix in world space and create a Gaussians object.
        """
        spatial_latent = self.update_with_missing_3dgs_params(spatial_latent)

        if "background_float" in spatial_latent:
            spatial_latent["opacities"] = spatial_latent["opacities"] * spatial_latent["background_float"]
            spatial_latent["scales"] = spatial_latent["scales"] * spatial_latent["background_float"]

        gaussians = Gaussians(
            means=spatial_latent["means"],
            harmonics=spatial_latent["harmonics"],
            opacities=spatial_latent["opacities"],
            rotations=spatial_latent["rotations"],
            scales=spatial_latent["scales"],
            **{
                k: spatial_latent[k]
                for k in ["state_mask", "static_float", "background_float"]
                if k in spatial_latent
            },
        )
        return gaussians

    def update_with_missing_3dgs_params(self, spatial_latent: SpatialLatentsDict):
        """
        Makes sure that the spatial_latent dictionary contains all 3dgs parameters
        ["means", "rotations", "scales", "opacities", "harmonics"]

        It uses the "dynamic/static_latent_features" to predict the missing 3dgs parameters.
        """
        if not set(self.decoder_info.missing_3dgs_param_shapes.keys()).issubset(spatial_latent.keys()):
            features = torch.cat(
                [
                    spatial_latent[k]
                    for k in ["dynamic_latent_features", "static_latent_features"]
                    if k in spatial_latent
                ],
                dim=-1,
            )
            pattern = " ".join([f"dim{i}" for i in range(features.dim() - 1)])
            raw_gaussians = self.to_missing_3dgs(features)
            missing_3dgs_list = unpack(raw_gaussians, self.missing_ps, f"{pattern} *")

            # deal with harmonics at the end (need rotations)
            for k, feat in zip(self.missing_keys, missing_3dgs_list):
                if k != "harmonics":
                    regress_func = self.decoder_info.gaussian_regress_funcs.get(k, lambda x: x)
                    spatial_latent[k] = regress_func(feat)

            if "harmonics" in self.missing_keys:
                feat = missing_3dgs_list[self.missing_keys.index("harmonics")]
                regress_func = self.decoder_info.gaussian_regress_funcs["harmonics"]
                spatial_latent["harmonics"] = regress_func(feat, spatial_latent["rotations"])

        return spatial_latent

    def forward(self, *args, **kwargs):
        output = self.splatting_cuda.forward(*args, **kwargs)

        # try:

        #     import matplotlib
        #     import matplotlib.pyplot as plt

        #     matplotlib.use("Agg")

        #     color = output.color[0, 0].permute(0, 2, 3, 1)
        #     plt.imshow(color[0].squeeze().detach().cpu())
        #     plt.savefig("aas.png")
        #     # save_ply(args[0].to_batched_gaussians()[0], "eh.ply")
        # except Exception as e:
        #     pass
        return output
