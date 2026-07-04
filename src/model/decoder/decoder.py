from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

from jaxtyping import Float
from torch import Tensor, nn

from ...dataset import DatasetCfg
from ..state_adapter import DecoderInfo
from ..types import Gaussians, SpatialLatentsDict

DepthRenderingMode = Literal[
    "depth",
    "log",
    "disparity",
    "relative_disparity",
]


@dataclass
class DecoderOutput:
    color: Float[Tensor, "*batch view 3 height width"]
    depth: Float[Tensor, "*batch view height width"] | None
    extras: dict[str, Float[Tensor, "*batch view 3 height width"]] | None


T = TypeVar("T")


class Decoder(nn.Module, ABC, Generic[T]):
    cfg: T
    dataset_cfg: DatasetCfg
    decoder_info: DecoderInfo

    def __init__(self, cfg: T, dataset_cfg: DatasetCfg, decoder_info: DecoderInfo) -> None:
        super().__init__()
        self.cfg = cfg
        self.dataset_cfg = dataset_cfg
        self.decoder_info = decoder_info

    # -------------------------------------------------------------------------------
    # Abstract methods
    # -------------------------------------------------------------------------------
    @abstractmethod
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
        pass

    @abstractmethod
    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "*batch view 4 4"],
        intrinsics: Float[Tensor, "*batch view 3 3"],
        near: Float[Tensor, "*batch view"],
        far: Float[Tensor, "*batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
        extras: dict[str, Float[Tensor, "*batch gaussian one_or_three"]] | None = None,
    ) -> DecoderOutput:
        pass
