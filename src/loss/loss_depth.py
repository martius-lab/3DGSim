from dataclasses import dataclass

import torch
from einops import reduce
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedTempExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossDepthCfg:
    weight: float
    sigma_image: float | None
    use_second_derivative: bool


@dataclass
class LossDepthCfgWrapper:
    depth: LossDepthCfg


class LossDepth(Loss[LossDepthCfg, LossDepthCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedTempExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, " time"]:
        # Scale the depth between the near and far planes.
        near = batch["target"]["near"][..., None, None].log()
        far = batch["target"]["far"][..., None, None].log()
        depth = prediction.depth.minimum(far).maximum(near)
        depth = (depth - near) / (far - near)

        # Compute the difference between neighboring pixels in each direction.
        depth_dx = depth.diff(dim=-1)
        depth_dy = depth.diff(dim=-2)

        # If desired, compute a 2nd derivative.
        if self.cfg.use_second_derivative:
            depth_dx = depth_dx.diff(dim=-1)
            depth_dy = depth_dy.diff(dim=-2)

        # If desired, add bilateral filtering.
        if self.cfg.sigma_image is not None:
            color_gt = batch["target"]["image"]
            color_dx = reduce(color_gt.diff(dim=-1), "b ... c h w -> b ... h w", "max")
            color_dy = reduce(color_gt.diff(dim=-2), "b ... c h w -> b ... h w", "max")
            if self.cfg.use_second_derivative:
                color_dx = color_dx[..., :, 1:].maximum(color_dx[..., :, :-1])
                color_dy = color_dy[..., 1:, :].maximum(color_dy[..., :-1, :])
            depth_dx = depth_dx * torch.exp(-color_dx * self.cfg.sigma_image)
            depth_dy = depth_dy * torch.exp(-color_dy * self.cfg.sigma_image)  # (b, ..., h, w)

        depth_dx_loss = reduce(depth_dx.abs(), "b t ... -> t", "mean")
        depth_dy_loss = reduce(depth_dy.abs(), "b t ... -> t", "mean")

        return self.cfg.weight * (depth_dx_loss + depth_dy_loss)
