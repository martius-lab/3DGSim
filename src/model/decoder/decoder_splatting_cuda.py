import math

import torch
from einops import pack, repeat, unpack
from jaxtyping import Float
from torch import Tensor

from ..types import Gaussians
from .cuda_splatting import DepthRenderingMode, render_cuda, render_depth_cuda
from .decoder import Decoder, DecoderOutput


class DecoderSplattingCUDA(torch.nn.Module):
    background_color: Float[Tensor, "3"]

    def __init__(self, background_color: list[float]) -> None:
        super().__init__()
        self.register_buffer(
            "background_color",
            torch.tensor(background_color, dtype=torch.float32),
            persistent=False,
        )

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
        # --------------------------------------------
        # Prepare inputs for CUDA kernel.
        # --------------------------------------------
        v = near.shape[-1]

        extrinsics, ps = pack([extrinsics], "* i j")
        intrinsics, ps = pack([intrinsics], "* i j")
        near, ps = pack([near], "*")
        far, ps = pack([far], "*")
        bckg = repeat(self.background_color, "c -> bb c", bb=math.prod(ps[0]))

        means = repeat(pack([gaussians.means], "* g xyz")[0], "bb g xyz -> (bb v) g xyz", v=v)
        covariance = repeat(
            pack([gaussians.get_covariances()], "* g i j")[0], "bb g i j -> (bb v) g i j", v=v
        )
        harmonics = repeat(pack([gaussians.harmonics], "* g c sh")[0], "bb g c sh -> (bb v) g c sh", v=v)
        opacities = repeat(pack([gaussians.opacities], "* g one")[0], "bb g one -> (bb v) g one", v=v)
        if gaussians.state_mask is not None:
            masks = repeat(pack([gaussians.state_mask], "* g c")[0], "bb g 1 -> (bb v) g 1", v=v)
        else:
            masks = None

        # --------------------------------------------
        # Call CUDA kernel.
        # --------------------------------------------
        color = render_cuda(
            extrinsics,
            intrinsics,
            near,
            far,
            image_shape,
            bckg,
            means,
            covariance,
            harmonics,
            opacities,
            masks,
        )
        [color] = unpack(color, ps, "* c h w")

        depth = None
        if depth_mode is not None:
            # Call CUDA kernel.
            depth = render_depth_cuda(
                extrinsics,
                intrinsics,
                near,
                far,
                image_shape,
                means,
                covariance,
                opacities,
                masks,
                mode=depth_mode,
            )
            [depth] = unpack(depth, ps, "* h w")

        extra_outputs = None
        if extras is not None:
            extra_outputs = {}
            for key, extra in extras.items():
                channel_dim = extra.size(-1)

                extra = repeat(pack([extra], "* g c")[0], "bb g c -> (bb v) g c", v=v)
                if channel_dim == 1:
                    extra = torch.cat([extra, torch.zeros_like(extra), 1 - extra], dim=-1)
                    # extra = repeat(extra, "b g -> b g c", c=3),
                elif channel_dim != 3:
                    # Skip invalid extra output.
                    continue

                extra_img = render_cuda(
                    extrinsics,
                    intrinsics,
                    near,
                    far,
                    image_shape,
                    bckg,
                    means,
                    covariance,
                    extra.unsqueeze(-1),
                    opacities,
                    masks,
                )
                [extra_outputs[key]] = unpack(extra_img, ps, "* c h w")

        return DecoderOutput(color, depth, extra_outputs)
