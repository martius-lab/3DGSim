from dataclasses import dataclass

import torch
from einops import pack, reduce, unpack
from jaxtyping import Float
from lpips import LPIPS
from torch import Tensor

from ..dataset.types import BatchedTempExample
from ..misc.nn_module_tools import convert_to_buffer
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossLpipsCfg:
    weight: float
    apply_after_step: int


@dataclass
class LossLpipsCfgWrapper:
    lpips: LossLpipsCfg


class LossLpips(Loss[LossLpipsCfg, LossLpipsCfgWrapper]):
    lpips: LPIPS

    def __init__(self, cfg: LossLpipsCfgWrapper) -> None:
        super().__init__(cfg)

        self.lpips = LPIPS(net="vgg")
        convert_to_buffer(self.lpips, persistent=False)

    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedTempExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, " time"]:
        image = batch["target"]["image"]

        # Before the specified step, don't apply the loss.
        if global_step < self.cfg.apply_after_step:
            return torch.tensor(0, dtype=torch.float32, device=image.device)

        gt, ps = pack([image], "* channel height width")
        pred, ps = pack([prediction.color], "* channel height width")
        # --
        loss = self.lpips.forward(gt, pred, normalize=True)[:, 0, 0, 0]
        # --
        [loss] = unpack(loss, ps, "*")  # B, T, V..

        ret_loss = reduce(loss, "b t ... -> t", "mean")
        return self.cfg.weight * ret_loss
