from dataclasses import dataclass

from einops import reduce
from jaxtyping import Float
from torch import Tensor

from ..dataset.types import BatchedTempExample
from ..model.decoder.decoder import DecoderOutput
from ..model.types import Gaussians
from .loss import Loss


@dataclass
class LossMseCfg:
    weight: float


@dataclass
class LossMseCfgWrapper:
    mse: LossMseCfg


class LossMse(Loss[LossMseCfg, LossMseCfgWrapper]):
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedTempExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, " time"]:
        delta = prediction.color - batch["target"]["image"]

        return self.cfg.weight * reduce(delta**2, "b t ... -> t", "mean")
