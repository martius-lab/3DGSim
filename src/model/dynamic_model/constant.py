from dataclasses import dataclass
from typing import Literal, Optional

from einops import repeat
from jaxtyping import Float
from torch import Tensor

from ...dataset import DatasetCfg
from .dynamic_model import DynamicModel, DynamicModelCfg, StateInfo


@dataclass
class DynamicModelConstantCfg(DynamicModelCfg):
    name: Literal["constant"]


class DynamicModelConstant(DynamicModel[DynamicModelConstantCfg]):
    def __init__(
        self,
        cfg: DynamicModelConstantCfg,
        dataset_cfg: DatasetCfg,
        state_info: StateInfo,
    ) -> None:
        super().__init__(cfg, dataset_cfg, state_info)

    def forward(
        self,
        xyz: Float[Tensor, "batch time gaussian 3"],
        feature: Float[Tensor, "batch time gaussian d_3_p_feat"],
        state_mask: Optional[Float[Tensor, "batch time gaussian 1"]] = None,
        cond: Optional[dict[str, Tensor]] = None,
    ) -> tuple[Float[Tensor, "batch time_pred gaussian d_3_p_feat"], dict[str, Tensor]]:
        return repeat(feature[:, -1:], "b t ... -> b (t n_pred) ...", n_pred=self.n_step_predict), {}
