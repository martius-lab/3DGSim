import math
from dataclasses import dataclass
from typing import Literal, Optional

from einops import rearrange
from jaxtyping import Bool, Float
from torch import Tensor

from ...dataset import DatasetCfg
from .backbone import PointTrainsformerV3Cfg, PointTransformerV3
from .dynamic_model import DynamicModel, DynamicModelCfg, StateInfo


@dataclass
class DynamicModelPTV3Cfg(DynamicModelCfg):
    name: Literal["ptv3"]
    model: PointTrainsformerV3Cfg


class DynamicModelPTV3(DynamicModel[DynamicModelPTV3Cfg]):
    def __init__(
        self,
        cfg: DynamicModelPTV3Cfg,
        dataset_cfg: DatasetCfg,
        state_info: StateInfo,
    ) -> None:
        super().__init__(cfg, dataset_cfg, state_info)

        cfg.model.in_channels = self.state_dim()
        cfg.model.out_channels = self.pred_dim() * self.n_step_predict_multiple
        cfg.model.temporal_merger = True
        self.net = PointTransformerV3(cfg=cfg.model)

        self.cfg = cfg

    def forward(
        self,
        xyz: Float[Tensor, "batch time gaussian 3"],
        feature: Float[Tensor, "batch time gaussian d_3_p_feat"],
        state_mask: Optional[Bool[Tensor, "batch time gaussian _"]] = None,
        cond: Optional[dict[str, Tensor]] = None,
    ) -> tuple[Float[Tensor, "batch time_multiple time_pred gaussian d_3_p_feat_incr"], dict[str, Tensor]]:
        out = self.net.step(xyz, feature, state_mask=state_mask)
        if self.n_step_state_predict < self.n_step_state:
            out = out[:, -self.n_step_state_predict :]  # only keep the last n_step_state_predict
        if self.n_step_predict_multiple > 1:
            out = rearrange(out, "b t1 n (t c) -> b t t1 n c", t=self.n_step_predict_multiple)
        else:
            out = out.unsqueeze(1)
        return out, {}


if __name__ == "__main__":

    import torch

    B, Tin, Tout, N, F = 2, 4, 6, 100, 64

    pred_modality = dict(
        means="x",
        covariances="x",
        harmonics="x",
        opacities="x",
        rotations="x",
        scales="x",
    )
    cfg = DynamicModelPTV3Cfg(
        name="ptv3",
        pred_modality=pred_modality,
        model=PointTrainsformerV3Cfg(in_channels=F, out_channels=Tout * F),
    )

    my_dm = DynamicModelPTV3(cfg, DatasetCfg(), {"feat": (3,)})

    in_points_xyz = torch.randn(B, Tin, N, 3)
    in_points_feat = torch.randn(B, Tin, N, F)

    out = my_dm(in_points_xyz, in_points_feat, in_points_feat)
    print(out.shape)
