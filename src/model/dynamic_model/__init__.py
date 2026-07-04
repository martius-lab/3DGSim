from ...dataset import DatasetCfg
from ..state_adapter import StateInfo
from .constant import DynamicModelConstant, DynamicModelConstantCfg
from .dynamic_model import DynamicModel, StateInfo
from .temporal_ptv3 import DynamicModelPTV3, DynamicModelPTV3Cfg

DYNAMIC_MODELS = {
    "constant": DynamicModelConstant,
    "ptv3": DynamicModelPTV3,
}

DynamicModelCfg = DynamicModelConstantCfg | DynamicModelPTV3Cfg


def get_dynamic_model(
    cfg: DynamicModelCfg,
    dataset_cfg: DatasetCfg,
    state_info: StateInfo,
) -> DynamicModel:
    return DYNAMIC_MODELS[cfg.name](cfg, dataset_cfg, state_info)
