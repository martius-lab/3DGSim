from typing import Optional

from ...dataset import DatasetCfg
from .encoder import Encoder, EncoderInfo, StateInfo
from .encoder_costvolume import EncoderCostVolume, EncoderCostVolumeCfg
from .visualization.encoder_visualizer import EncoderVisualizer
from .visualization.encoder_visualizer_costvolume import EncoderVisualizerCostVolume

ENCODERS = {
    "costvolume": (EncoderCostVolume, EncoderVisualizerCostVolume),
}


EncoderCfg = EncoderCostVolumeCfg


def get_encoder(
    cfg: EncoderCfg,
    dataset_cfg: DatasetCfg,
    state_info: StateInfo,
    encoder_info: EncoderInfo,
) -> tuple[Encoder, Optional[EncoderVisualizer]]:
    encoder, visualizer = ENCODERS[cfg.name]
    encoder = encoder(cfg, dataset_cfg, state_info, encoder_info)
    if visualizer is not None:
        visualizer = visualizer(cfg.visualizer, encoder)
    return encoder, visualizer
