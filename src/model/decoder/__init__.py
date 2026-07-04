from ...dataset import DatasetCfg
from ..state_adapter import DecoderInfo
from .decoder import Decoder
from .decoder_mvsplat import DecoderMVSplat, DecoderMVSplatCfg

DECODERS = {
    "mvsplat": DecoderMVSplat,
}

DecoderCfg = DecoderMVSplatCfg


def get_decoder(decoder_cfg: DecoderCfg, dataset_cfg: DatasetCfg, decoder_info: DecoderInfo) -> Decoder:
    return DECODERS[decoder_cfg.name](decoder_cfg, dataset_cfg, decoder_info)
