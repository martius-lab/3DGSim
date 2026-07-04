from .loss import Loss
from .loss_depth import LossDepth, LossDepthCfgWrapper
from .loss_lpips import LossLpips, LossLpipsCfgWrapper
from .loss_mse import LossMse, LossMseCfgWrapper

LOSSES = {
    LossDepthCfgWrapper: LossDepth,
    LossLpipsCfgWrapper: LossLpips,
    LossMseCfgWrapper: LossMse,
}

LossCfgWrapper = LossDepthCfgWrapper | LossLpipsCfgWrapper | LossMseCfgWrapper


def get_losses(cfgs: list[LossCfgWrapper]) -> list[Loss]:
    losses = [LOSSES[type(cfg)](cfg) for cfg in cfgs]

    print("\nLosses:")
    for loss, c in zip(losses, cfgs):
        print(f" {loss.__class__.__name__}, loss_weight: {loss.cfg.weight}")
    return losses
