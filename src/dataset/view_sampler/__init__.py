from typing import Any

from ...misc.step_tracker import StepTracker
from ..types import Stage
from .view_sampler import ViewSampler
from .view_sampler_all import ViewSamplerAll, ViewSamplerAllCfg
from .view_sampler_arbitrary import ViewSamplerArbitrary, ViewSamplerArbitraryCfg
from .view_sampler_bounded import ViewSamplerBounded, ViewSamplerBoundedCfg
from .view_sampler_bounded_v2 import ViewSamplerBoundedV2, ViewSamplerBoundedV2Cfg
from .view_sampler_evaluation import ViewSamplerEvaluation, ViewSamplerEvaluationCfg
from .view_sampler_no_replacement import ViewSamplerNoReplacement, ViewSamplerNoReplacementCfg

VIEW_SAMPLERS: dict[str, ViewSampler[Any]] = {
    "all": ViewSamplerAll,
    "arbitrary": ViewSamplerArbitrary,
    "bounded": ViewSamplerBounded,
    "evaluation": ViewSamplerEvaluation,
    "no_replacement": ViewSamplerNoReplacement,
    "boundedv2": ViewSamplerBoundedV2,
}

ViewSamplerCfg = (
    ViewSamplerArbitraryCfg
    | ViewSamplerBoundedCfg
    | ViewSamplerEvaluationCfg
    | ViewSamplerAllCfg
    | ViewSamplerNoReplacementCfg
    | ViewSamplerBoundedV2Cfg
)


def get_view_sampler(
    cfg: ViewSamplerCfg,
    stage: Stage,
    overfit: bool,
    cameras_are_circular: bool,
    step_tracker: StepTracker | None,
) -> ViewSampler[Any]:
    return VIEW_SAMPLERS[cfg.name](
        cfg,
        stage,
        overfit,
        cameras_are_circular,
        step_tracker,
    )
