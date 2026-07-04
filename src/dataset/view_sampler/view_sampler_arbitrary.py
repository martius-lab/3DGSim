from dataclasses import dataclass
from typing import Literal

import torch
from jaxtyping import Float, Int64
from torch import Tensor

from .view_sampler import ViewSampler


@dataclass
class ViewSamplerArbitraryCfg:
    name: Literal["arbitrary"]
    num_context_views: int
    num_target_views: int
    context_views: list[int] | None
    target_views: list[int] | None


class ViewSamplerArbitrary(ViewSampler[ViewSamplerArbitraryCfg]):
    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
    ]:
        """Arbitrarily sample context and target views."""
        num_views, _, _ = extrinsics.shape
        assert (
            self.num_context_views != -1
        ), "num_context_views cannot be -1, only possible for num_target_views."

        # 1. Context views
        rand_order = torch.randperm(num_views, device=device)
        index_context = rand_order[: self.num_context_views]

        # Allow the context views to be fixed.
        if self.cfg.context_views is not None:
            assert len(self.cfg.context_views) == self.num_context_views
            index_context = torch.tensor(self.cfg.context_views, dtype=torch.int64, device=device)

        # 2. Target views
        nt = num_views if self.cfg.num_target_views == -1 else self.cfg.num_target_views
        rand_order = torch.randperm(num_views, device=device)
        index_target = rand_order[:nt]

        # Allow the target views to be fixed.
        if self.cfg.target_views is not None:
            assert len(self.cfg.target_views) == nt
            index_target = torch.tensor(self.cfg.target_views, dtype=torch.int64, device=device)

        return index_context, index_target

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views
