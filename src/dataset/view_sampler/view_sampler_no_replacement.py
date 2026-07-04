from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import torch
from jaxtyping import Float, Int64
from torch import Tensor

from .view_sampler import ViewSampler


@dataclass
class ViewSamplerNoReplacementCfg:
    name: Literal["no_replacement"]
    num_context_views: int
    num_target_views: int
    context_views: list[int] | list[list[int]] | None
    test_context_views: list[int] | list[list[int]] | None
    test_target_views: list[int] | Literal["all", "rest"] | None


class ViewSamplerNoReplacement(ViewSampler[ViewSamplerNoReplacementCfg]):
    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        camera_group_ix: int | None = None,
        device: torch.device = torch.device("cpu"),
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
    ]:
        """Arbtirary sample context and target views without replacement."""
        num_views, _, _ = extrinsics.shape

        assert (
            self.num_context_views + self.num_target_views <= num_views
        ), f"num_context_views + num_target_views must be less than the number of views. Got {self.num_context_views} + {self.num_target_views} > {num_views}."

        assert (
            self.num_context_views != -1
        ), "num_context_views cannot be -1, only possible for num_target_views."

        # 0. For testing, use the specified context and target views.
        if self.stage == "test" and self.cfg.test_context_views is not None:
            if isinstance(self.cfg.test_context_views[0], list):
                context_views = self.cfg.test_context_views
                if camera_group_ix is None:
                    context_views = context_views[np.random.choice(len(context_views))]
                else:
                    context_views = context_views[camera_group_ix % len(context_views)]
                index_context = torch.tensor(context_views, dtype=torch.int64, device=device)

            else:
                index_context = torch.tensor(self.cfg.test_context_views, dtype=torch.int64, device=device)

                # If target views are specified, use them and return.
                # Otherwise take index_context as context views and sample the rest as target views.
            if self.cfg.test_target_views is not None:
                if self.cfg.test_target_views == "all":  # all views are target views
                    index_target = torch.arange(num_views, dtype=torch.int64, device=device)
                elif self.cfg.test_target_views == "rest":  # rest of the views are target views
                    index_target = torch.tensor(
                        [i for i in range(num_views) if i not in self.cfg.test_context_views],
                        dtype=torch.int64,
                        device=device,
                    )
                else:  # specified target views
                    index_target = torch.tensor(self.cfg.test_target_views, dtype=torch.int64, device=device)
                return index_context, index_target
        else:
            context_views = self.cfg.context_views

        # 1. Context views
        rand_order = torch.randperm(num_views, device=device)
        index_context = rand_order[: self.num_context_views]

        # Allow the context views to be fixed.
        # - if more context views are specified, randomly sample from them.
        # - if less context views are specified, use the specified one and sample the rest from the other views.
        if context_views is not None:

            if isinstance(context_views[0], list):
                if camera_group_ix is None:
                    context_views = context_views[np.random.choice(len(context_views))]
                else:
                    context_views = context_views[camera_group_ix % len(context_views)]

            assert len(context_views) >= self.num_context_views
            index_context = torch.tensor(context_views, dtype=torch.int64, device=device)
            if len(context_views) > self.num_context_views:
                rand_order = torch.randperm(len(index_context), device=device)
                index_context = index_context[rand_order[: self.num_context_views]]
            elif len(context_views) < self.num_context_views:
                # never enters here !!
                rand_order = torch.randperm(num_views, device=device)
                rest_views = torch.tensor(
                    [i for i in range(num_views) if i not in index_context.tolist()],
                    dtype=torch.int64,
                    device=device,
                )
                mising_views = self.num_context_views - len(index_context)
                index_context = torch.cat([index_context, rest_views[rand_order[:mising_views]]])

        # 2. Target views
        num_targets = (
            num_views - self.num_context_views
            if self.cfg.num_target_views == -1
            else self.cfg.num_target_views
        )
        rest_views = torch.tensor(
            [i for i in range(num_views) if i not in index_context.tolist()], dtype=torch.int64, device=device
        )
        rand_order = torch.randperm(len(rest_views), device=device)
        index_target = rest_views[rand_order[:num_targets]].detach().clone()

        return index_context, index_target

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views
