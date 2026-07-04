from dataclasses import dataclass

from .view_sampler import ViewSamplerCfg


@dataclass(kw_only=True)
class DatasetCfgCommon:
    image_shape: list[int]
    background_color: list[float]
    cameras_are_circular: bool
    overfit_to_scene: str | None
    view_sampler: ViewSamplerCfg

    # dynamics dataset
    n_step_state: int = 1
    n_step_predict: int = 0

    n_step_state: int = 1  # nr of steps for the state
    n_step_state_predict: int = -1  # nr of steps from the step that predict forward (starting from the back)
    n_step_predict_multiple: int = 0  # multiples that each of n_step_state_predict predicts forward in time

    with_mask: bool = False
    with_seg: bool = False
    with_depth: bool = False

    workspace_limits: list[list[float]] | None = None
    test_validity: bool = False

    def __post_init__(self):
        if self.n_step_state_predict == -1:
            self.n_step_state_predict = self.n_step_state

        self.n_step_state_predict = min(self.n_step_state_predict, self.n_step_state)
        self.n_step_predict = self.n_step_state_predict * self.n_step_predict_multiple

        if self.workspace_limits is not None:
            assert len(self.workspace_limits) == 2
            assert len(self.workspace_limits[0]) == 3
            assert len(self.workspace_limits[1]) == 3

        print(
            f"\nDataset:\n",
            f"n_step_state: {self.n_step_state} n_step_state_predict: {self.n_step_state_predict} n_step_predict_multiple: {self.n_step_predict_multiple}\n",
            f"with_mask: {self.with_mask} with_seg: {self.with_seg} with_depth: {self.with_depth}\n",
            f"workspace_limits: {self.workspace_limits}\n",
            f"view_sampler: {self.view_sampler}\n",
        )
