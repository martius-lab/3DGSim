import random
from dataclasses import dataclass

# import partial
from functools import partial
from typing import Callable

import numpy as np
import torch
from einops import pack, unpack
from kornia import morphology as morph
from pytorch_lightning import LightningDataModule
from torch import Generator, nn
from torch.utils.data import DataLoader, Dataset, IterableDataset

from ..misc.step_tracker import StepTracker
from . import DatasetCfg, get_dataset
from .types import DataShim, Stage, to_batched_temp_shim
from .validation_wrapper import ValidationWrapper


def dilate(mask: torch.Tensor, kernel: torch.Tensor | None = None) -> torch.Tensor:
    if kernel is None:
        return mask
    mask, ps = pack([mask], "* c h w")
    mask = morph.dilation(mask, kernel)
    mask = unpack(mask, ps, "* c h w")[0]
    return mask


def to_seperate_state_masks(batch, keep_state_mask=False, keep_static_float=False, dilate_masks=True):
    batch["context"]["image"], state_mask = unpack(batch["context"]["image"], [[3], [-1]], "b t v * h w")
    batch["target"]["image"], state_mask_t = unpack(batch["target"]["image"], [[3], [-1]], "b t v * h w")

    kernel = torch.ones(3, 3).to(batch["context"]["image"].device) if dilate_masks else None
    if keep_state_mask:
        if torch.numel(state_mask) > 0:
            # in case state_mask is provided as part of rgba
            batch["context"]["state_mask"] = state_mask > 0.00
            batch["target"]["state_mask"] = state_mask_t > 0.00
        elif "state_mask" in batch["context"]:
            # state_mask is provided as a separate tensor
            batch["context"]["state_mask"] = batch["context"]["state_mask"] > 0.00
            batch["target"]["state_mask"] = batch["target"]["state_mask"] > 0.00

    if keep_static_float and "static_float" in batch["context"]:
        if "static_float" in batch["context"]:
            # 0 if static, 1 if dynamic
            batch["context"]["static_float"] = (
                dilate(batch["context"]["static_float"], kernel) > 0.00
            ).float()
            batch["target"]["static_float"] = (dilate(batch["target"]["static_float"], kernel) > 0.00).float()
    return batch


def get_data_shim(encoder: nn.Module) -> DataShim:
    """Get functions that modify the batch. It's sometimes necessary to modify batches
    outside the data loader because GPU computations are required to modify the batch or
    because the modification depends on something outside the data loader.
    """

    shims: list[DataShim] = [to_batched_temp_shim]
    if hasattr(encoder, "get_data_shim"):
        shims.append(encoder.get_data_shim())

    keep_static_float = not encoder.encoder_info.learn_static_mask
    keep_state_mask = not encoder.encoder_info.learn_background_mask

    shims.append(
        partial(to_seperate_state_masks, keep_state_mask=keep_state_mask, keep_static_float=keep_static_float)
    )

    def combined_shim(batch):
        for shim in shims:
            batch = shim(batch)
        return batch

    return combined_shim


@dataclass
class DataLoaderStageCfg:
    batch_size: int
    num_workers: int
    persistent_workers: bool
    seed: int | None


@dataclass
class DataLoaderCfg:
    dynamic_train_batch_size: bool
    train: DataLoaderStageCfg
    test: DataLoaderStageCfg
    val: DataLoaderStageCfg


DatasetShim = Callable[[Dataset, Stage], Dataset]


def worker_init_fn(worker_id: int) -> None:
    random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))


class DataModule(LightningDataModule):
    dataset_cfg: DatasetCfg
    data_loader_cfg: DataLoaderCfg
    step_tracker: StepTracker | None
    dataset_shim: DatasetShim
    global_rank: int

    def __init__(
        self,
        dataset_cfg: DatasetCfg,
        data_loader_cfg: DataLoaderCfg,
        step_tracker: StepTracker | None = None,
        dataset_shim: DatasetShim = lambda dataset, _: dataset,
        global_rank: int = 0,
    ) -> None:
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.data_loader_cfg = data_loader_cfg
        self.step_tracker = step_tracker
        self.dataset_shim = dataset_shim
        self.global_rank = global_rank
        self.batch_size = data_loader_cfg.train.batch_size

    def get_persistent(self, loader_cfg: DataLoaderStageCfg) -> bool | None:
        return None if loader_cfg.num_workers == 0 else loader_cfg.persistent_workers

    def get_generator(self, loader_cfg: DataLoaderStageCfg) -> torch.Generator | None:
        if loader_cfg.seed is None:
            return None
        generator = Generator()
        generator.manual_seed(loader_cfg.seed + self.global_rank)
        return generator

    def train_dataloader(self):
        dataset = get_dataset(self.dataset_cfg, "train", self.step_tracker)
        dataset = self.dataset_shim(dataset, "train")
        return DataLoader(
            dataset,
            self.batch_size,
            # self.data_loader_cfg.train.batch_size,
            shuffle=not isinstance(dataset, IterableDataset),
            num_workers=self.data_loader_cfg.train.num_workers,
            generator=self.get_generator(self.data_loader_cfg.train),
            worker_init_fn=worker_init_fn,
            persistent_workers=self.get_persistent(self.data_loader_cfg.train),
        )

    def val_dataloader(self):
        dataset = get_dataset(self.dataset_cfg, "val", self.step_tracker)
        dataset = self.dataset_shim(dataset, "val")
        return DataLoader(
            ValidationWrapper(dataset, 1),
            self.data_loader_cfg.val.batch_size,
            num_workers=self.data_loader_cfg.val.num_workers,
            generator=self.get_generator(self.data_loader_cfg.val),
            worker_init_fn=worker_init_fn,
            persistent_workers=self.get_persistent(self.data_loader_cfg.val),
        )

    def test_dataloader(self, dataset_cfg=None):
        dataset = get_dataset(
            self.dataset_cfg if dataset_cfg is None else dataset_cfg,
            "test",
            self.step_tracker,
        )
        dataset = self.dataset_shim(dataset, "test")
        return DataLoader(
            dataset,
            self.data_loader_cfg.test.batch_size,
            num_workers=self.data_loader_cfg.test.num_workers,
            generator=self.get_generator(self.data_loader_cfg.test),
            worker_init_fn=worker_init_fn,
            persistent_workers=self.get_persistent(self.data_loader_cfg.test),
            shuffle=False,
        )

    def check_validity(self):
        from tqdm import tqdm

        train_dl = self.train_dataloader()
        test_dl = self.test_dataloader()
        try:
            for x in tqdm(train_dl, desc="Train"):
                pass
            for x in tqdm(test_dl, desc="Test"):
                pass

        except Exception as e:
            print(f"Error in dataset: {e}")
            raise e
        print("DataModule is valid")
