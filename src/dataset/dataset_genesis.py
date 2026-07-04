import json
from dataclasses import dataclass
from itertools import groupby

# from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat  # noqa
from jaxtyping import Float  # , UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

# from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler


@dataclass
class DatasetGenesisCfg(DatasetCfgCommon):
    name: Literal["genesis"]
    roots: list[Path]
    root_depths: list[int]  # [int]

    augment: bool

    # n_step_state: 1  # nr of steps for the state
    # n_step_predict: 0  # nr of steps for the prediction (n_step_predict_multiple * n_step_state_predict)
    # n_step_state_predict: 1  # nr of steps from the step that predict forward (starting from the back)
    # n_step_predict_multiple:  1 # multiples that each of n_step_state_predict predicts forward in time

    test_chunk_interval: int  # skip some chunks in testing during training
    test_times_per_scene: int  # how many times to sample each scene (max 4 for omni ds)
    test_len: int
    shuffle_val: bool = True
    speedup: int = 1

    # scale the world to make the baseline 1
    make_baseline_1: bool = False
    baseline_scale_bounds: bool = False

    baseline_epsilon: float = 1e-3
    near: float = -1.0
    far: float = -1.0

    camera_ixs_allowed: list[int] | None = None


def get_list_of_scene_configs(root, root_depth=0):
    config_paths = list(Path(root).glob("*/" * root_depth + "metadata.json"))
    scene_paths = [p.parent for p in config_paths]
    list_of_scene_configs = [json.load(open(p, "r")) for p in config_paths]

    # file_path is absolute path to image
    for scene_path, scene_config in zip(scene_paths, list_of_scene_configs):
        scene_config["file_path_template"] = str(Path(scene_path) / scene_config["file_path_template"])
        scene_config["scene_name"] = str(scene_path).split("/")[-1] + "_" + str(scene_path).split("/")[-1]

    return list_of_scene_configs

    # # test: laoad metadata and go through all filepaths and load the image and assert their shapes
    # metadata = file_io.read_json("output/metadata.json")
    # file_path_template = "output/" + metadata["file_path_template"]
    # n_cams = metadata["n_cams"]
    # n_stps = metadata["n_steps"]
    # for i in range(n_stps):
    #     for j in range(n_cams):
    #         for k, img_type in enumerate(["rgb", "seg"]):
    #             img_path = file_path_template.format(cam_ix=j, img_type=img_type, n_step=i) + ".png"
    #             img = file_io.read_png(img_path)
    #             print(img_type, f"step_{i}", f"cam_{j}", img.shape, img_path)


class DatasetGenesis(Dataset):
    cfg: DatasetGenesisCfg
    stage: Stage
    view_sampler: ViewSampler

    n_step_full: int
    cams: list[str]

    to_tensor: tf.ToTensor
    scene_templates: list[list[str]]
    chunks: list[Path]
    near: float = 0.1
    far: float = 1000.0

    def __init__(
        self,
        cfg: DatasetGenesisCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg

        self.image_keys = ["image"]
        if cfg.with_mask:
            self.image_keys.append("state_mask")
        if cfg.with_seg:
            self.image_keys.append("static_float")

        self.validation = stage in ["val", "test"]

        self.n_step_state = cfg.n_step_state
        self.stage = stage if not cfg.overfit_to_scene else "train"
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        self.to_tensor_unscaled = tf.PILToTensor()
        # NOTE: update near & far; remember to DISABLE `apply_bounds_shim` in encoder
        if cfg.near != -1:
            self.near = cfg.near
        if cfg.far != -1:
            self.far = cfg.far

        # Collect chunks.
        roots = []
        if self.cfg.overfit_to_scene is not None and Path(self.cfg.overfit_to_scene).exists():
            print(f"Overfitting to scene: {self.cfg.overfit_to_scene}")
            # overfit to a single scene
            list_of_scene_configs = get_list_of_scene_configs(Path(self.cfg.overfit_to_scene))
        else:
            list_of_scene_configs = []
            for root, root_depth in zip(cfg.roots, cfg.root_depths):
                list_of_scene_configs += get_list_of_scene_configs(root, root_depth)

            assert len(list_of_scene_configs) > 0, f"No scene found in {root}"

            # Overfit to the first scene
            if self.cfg.overfit_to_scene is not None:
                list_of_scene_configs = [list_of_scene_configs[0]]  # choose only one scene

        # make sure all scenes have same nr of steps and cams
        # only a check, can be outcommented
        n_frames_per_scene = list_of_scene_configs[0]["n_steps"]
        num_cams = list_of_scene_configs[0]["n_cams"]
        for scene in list_of_scene_configs:
            assert scene["n_steps"] == n_frames_per_scene
            assert scene["n_cams"] == num_cams

            # check if all steps have same c2w and intrinsic (static cameras)
            for j in range(num_cams):
                assert np.allclose(
                    list_of_scene_configs[0][f"cam_{j}"]["extrinsics"], scene[f"cam_{j}"]["extrinsics"]
                )
                assert np.allclose(
                    list_of_scene_configs[0][f"cam_{j}"]["intrinsics"], scene[f"cam_{j}"]["intrinsics"]
                )

        # Split the data
        test_ratio = 0.12
        self.list_of_scene_configs = get_split(
            list_of_scene_configs, test_ratio, self.stage, self.cfg.test_chunk_interval
        )

        # Set the predict length
        self.num_cams = num_cams
        self.cams = list(range(num_cams))
        if self.cfg.camera_ixs_allowed is not None:
            self.num_cams = len(self.cfg.camera_ixs_allowed)
            self.cams = self.cfg.camera_ixs_allowed

        self.cam_ixs = {i: cam for i, cam in enumerate(self.cams)}

        self.n_scenes = len(self.list_of_scene_configs)
        self.n_frames_per_scene = n_frames_per_scene
        self.set_n_step_predict(cfg.n_step_predict)
        if self.n_step_full > self.n_frames_per_scene:
            raise ValueError(
                f"n_step_full: {self.n_step_full} > n_frames_per_scene: {self.n_frames_per_scene}"
            )
        print(f"Dytaset has length: {len(self)}")

    def set_n_step_predict(self, n_step_predict: int):
        # sets: n_step_predict, n_step_full
        if self.validation and n_step_predict != 0:
            n_step_predict = self.n_frames_per_scene - self.n_step_state
        else:
            n_step_predict = n_step_predict
        self.n_step_full = self.n_step_state + n_step_predict
        self.valid_num_steps = self.n_frames_per_scene - self.n_step_state - n_step_predict + 1

    def __len__(self):
        return self.n_scenes * self.valid_num_steps

    def convert_images(
        self,
        images: list,
        k: str = "image",
    ) -> Float[Tensor, "batch _ height width"]:

        to_tensor = self.to_tensor if k == "image" else self.to_tensor_unscaled
        torch_images = []
        for image in images:
            torch_images.append(to_tensor(image))
        torch_images = torch.stack(torch_images)

        if 0 not in set(torch_images.flatten().tolist()):
            torch_images -= 1

        if k == "state_mask":
            torch_images = (torch_images > 0).float()
        elif k == "static_float":
            torch_images = (torch_images == 3).float()

        return torch_images

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    def get_chunk(self, scene_idx, step_idx, n_steps):
        # load image paths
        # loads cam internsic and extrinsic params
        # 1. Normalize intrinsics
        def get_normalized_intrinsics(intrinsics, width, height):
            # intrinsics: 3x3 list of lists
            intrinsics = np.array(intrinsics.copy())
            intrinsics[0, 0] /= width
            intrinsics[1, 1] /= height
            intrinsics[0, 2] /= width
            intrinsics[1, 2] /= height

            return intrinsics

        def get_w2c(c2w):
            # extrinsics: 4x4 list of lists
            return np.linalg.inv(np.array(c2w.copy(), dtype=np.float32))

        scene_config = self.list_of_scene_configs[scene_idx]

        # load the chunk
        scene_name = f"{scene_config['scene_name']}_{step_idx}:{step_idx+n_steps}"

        w2cs = [get_w2c(scene_config[f"cam_{j}"]["extrinsics"]) for j in self.cams]
        K = [
            get_normalized_intrinsics(
                scene_config[f"cam_{j}"]["intrinsics"], *scene_config[f"cam_{j}"]["resolution"]
            )
            for j in self.cams
        ]

        chunk_cfgs = []
        file_path_template = scene_config["file_path_template"]
        for step_i in range(step_idx, step_idx + n_steps):
            rgbs = [
                Image.open(file_path_template.format(cam_ix=j, n_step=step_i, img_type="rgb") + ".png")
                for j in self.cams
            ]
            mydict = dict(extrinsics=w2cs, intrinsics=K, image=rgbs)

            if self.cfg.with_mask:
                mydict["state_mask"] = [
                    Image.open(file_path_template.format(cam_ix=j, n_step=step_i, img_type="seg") + ".png")
                    for j in self.cams
                ]
            if self.cfg.with_seg:
                mydict["static_float"] = [
                    Image.open(file_path_template.format(cam_ix=j, n_step=step_i, img_type="seg") + ".png")
                    for j in self.cams
                ]

            chunk_cfgs.append(mydict)

        return scene_name, chunk_cfgs

    def __getitem__(self, index):
        # a chunk is a list of templates to different timesteps of a scene
        scene_idx = index // self.valid_num_steps
        step_idx = index % self.valid_num_steps

        scene_name, chunk_cfgs = self.get_chunk(scene_idx, step_idx, self.n_step_full)
        contexts = []
        targets = []
        for t, frame in enumerate(chunk_cfgs):
            # an example is a multi-view image-camera-etc data
            extrinsics = torch.tensor(np.array(frame["extrinsics"]), dtype=torch.float32)
            intrinsics = torch.tensor(np.array(frame["intrinsics"]), dtype=torch.float32)
            context_indices, target_indices = self.view_sampler.sample(
                scene_name,
                extrinsics,
                intrinsics,
                camera_group_ix=t,
            )

            # Resize the world to make the baseline 1. (TODO: off for now)
            if self.cfg.make_baseline_1:
                scale = extrinsics[:, :3, 3].abs().max()
                if scale < self.cfg.baseline_epsilon:
                    pass  # skip
                else:
                    extrinsics[:, :3, 3] /= scale
            else:
                scale = 1
            nf_scale = scale if self.cfg.baseline_scale_bounds else 1.0

            # Load images and prepare the context and target data.
            if t < self.cfg.n_step_state:
                # No context for the future.
                context_imagges = {
                    k: self.convert_images([frame[k][index.item()] for index in context_indices], k)
                    for k in self.image_keys
                }

                contexts.append(
                    {
                        "extrinsics": extrinsics[context_indices],
                        "intrinsics": intrinsics[context_indices],
                        "near": self.get_bound("near", len(context_indices)) / nf_scale,
                        "far": self.get_bound("far", len(context_indices)) / nf_scale,
                        "index": context_indices,
                        **context_imagges,
                    }
                )
            # During testing, we need to keep the target indices the same (view consistency).
            if self.validation and t > 0:
                target_indices = targets[-1]["index"]

            target_imagges = {
                k: self.convert_images([frame[k][index.item()] for index in target_indices], k)
                for k in self.image_keys
            }
            targets.append(
                {
                    "extrinsics": extrinsics[target_indices],
                    "intrinsics": intrinsics[target_indices],
                    "near": self.get_bound("near", len(target_indices)) / nf_scale,
                    "far": self.get_bound("far", len(target_indices)) / nf_scale,
                    "index": target_indices,
                    **target_imagges,
                }
            )

        # Stack diferent timesteps
        # contexts["image"]: T_in, View C, H, W
        # targets["image"]: T_in+T_out, View, C, H, W
        keys = contexts[0].keys()
        targets = {key: torch.stack([target[key] for target in targets]) for key in keys}
        contexts = {key: torch.stack([context[key] for context in contexts]) for key in keys}
        examples = {"scene": scene_name, "context": contexts, "target": targets}

        if self.stage == "train" and self.cfg.augment:
            examples = apply_augmentation_shim(examples)
        examples = apply_crop_shim(examples, tuple(self.cfg.image_shape))

        return examples


def get_split(list_of_scene_configs: list[dict], test_ratio: float, stage: str, test_every_nth: int):
    n_scenes = len(list_of_scene_configs)

    N_test = 1 if n_scenes < 15 else int(test_ratio * n_scenes)
    every_nth = int(n_scenes / N_test)
    test_idxs = list(range(0, n_scenes, every_nth))

    if stage == "train":
        final_list_of_scene_configs = [
            list_of_scene_configs[i] for i in range(n_scenes) if i not in test_idxs or len(test_idxs) < 15
        ]
    elif test_every_nth == -1:
        print(f"All scenes are used for {stage}")
        final_list_of_scene_configs = list_of_scene_configs
    else:
        # test and val
        final_list_of_scene_configs = [list_of_scene_configs[i] for i in test_idxs]
        if stage == "test":
            # NOTE: hack to skip some chunks in testing during training, but the index
            # is not change, this should not cause any problem except for the display
            final_list_of_scene_configs = final_list_of_scene_configs[::test_every_nth]
    print(f"stage: {stage} -> scene templates: {len(final_list_of_scene_configs)}")

    return final_list_of_scene_configs
