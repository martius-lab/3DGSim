import json

# import math
from dataclasses import dataclass, field

# from functools import cached_property
# from io import BytesIO
from pathlib import Path
from typing import Literal

import imageio
import numpy as np
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat  # noqa
from jaxtyping import Float  # , UInt8
from PIL import Image
from scipy.spatial.transform import Rotation
from torch import Tensor
from torch.utils.data import Dataset

# from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler


@dataclass
class DatasetMOVICfg(DatasetCfgCommon):
    name: Literal["movi"]
    roots: list[Path]

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

    keys: dict[str, dict[str, str]] = field(
        default_factory=lambda: {
            "image": {
                "folder_name": "rgba",
                "name": "rgba",
                "ext": "png",
            },
            "depth": {
                "folder_name": "depth",
                "name": "depth",
                "ext": "tiff",
            },
            "cam_param": {
                "folder_name": "--",
                "name": "metadata",
                "ext": "json",
            },
            "static_float": {
                "folder_name": "segmentation",
                "name": "segmentation",
                "ext": "png",
            },
            # "seg_label": {
            #     "folder_name": "semantic_segmentation",
            #     "name": "semantic_segmentation_labels",
            #     "ext": "json",
            # },
        }
    )


def format_camera_information(metadata):
    # In the default case, the positive Y values in the camera coordinate system point upwards, positive
    # Z values point backwards from the scene into the camera, positive X values point leftwards.
    R = np.eye(3)
    R[1, 1] = -1
    R[2, 2] = -1
    positions = np.array(metadata["camera"]["positions"], np.float32)
    # Position/Quaternion([w, x, y, z]) of the camera for each frame in world-coordinates.
    # In the default case,
    # positive Y values in the camera coordinate system point upwards,
    # positive Z values point backwards from the scene into the camera,
    # positive X values point leftwards.
    # According to:
    # https://github.com/google-research/kubric/blob/main/challenges/movi/README.md#:~:text=%22-,camera,-%22%20This%20key
    # and https://kieranwynn.github.io/pyquaternion/#from-a-numpy-array
    rotations = Rotation.from_quat(
        np.array(metadata["camera"]["quaternions"], np.float32), scalar_first=True
    ).as_matrix()
    N = len(positions)
    c2w = np.empty((N, 4, 4))
    c2w[:] = np.eye(4)
    c2w[:, :3, :3] = rotations @ R
    c2w[:, :3, -1] = positions
    # c2w[:, 3, :] = [0, 0, 0, 1]

    w2c = c2w  # lol so the camera's pose  in world frame is required
    # w2c = np.linalg.inv(c2w)

    K = np.array(metadata["camera"]["K"], np.float32)
    # K[0, 0] *= 1.0 / metadata["camera"]["sensor_width"]  # * metadata["metadata"]["resolution"][1]
    # K[1, 1] *= 1.0 / metadata["camera"]["sensor_width"]  # * metadata["metadata"]["resolution"][0]

    return {
        "K": K @ R,
        # "R": np.array(metadata["camera"]["R"], np.float32),
        "focal_length": metadata["camera"]["focal_length"],
        "sensor_width": metadata["camera"]["sensor_width"],
        "field_of_view": metadata["camera"]["field_of_view"],
        # "positions": positions,
        # "rotations": rotations,
        "height": metadata["metadata"]["resolution"][0],
        "width": metadata["metadata"]["resolution"][1],
        "w2cs": w2c,
    }


class DatasetMOVI(Dataset):
    cfg: DatasetMOVICfg
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
        cfg: DatasetMOVICfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg

        self.image_keys = ["image"]
        if cfg.with_depth:
            self.image_keys.append("depth")
        if cfg.with_seg:
            # self.cfg.with_mask = True
            self.image_keys.append("static_float")

        self.validation = stage in ["val", "test"]

        self.n_step_state = cfg.n_step_state
        self.stage = stage if not cfg.overfit_to_scene else "train"
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        # NOTE: update near & far; remember to DISABLE `apply_bounds_shim` in encoder
        if cfg.near != -1:
            self.near = cfg.near
        if cfg.far != -1:
            self.far = cfg.far

        # Collect chunks.
        scene_paths = []

        if self.cfg.overfit_to_scene is not None and Path(self.cfg.overfit_to_scene).exists():
            print(f"Overfitting to scene: {self.cfg.overfit_to_scene}")
            scene_paths = [Path(self.cfg.overfit_to_scene)]  # overfit to a single scene
        else:
            for root in cfg.roots:
                _scene_paths = sorted(
                    list(Path(root).glob("episode*")),
                    key=lambda x: int(x.name.split("_")[-1]),
                )
                assert len(_scene_paths) > 0, f"No scene found in {root}"
                scene_paths.extend(_scene_paths)

            # Overfit to the first scene
            if self.cfg.overfit_to_scene is not None:
                scene_paths = [scene_paths[0]]  # choose only one scene

        # Camera setup is the same -> so, take them from first scene
        self.cams = [str(cam.relative_to(scene_paths[0])) for cam in scene_paths[0].glob("cam*")]
        self.cams = sorted(self.cams, key=lambda x: int(x.split("_")[-1]))

        # Collect the traj paths: list of list of paths: N_scene x N_frame
        scene_templates = []
        cam = self.cams[0]  # take the first camera
        for root in scene_paths:
            traj_paths = list((root / cam / "rgba").glob("*.png"))
            # sort traj_paths by time
            traj_paths = sorted(traj_paths, key=lambda x: int(x.name.split("_")[-1].split(".")[0]))

            traj_paths = traj_paths[:: cfg.speedup]

            # Collect the traj
            templates = [self._extract_template(traj, cam) for traj in traj_paths]
            templates = sorted(templates, key=lambda x: int(x.split("_")[-1].split(".")[0]))
            scene_templates.append(templates)

        # Split the data
        test_ratio = 0.12
        self.scene_templates = get_split(scene_templates, test_ratio, self.stage, cfg.test_chunk_interval)

        # Set the predict length
        self.n_scenes = len(self.scene_templates)
        self.n_frames_per_scene = len(self.scene_templates[0])
        self.set_n_step_predict(cfg.n_step_predict)

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

    def _extract_template(self, path: Path, cam: str) -> str:
        _path = str(path).replace(cam, "{cam}")
        _path = _path.replace("rgba", "{folder_name}", 1)
        _path = _path.replace("rgba", "{name}")
        _path = _path.replace(".png", ".{ext}")
        return _path

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def load_multicam_param(self, path: str):
        # return torch.Size([CAM, 3,3]), torch.Size([CAM, 4,4])
        # 18: [fx, fy, cx, cy, 0., 0., w2c[:3, :].flatten()]

        intrinsics = []
        extrinsics = []
        for cam in self.cams:
            _path = path.format(cam=cam, **self.cfg.keys["cam_param"])
            time_step = int(_path.split("/")[-1].split("_")[1].split(".")[0])

            metadata = json.load(
                open(
                    _path.split(self.cfg.keys["cam_param"]["folder_name"])[0] + "metadata.json",
                )
            )
            cam_param = format_camera_information(metadata)

            # INTRENSICS
            intrinsics.append(cam_param["K"])

            # EXTRENSICS
            extrinsics.append(cam_param["w2cs"][time_step])

        intrinsics = torch.tensor(np.array(intrinsics), dtype=torch.float32)
        extrinsics = torch.tensor(np.array(extrinsics), dtype=torch.float32)

        return intrinsics, extrinsics

    def load_chunk(self, chunk_paths: list["str"]) -> list[dict]:
        """_summary_
        returns a list of multiview data:
        [
            {
                "path": str,
                "time_stamp": int,
                "cam_param": Float[Tensor, "18"],
                "rgb": list[UInt8[Tensor, "..."]],
                "other": ...

            },
            ...
            {
                ..
            }
        ]  # n_step_full


        """
        chunks = []
        im_type = "RGBA" if self.cfg.with_mask else "RGB"
        for i, path in enumerate(chunk_paths):
            curr_data = dict(path=path.format(cam=self.cams[0], **self.cfg.keys["image"]), time_stamp=i)
            for key in self.cfg.keys.keys():
                if key == "image":
                    data = [
                        Image.open(path.format(cam=cam, **self.cfg.keys[key])).convert(im_type)
                        for cam in self.cams
                    ]
                    curr_data[key] = data
                elif key == "cam_param":
                    intrinsics, extrinsics = self.load_multicam_param(path)
                    curr_data["extrinsics"] = extrinsics
                    curr_data["intrinsics"] = intrinsics
                elif key == "depth":
                    if self.cfg.with_depth:
                        data = [
                            imageio.v2.imread(path.format(cam=cam, **self.cfg.keys[key]), format="tiff")
                            for cam in self.cams
                        ]
                        curr_data[key] = data
                elif key == "static_float":
                    if self.cfg.with_seg:
                        data = [
                            Image.open(path.format(cam=cam, **self.cfg.keys[key])).convert("L")
                            for cam in self.cams
                        ]
                        curr_data[key] = data
                else:
                    raise ValueError(f"Invalid key {key}")

            chunks.append(curr_data)
        return chunks

    def _get_scene_name(self, chunk) -> str:
        path0 = chunk[0]["path"]
        path1 = chunk[-1]["path"]
        episode = [p for p in path0.split("/") if "episode" in p][0]
        st = f"{path0.split('/')[-1].split('_')[1].split('.')[0]}"
        end = f"{path1.split('/')[-1].split('_')[1].split('.')[0]}"
        scene = f"{episode}_{st}:{end}"
        return scene

    def convert_images(
        self,
        images: list,
    ) -> Float[Tensor, "batch _ height width"]:
        torch_images = []
        for image in images:
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    def __getitem__(self, index):
        # a chunk is a list of templates to different timesteps of a scene
        scene_idx = index // self.valid_num_steps
        step_idx = index % self.valid_num_steps
        chunk_templates = self.scene_templates[scene_idx][step_idx : step_idx + self.n_step_full]

        chunk = self.load_chunk(chunk_templates)
        contexts = []
        targets = []
        scene = self._get_scene_name(chunk)
        for t, frame in enumerate(chunk):
            # an example is a multi-view image-camera-etc data
            extrinsics, intrinsics = frame["extrinsics"], frame["intrinsics"]
            context_indices, target_indices = self.view_sampler.sample(
                scene,
                extrinsics,
                intrinsics,
            )
            # context_indices = np.array([0, 1])
            # target_indices = np.array([2, 3])

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
                    k: self.convert_images([frame[k][index.item()] for index in context_indices])
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
                k: self.convert_images([frame[k][index.item()] for index in target_indices])
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
        examples = {"scene": scene, "context": contexts, "target": targets}

        if self.stage == "train" and self.cfg.augment:
            examples = apply_augmentation_shim(examples)
        examples = apply_crop_shim(examples, tuple(self.cfg.image_shape))

        return examples


def get_split(scene_templates: list[list[str]], test_ratio: float, stage: str, test_every_nth: int):
    n_paths = len(scene_templates)
    N_test = 1 if n_paths < 15 else int(test_ratio * n_paths)
    every_nth = int(n_paths / N_test)
    test_idxs = list(range(0, n_paths, every_nth))

    if stage == "train":
        final_scene_templates = [
            scene_templates[i] for i in range(n_paths) if i not in test_idxs or len(test_idxs) < 15
        ]
    elif test_every_nth == -1:
        print(f"All scenes are used for {stage}")
        final_scene_templates = scene_templates
    else:
        # test and val
        final_scene_templates = [scene_templates[i] for i in test_idxs]
        if stage == "test":
            # NOTE: hack to skip some chunks in testing during training, but the index
            # is not change, this should not cause any problem except for the display
            final_scene_templates = final_scene_templates[::test_every_nth]
    print(f"stage: {stage} -> scene templates: {len(final_scene_templates)}")

    return final_scene_templates
