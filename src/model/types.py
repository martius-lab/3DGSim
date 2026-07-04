import os
from dataclasses import dataclass, fields
from typing import Generator, Literal, Optional, TypedDict

import einops as eo
import numpy as np
import torch
from jaxtyping import Bool, Float
from torch import Tensor

from ..geometry.rotation_conversions import build_covariance


def inverse_sigmoid(x):
    return np.log(x / (1 - x))


class SpatialLatentsDict(TypedDict, total=False):
    means: Float[Tensor, "*batch spatial dim"]
    # 3dgs
    rotations: Optional[Float[Tensor, "*batch spatial d_rot"]]
    scales: Optional[Float[Tensor, "*batch spatial 3"]]
    opacities: Optional[Float[Tensor, "*batch spatial 1"]]
    harmonics: Optional[Float[Tensor, "*batch spatial 3 d_sh"]]
    # latent features
    features: Optional[Float[Tensor, "*batch spatial features"]]
    # masks
    state_mask: Optional[Bool[Tensor, "*batch spatial 1"]]
    static_float: Optional[Float[Tensor, "*batch spatial 1"]]
    background_float: Optional[Float[Tensor, "*batch spatial 1"]]


@dataclass
class Gaussians:
    means: Float[Tensor, "*batch gaussian dim"]
    harmonics: Float[Tensor, "*batch gaussian 3 d_sh"]
    opacities: Float[Tensor, "*batch gaussian 1"]
    scales: Optional[Float[Tensor, "*batch gaussian 3"]]
    rotations: Optional[Float[Tensor, "*batch gaussian 4"]]

    covariances: Optional[Float[Tensor, "*batch gaussian dim dim"]] = None

    state_mask: Optional[Bool[Tensor, "*batch gaussian 1"]] = None
    static_float: Optional[Float[Tensor, "*batch gaussian 1"]] = None
    background_float: Optional[Float[Tensor, "*batch gaussian 1"]] = None

    def get_type(self):
        batch_dims = self._batch_dims()
        if batch_dims == 2:
            temp = "BatchTemp"
        elif batch_dims == 1:
            temp = "Batched"
        elif batch_dims == 0:
            temp = "UnBatched"
        else:
            temp = f"Batch_{batch_dims}D"
        return temp

    def _batch_dims(self):
        return self.means.dim() - 2

    def get_covariances(self):
        if self.covariances is None:
            self.covariances = build_covariance(self.scales, self.rotations, normalize=True, eps=1e-8)
        return self.covariances

    def __getitem__(self, item):
        if isinstance(item, int):
            if self._batch_dims() == 0:
                raise ValueError("Invalid attempt to index UnBatched Gaussians")

        return Gaussians(**{k: v[item] for k, v in self.items()})

    def __str__(self):
        return f"{self.get_type()}_Gaussian(means={self.means.shape}, rotations={self.rotations.shape}, scales={self.scales.shape}, harmonics={self.harmonics.shape}, opacities={self.opacities.shape})"

    def extras(self):
        for k in ["state_mask", "static_float", "background_float"]:
            if getattr(self, k) is not None:
                yield (k, getattr(self, k))

    def items(self) -> Generator[tuple[str, Tensor], None, None]:
        for field in fields(self):
            if getattr(self, field.name) is not None:
                yield (field.name, getattr(self, field.name))

    def to_batched_gaussians(self) -> "Gaussians":
        if self._batch_dims() == 1:
            return self

        batch_dims = self._batch_dims()
        return Gaussians(
            **{k: v.flatten(start_dim=0, end_dim=batch_dims - 1) for k, v in self.items()},
        )

    def unpack_batch(self, ps: list[eo.packing.Shape]) -> "Gaussians":
        if self._batch_dims() == 1:
            return self

        if len(ps[0]) == 1:
            ps = [ps[0] + [1]]

        batch_dims = self._batch_dims()
        get_shape = lambda x: " ".join([f"dim_{i}" for i in range(x.dim() - batch_dims)])
        return Gaussians(
            **{k: eo.unpack(v, ps, f"* {get_shape(v)}")[0] for k, v in self.items()},
        )

    def save(self, path: str):
        if self._batch_dims() > 2:
            raise ValueError("Can only save  up to batched-temporal Gaussians")
        elif self._batch_dims() == 2:
            for b in range(self.means.shape[0]):
                self[b].save(f"{path.split('.ply')[0]}/b_{b}")
        elif self._batch_dims() == 1:
            for t in range(self.means.shape[0]):
                self[t].save(f"{path.split('.ply')[0]}/t_{t:03d}")
        elif self._batch_dims() == 0:

            if self.state_mask is not None:
                gs = Gaussians(**{k: v[self.state_mask.squeeze(-1)] for k, v in self.items()})
            else:
                gs = self
            os.makedirs(os.path.dirname(path), exist_ok=True)
            save_ply(gs, path if path.endswith(".ply") else f"{path}.ply")


def construct_list_of_attributes(gs: Gaussians):
    l = ["x", "y", "z", "nx", "ny", "nz"]
    features_dc = gs.harmonics[:, :1]
    features_rest = gs.harmonics[:, 1:]
    for i in range(features_dc.shape[1] * features_dc.shape[2]):
        l.append("f_dc_{}".format(i))
    for i in range(features_rest.shape[1] * features_rest.shape[2]):
        l.append("f_rest_{}".format(i))
    l.append("opacity")
    for i in range(gs.scales.shape[1]):
        l.append("scale_{}".format(i))
    for i in range(gs.rotations.shape[1]):
        l.append("rot_{}".format(i))
    return l


def construct_list_of_attributes(num_rest: int) -> list[str]:
    attributes = ["x", "y", "z", "nx", "ny", "nz"]
    for i in range(3):
        attributes.append(f"f_dc_{i}")
    for i in range(3 * num_rest):
        attributes.append(f"f_rest_{i}")
    attributes.append("opacity")
    for i in range(3):
        attributes.append(f"scale_{i}")
    for i in range(4):
        attributes.append(f"rot_{i}")
    return attributes


def save_ply(gs: Gaussians, path):
    assert gs._batch_dims() == 0, "Can only save unbatched Gaussians"

    from plyfile import PlyData, PlyElement

    xyz = gs.means.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    opacities = inverse_sigmoid(torch.clamp(gs.opacities, 1e-3, 1 - 1e-3).detach().cpu().numpy())
    scale = np.log(gs.scales.detach().cpu().numpy())

    x, y, z, w = eo.rearrange(gs.rotations, "g xyzw -> xyzw g")
    rotation = torch.stack((w, x, y, z), dim=-1).detach().cpu().numpy()  # convert from xyzw to wxyz?

    harmonics = gs.harmonics.transpose(-1, -2)  # G, 3, d_sh -> G, d_sh, 3
    features_dc, features_rest = harmonics[:, :1], harmonics[:, 1:]
    f_dc = features_dc.detach().flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = features_rest.detach().flatten(start_dim=1).contiguous().cpu().numpy()

    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes(harmonics.shape[1] - 1)]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate(
        (xyz, normals, f_dc, torch.zeros_like(f_rest), opacities, scale, rotation), axis=1
    )
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    PlyData([el]).write(path)
