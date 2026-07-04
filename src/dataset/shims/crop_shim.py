import numpy as np
import torch
import torchvision.transforms as tf
from jaxtyping import Float
from torch import Tensor

from ..types import AnyExample, AnyViews

# from PIL import Image
# def rescale(
#     image: Float[Tensor, "_ h_in w_in"],
#     shape: tuple[int, int],
# ) -> Float[Tensor, "_ h_out w_out"]:
#     h, w = shape
#     image_new = (image * 255).clip(min=0, max=255).type(torch.uint8)
#     image_new = rearrange(image_new, "c h w -> h w c").detach().cpu().numpy()
#     image_new = Image.fromarray(image_new)
#     image_new = image_new.resize((w, h), Image.LANCZOS)
#     image_new = np.array(image_new) / 255
#     image_new = torch.tensor(image_new, dtype=image.dtype, device=image.device)
#     return rearrange(image_new, "h w c -> c h w")


def rescale(
    image: Float[Tensor, "_ h_in w_in"],
    shape: tuple[int, int],
) -> Float[Tensor, "_ h_out w_out"]:
    h, w = shape

    interp_mode = tf.InterpolationMode.BILINEAR
    return tf.Resize((h, w), interpolation=interp_mode, antialias=True)(image)


def center_crop(
    imagges: list[Float[Tensor, "*#batch _ _ _"]],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
) -> tuple[
    list[Float[Tensor, "*#batch _ h_out w_out"]],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
]:
    *_, h_in, w_in = imagges[0].shape
    h_out, w_out = shape

    # Note that odd input dimensions induce half-pixel misalignments.
    row = (h_in - h_out) // 2
    col = (w_in - w_out) // 2

    # Center-crop the image.
    for i, images in enumerate(imagges):
        images = images[..., :, row : row + h_out, col : col + w_out]
        imagges[i] = images

    # Adjust the intrinsics to account for the cropping.
    intrinsics = intrinsics.clone()
    intrinsics[..., 0, 0] *= w_in / w_out  # fx
    intrinsics[..., 1, 1] *= h_in / h_out  # fy

    return imagges, intrinsics


def rescale_and_crop(
    imagges: list[Float[Tensor, "*#batch _ _ _"]],
    intrinsics: Float[Tensor, "*#batch 3 3"],
    shape: tuple[int, int],
) -> tuple[
    list[Float[Tensor, "*#batch c h_out w_out"]],  # updated images
    Float[Tensor, "*#batch 3 3"],  # updated intrinsics
]:
    *_, h_in, w_in = imagges[0].shape
    h_out, w_out = shape
    assert h_out <= h_in and w_out <= w_in

    scale_factor = max(h_out / h_in, w_out / w_in)
    h_scaled = round(h_in * scale_factor)
    w_scaled = round(w_in * scale_factor)
    assert h_scaled == h_out or w_scaled == w_out

    # Reshape the images to the correct size. Assume we don't have to worry about
    # changing the intrinsics based on how the images are rounded.
    for i, images in enumerate(imagges):
        *batch, c, h, w = images.shape
        images = images.reshape(-1, c, h, w)
        images = torch.stack([rescale(image, (h_scaled, w_scaled)) for image in images])
        images = images.reshape(*batch, c, h_scaled, w_scaled)
        imagges[i] = images
    return center_crop(imagges, intrinsics=intrinsics, shape=shape)


def apply_crop_shim_to_views(views: AnyViews, shape: tuple[int, int]) -> AnyViews:
    keys = [k for k in ["image", "depth", "static_float", "state_mask"] if k in views]
    imagges = [views[k] for k in keys]

    images, intrinsics = rescale_and_crop(imagges, intrinsics=views["intrinsics"], shape=shape)
    return {
        **views,
        "intrinsics": intrinsics,
        **{k: images[i] for i, k in enumerate(keys)},
    }


def apply_crop_shim(example: AnyExample, shape: tuple[int, int]) -> AnyExample:
    """Crop images in the example."""
    return {
        **example,
        "context": apply_crop_shim_to_views(example["context"], shape),
        "target": apply_crop_shim_to_views(example["target"], shape),
    }
