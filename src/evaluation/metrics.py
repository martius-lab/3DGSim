from functools import cache

import torch
from einops import pack, reduce, unpack
from jaxtyping import Bool, Float
from lpips import LPIPS
from skimage.metrics import structural_similarity
from torch import Tensor


@torch.no_grad()
def compute_psnr(
    ground_truth: Float[Tensor, "*batch channel height width"],
    predicted: Float[Tensor, "*batch channel height width"],
    masks: Bool[Tensor, "*batch 1 height width"] | None = None,
) -> Float[Tensor, " *batch"]:
    # B, C, H, W -> B, C, HW
    ground_truth = ground_truth.clip(min=0, max=1)
    predicted = predicted.clip(min=0, max=1)
    delta2 = (ground_truth - predicted) ** 2
    if masks is not None:
        masks = masks.flatten(-2).float()
        delta2 = masks * delta2.flatten(-2)
        mse = reduce(delta2, "... hw -> ...", "sum") / masks.sum(dim=(-1))
    else:
        mse = reduce(delta2, "... h w -> ...", "mean")

    mse = reduce(mse, "... c -> ...", "mean")
    return -10 * mse.log10()


@cache
def get_lpips(device: torch.device) -> LPIPS:
    return LPIPS(net="vgg").to(device)


@torch.no_grad()
def compute_lpips(
    ground_truth: Float[Tensor, "*batch channel height width"],
    predicted: Float[Tensor, "*batch channel height width"],
) -> Float[Tensor, " *batch"]:
    gt, ps = pack([ground_truth], "* channel height width")
    pred, ps = pack([predicted], "* channel height width")
    # --
    # LPIPS is per-image; chunk the batch to bound VGG activation memory.
    lpips = get_lpips(predicted.device)
    chunk = 32
    value = torch.cat(
        [
            lpips.forward(gt[i : i + chunk], pred[i : i + chunk], normalize=True)[:, 0, 0, 0]
            for i in range(0, gt.shape[0], chunk)
        ]
    )
    # --
    [value] = unpack(value, ps, "*")
    return value


@torch.no_grad()
def compute_ssim(
    ground_truth: Float[Tensor, "*batch channel height width"],
    predicted: Float[Tensor, "*batch channel height width"],
) -> Float[Tensor, " *batch"]:
    gt, ps = pack([ground_truth], "* channel height width")
    pred, ps = pack([predicted], "* channel height width")
    # --
    ssim = [
        structural_similarity(
            _gt.detach().cpu().numpy(),
            _hat.detach().cpu().numpy(),
            win_size=11,
            gaussian_weights=True,
            channel_axis=0,
            data_range=1.0,
        )
        for _gt, _hat in zip(gt, pred)
    ]
    ret_ssim = torch.tensor(ssim, dtype=predicted.dtype, device=predicted.device)
    # --
    [ret_ssim] = unpack(ret_ssim, ps, "*")
    return ret_ssim
