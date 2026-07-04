from math import isqrt

import torch
from e3nn.o3 import angles_to_matrix, wigner_D  # , xyz_to_angles, matrix_to_angles
from einops import einsum
from jaxtyping import Float
from torch import Tensor


def xyz_to_angles(xyz):
    # TAKEN FROM from e3nn.o3, problematic acos
    r"""convert a point :math:`\vec r = (x, y, z)` on the sphere into angles :math:`(\alpha, \beta)`

    .. math::

        \vec r = R(\alpha, \beta, 0) \vec e_z


    Parameters
    ----------
    xyz : `torch.Tensor`
        tensor of shape :math:`(..., 3)`

    Returns
    -------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`
    """
    eps = 1e-6
    xyz = torch.nn.functional.normalize(xyz, p=2, dim=-1)  # forward 0's instead of nan for zero-radius
    xyz = xyz.clamp(-1 + eps, 1 - eps)

    beta = torch.acos(xyz[..., 1])
    alpha = torch.atan2(xyz[..., 0], xyz[..., 2])
    return alpha, beta


def matrix_to_angles(R):
    # TAKEN FROM from e3nn.o3
    r"""conversion from matrix to angles

    Parameters
    ----------
    R : `torch.Tensor`
        matrices of shape :math:`(..., 3, 3)`

    Returns
    -------
    alpha : `torch.Tensor`
        tensor of shape :math:`(...)`

    beta : `torch.Tensor`
        tensor of shape :math:`(...)`

    gamma : `torch.Tensor`
        tensor of shape :math:`(...)`
    """
    # assert torch.allclose(torch.det(R), R.new_tensor(1))
    x = R @ R.new_tensor([0.0, 1.0, 0.0])
    a, b = xyz_to_angles(x)
    R = angles_to_matrix(a, b, torch.zeros_like(a)).transpose(-1, -2) @ R
    c = torch.atan2(R[..., 0, 2], R[..., 0, 0])
    return a, b, c


def rotate_sh(
    sh_coefficients: Float[Tensor, "*#batch n"],
    rotations: Float[Tensor, "*#batch 3 3"],
) -> Float[Tensor, "*batch n"]:
    device = sh_coefficients.device
    dtype = sh_coefficients.dtype

    *_, n = sh_coefficients.shape
    # assert torch.allclose(torch.det(rotations), rotations.new_tensor(1)), f">>>>>>>>\n{rotations}"
    alpha, beta, gamma = matrix_to_angles(rotations)
    result = []
    for degree in range(isqrt(n)):
        with torch.device(device):
            sh_rotations = wigner_D(degree, alpha, beta, gamma).type(dtype)
        sh_rotated = einsum(
            sh_rotations,
            sh_coefficients[..., degree**2 : (degree + 1) ** 2],
            "... i j, ... j -> ... i",
        )
        result.append(sh_rotated)

    return torch.cat(result, dim=-1)


if __name__ == "__main__":
    from pathlib import Path

    import matplotlib.pyplot as plt
    from e3nn.o3 import spherical_harmonics
    from matplotlib import cm
    from scipy.spatial.transform.rotation import Rotation as R

    device = torch.device("cuda")

    # Generate random spherical harmonics coefficients.
    degree = 4
    coefficients = torch.rand((degree + 1) ** 2, dtype=torch.float32, device=device)

    def plot_sh(sh_coefficients, path: Path) -> None:
        phi = torch.linspace(0, torch.pi, 100, device=device)
        theta = torch.linspace(0, 2 * torch.pi, 100, device=device)
        phi, theta = torch.meshgrid(phi, theta, indexing="xy")
        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)
        xyz = torch.stack([x, y, z], dim=-1)
        sh = spherical_harmonics(list(range(degree + 1)), xyz, True)
        result = einsum(sh, sh_coefficients, "... n, n -> ...")
        result = (result - result.min()) / (result.max() - result.min())

        # Set the aspect ratio to 1 so our sphere looks spherical
        fig = plt.figure(figsize=plt.figaspect(1.0))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(
            x.cpu().numpy(),
            y.cpu().numpy(),
            z.cpu().numpy(),
            rstride=1,
            cstride=1,
            facecolors=cm.seismic(result.cpu().numpy()),
        )
        # Turn off the axis planes
        ax.set_axis_off()
        path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(path)

    for i, angle in enumerate(torch.linspace(0, 2 * torch.pi, 30)):
        rotation = torch.tensor(R.from_euler("x", angle.item()).as_matrix(), device=device)
        plot_sh(rotate_sh(coefficients, rotation), Path(f"sh_rotation/{i:0>3}.png"))

    print("Done!")
