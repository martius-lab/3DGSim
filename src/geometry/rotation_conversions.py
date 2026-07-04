import roma
import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor


# base representation is rotation matrix
def to_rot(rot_repr: Float[Tensor, "*#batch d"], normalize=True, eps=1e-7):
    rot_mat = to_rot_mat(rot_repr, normalize=normalize, eps=eps)
    return rearrange(rot_mat, "... i j -> ... (i j)")


def to_rot_mat(
    rot_repr: Float[Tensor, "*#batch d"],
    normalize: bool = True,
    eps: float = 1e-8,
):
    """_summary_

    Args:
        rot_repr (Float[Tensor, ): It can be a quaternion, a 6D GramSchmidt Orthogonolization or a procrustes matrix.
                                   quaternion is always with real part last. (XYZW)

    Returns:
        rot: rotation matrix (Float[Tensor, "*#batch 9"])
    """
    if rot_repr.size(-1) == 4:
        if normalize:
            rot_repr = rot_repr / (rot_repr.norm(dim=-1, keepdim=True) + eps)
        rotation = quat2rotmat(rot_repr)
    elif rot_repr.size(-1) == 6:
        rotation = gram2rotmat(rot_repr)
    elif rot_repr.size(-1) == 9:
        rotation = procrustes2rotmat(rot_repr)
    else:
        raise ValueError(f"Invalid rotation shape {rot_repr.shape}.")
    return rotation


def normalize_repr(rot_repr: Float[Tensor, "*#batch d"], eps=1e-8) -> Float[Tensor, "*#batch d"]:
    if rot_repr.size(-1) == 4:
        rot_repr = rot_repr / (rot_repr.norm(dim=-1, keepdim=True) + eps)
    else:
        rot_repr = rotmat_to_repr(to_rot_mat(rot_repr), rot_repr.size(-1))
    return rot_repr


def rotmat_to_repr(rotmat: Float[Tensor, "*#batch 3 3"], rep_dim: int = 9) -> Float[Tensor, "*#batch d"]:
    if rep_dim == 9:
        return rot2procrustes(rotmat)
    elif rep_dim == 4:
        return rot2quat(rotmat)
    elif rep_dim == 6:
        return rot2gram(rotmat)
    else:
        raise ValueError(f"Invalid representation dimension {rep_dim}.")


def rot_to_repr(rot: Float[Tensor, "*#batch 9"], dim: int) -> Float[Tensor, "*#batch d"]:
    rotmat = rearrange(rot, "... (i j) -> ... i j", i=3)
    return rotmat_to_repr(rotmat, dim)


def increment_rot(
    rot_repr: Float[Tensor, "*#batch r"],
    delta: Float[Tensor, "*#batch c"],
    eps: float = 1e-7,
):
    dim = rot_repr.size(-1)

    if delta.size(-1) == 3:
        delta_rot = roma.rotvec_to_rotmat(delta, epsilon=eps)
    else:
        delta_rot = to_rot_mat(delta, normalize=True, eps=eps)

    new_rot_mat = delta_rot @ to_rot_mat(rot_repr)
    return rotmat_to_repr(new_rot_mat, dim)


def increment_increments(
    deltas_rot: Float[Tensor, "b t *#other 3"],
    eps: float = 1e-7,
) -> Float[Tensor, "b t *#other 3 3"]:

    delta_rots = roma.rotvec_to_rotmat(deltas_rot, epsilon=eps)

    U_cum = [delta_rots[:, 0]]
    for i in range(1, delta_rots.size(1)):
        U_cum.append(torch.matmul(delta_rots[:, i], U_cum[-1]))
    U_cum = torch.stack(U_cum, 1)

    return rearrange(U_cum, "... i j -> ... (i j)")


def build_covariance(
    scale: Float[Tensor, "*#batch 3"],
    rot_repr: Float[Tensor, "*#batch d"],
    normalize=True,
    eps=1e-7,
) -> Float[Tensor, "*batch 3 3"]:
    scale = scale.diag_embed()

    rotation = to_rot_mat(rot_repr, normalize=normalize, eps=eps)
    return (
        rotation @ scale @ rearrange(scale, "... i j -> ... j i") @ rearrange(rotation, "... i j -> ... j i")
    )


# in case base repr is quat (unused for now)
def quat2gram(quat: torch.Tensor) -> torch.Tensor:
    """
    Quaternion to 6D GramSchmidt Orthogonolization.
    See https://arxiv.org/abs/2404.11735
    """
    # from mujoco to roma/scipy ordering
    quat[..., 0], quat[..., -1] = quat[..., -1].clone(), quat[..., 0].clone()
    # roma uses XYZW ordering
    rotmat = roma.unitquat_to_rotmat(quat)
    gram = rotmat[..., :2]
    return gram


def gram2quat(gram: torch.Tensor) -> torch.Tensor:
    """
    6D GramSchmidt Orthogonolization to quaternion.
    See https://arxiv.org/abs/2404.11735
    """
    if gram.size(-1) == 6:
        gram = rearrange(gram, "... 6 -> ... 3 2")
    rotmat = roma.special_gramschmidt(gram, epsilon=1e-7)
    quat = roma.rotmat_to_unitquat(rotmat)
    # from roma/scipy to mujoco ordering
    # roma uses XYZW ordering
    quat[..., 0], quat[..., -1] = quat[..., -1].clone(), quat[..., 0].clone()
    return quat


# gramschmidt representation
def gram2rotmat(gram: torch.Tensor) -> torch.Tensor:
    """
    6D GramSchmidt Orthogonolization to rotation matrix.
    See https://arxiv.org/abs/2404.11735
    """
    if gram.size(-1) == 6:
        gram = rearrange(gram, "... (three two) -> ... three two", three=3)
    rotmat = roma.special_gramschmidt(gram, epsilon=1e-7)
    return rotmat


def rot2gram(rotmat: torch.Tensor) -> torch.Tensor:
    """
    Rotation matrix to 6D GramSchmidt Orthogonolization.
    See https://arxiv.org/abs/2404.11735
    """
    gram = rotmat[..., :2]
    return rearrange(gram, "... three two -> ... (three two)")


def matrix_to_quaternion(rotmat: torch.Tensor) -> torch.Tensor:
    """
    Rotation matrix to quaternion.
    """
    # roma uses XYZW ordering
    quat = roma.rotmat_to_unitquat(rotmat)
    return quat


# quaternion representation
def quat2rotmat(quat: torch.Tensor) -> torch.Tensor:
    """
    Quaternion to rotation matrix.
    """
    # roma uses XYZW ordering
    quat = quat / (quat.norm(dim=-1, keepdim=True) + 1e-8)
    rotmat = roma.unitquat_to_rotmat(quat)
    return rotmat


def rot2quat(rotmat: torch.Tensor) -> torch.Tensor:
    """
    Rotation matrix to quaternion.
    """
    # roma uses XYZW ordering
    quat = roma.rotmat_to_unitquat(rotmat)
    return quat


# procrustes
def procrustes2rotmat(procrustes: torch.Tensor) -> torch.Tensor:
    """
    Procrustes to rotation matrix.
    """
    if procrustes.size(-1) == 9:
        procrustes = rearrange(procrustes, "... (three threee) -> ... three threee", three=3)
    rotmat = roma.special_procrustes(procrustes)
    return rotmat


def rot2procrustes(rot: torch.Tensor) -> torch.Tensor:
    """
    Rotation matrix to procrustes.
    """
    return rearrange(rot, "... i j -> ... (i j)")


def rotmat_times_rotrepr(
    rotmat: Float[Tensor, "*#batch 3 3"], rot_repr: Float[Tensor, "*#batch d"]
) -> Float[Tensor, "*#batch d"]:
    if rot_repr.size(-1) == 4:
        rot_repr = rot_repr / (rot_repr.norm(dim=-1, keepdim=True) + 1e-8)
        return roma.quat_product(rot2quat(rotmat), rot_repr)
    else:
        return rotmat_to_repr(rotmat @ to_rot_mat(rot_repr), rot_repr.size(-1))


if __name__ == "__main__":
    B, T, G = 2, 4, 6
    eye = torch.eye(3)[None, None, None].repeat(B, T, G, 1, 1).flatten(-2)

    deltas = torch.randn(B, T, G, 3)

    z_x = rotation_update(eye, deltas, "z_x")
    cx_x = rotation_update(eye, deltas, "cx_x")
    cx_dx = rotation_update(eye, deltas, "cx_dx")
    quat = torch.tensor([0.5, 0.5, 0, 0.0])
    gram = quat2gram(quat)
    quat = gram2quat(gram)
    # expect result to be [0.5, 0, 0, 0.5] or [0.7, 0, 0, 0.7]
    print(quat)
