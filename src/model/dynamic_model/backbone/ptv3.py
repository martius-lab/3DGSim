"""
Point Transformer - V3 Mode1
Pointcept detached version

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import math
import sys
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import partial
from pprint import pprint
from typing import Any, List, Literal, Optional

import einops as eo

# import pointops2.pointops as pointops
import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.utils
import torch_scatter
from addict import Dict
from jaxtyping import Bool, Float
from timm.layers import DropPath

from ....global_cfg import get_cfg

# from ldbs.model.serialization import encode
from ....serialization import encode

try:
    import flash_attn
except ImportError:
    flash_attn = None


@torch.inference_mode()
def offset2bincount(offset):
    return torch.diff(offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long))


@torch.inference_mode()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(len(bincount), device=offset.device, dtype=torch.long).repeat_interleave(bincount)


@torch.inference_mode()
def batch2offset(batch):
    # assumes batch is sorted
    return torch.cumsum(batch.bincount(), dim=0).long()


# futhest point sampling
@torch.inference_mode()
def futhest_point_sampling(xyz: torch.Tensor, offset, down_ratio: int = 0.5, n_new: int | list[int] = None):
    # xyz: [N, 3]
    # offset: [B]
    # down_ratio: float or list of float
    # n_new: int or list of int
    B = offset.size(0)
    assert xyz.size(0) == offset[-1], f"{xyz.size(0)} != {offset[-1]}"

    bincount = offset2bincount(offset)

    # compute new bincount
    if n_new is not None:
        if isinstance(n_new, int):
            n_new = xyz.new_tensor([n_new] * B, dtype=torch.long)

        new_bincount = torch.min(n_new, bincount)
        assert len(new_bincount) == B, f"new_bincount != B,  {len(new_bincount)} != {B}"
    else:
        new_bincount = bincount * down_ratio

    # compute new offset
    new_offset = torch.cumsum(new_bincount, dim=0)

    # new_offset = torch.cuda.IntTensor(new_offset)
    down_idx = pointops.furthestsampling(xyz, offset.int(), new_offset.int())
    return down_idx, new_offset


class Point(Dict):
    """
    Point Structure of Pointcept

    A Point (point cloud) in Pointcept is a dictionary that contains various properties of
    a batched point cloud. The property with the following names have a specific definition
    as follows:

    - "coord": original coordinate of point cloud;
    - "grid_coord": grid coordinate for specific grid size (related to GridSampling);
    Point also support the following optional attributes:
    - "offset": if not exist, initialized as batch size is 1;
    - "batch": if not exist, initialized as batch size is 1;
    - "feat": feature of point cloud, default input of model;
    - "grid_size": Grid size of point cloud (related to GridSampling);
    (related to Serialization)
    - "serialized_depth": depth of serialization, 2 ** depth * grid_size is the maximum range of point cloud
    - "serialized_code": a list of serialization codes;
    - "serialized_order": a list of serialization order determined by code;
    - "serialized_inverse": a list of inverse mapping determined by code;
    (related to Sparsify: SpConv)
    - "sparse_shape": Sparse shape for Sparse Conv Tensor;
    - "sparse_conv_feat": SparseConvTensor init with information provide by Point;
    (traceable attributes)
    - "merger_parent": parent pointdict for merger;
    - "pooling_parent": parent pointdict for pooling;
    - "pooling_inverse": inverse index for pooling (gives parent from the pooled point set);

    (memorized attributes for attention)
    - "rel_pos_key": relative position key for attention;
    - "pad": padding index for serialized point cloud;
    - "unpad": unpadding index for serialized point cloud;
    - "cu_seqlens_key": cumulative sequence length key for serialized point cloud;
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # If one of "offset" or "batch" do not exist, generate by the existing one
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)

        if "batch_time_coord" not in self.keys() and "time_depth" not in self.keys():
            if "time_coord" in self.keys():
                # make sure the batch is sorted so that with point.batch[-1] we get the batch_size
                # (needed in sparsify for spconv)
                # assumes that point clouds where sorted by: B,T
                time_coord = self.time_coord
                batch = self.batch
                assert torch.all(batch[:1] <= batch[1:]) and torch.all(
                    time_coord[:1] <= time_coord[1:]
                ), "Input temporal point cloud should be sorted by B,T"
                # check that time is power of 2
                time_coord_max_bit_length = int(time_coord.max()).bit_length()
                assert time_coord[-1] + 1 == 2**time_coord_max_bit_length, "Time should be power of 2"
                # the models don't need to know the real batch index.
                # batch_ixs and time_ixs are now encoded in the batch_time_coord
                # we can get them back by:
                # batch_ixs = batch_time_coord >> time_depth
                # time_ixs = batch_time_coord & ((1 << time_depth) - 1)

                batch_ixs = batch.int()
                time_ixs = time_coord.int()
                self["time_depth"] = time_coord_max_bit_length
                self["batch_time_coord"] = batch_ixs << self.time_depth | time_ixs

                # update batch and offset (we see different time as different batch)
                unique, cluster = torch.unique(self["batch_time_coord"], return_inverse=True)
                self.batch = torch.arange(len(unique), device=self.coord.device)[cluster]
                self.offset = batch2offset(self.batch)

                # the order is only changed during pooling and unpooling.
                # But since batch has the highest priority, and batch itself is ordered via B,T,
                # we can assume that the point cloud always keeps the order B,T.
                # The order of the points in each B,T is controlled by the serialization code.

    def prepare_grid_coord_ifnotexist(self):
        if "grid_coord" not in self.keys():
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()

    def indexing(self, index):
        """
        Indexing Point Cloud

        index: a tensor with the same length of point cloud
        """
        assert isinstance(index, torch.Tensor)
        assert index.shape[0] == self.coord.shape[0]

        # 1. SPCONV (can be ignored in the decoder, we dont use any spconv operations)
        # ["sparse_shape", "sparse_conv_feat"]

        # 2. Attention related (can be recomputed but may be unnecessary, if we want to save time)
        # ["pad", "unpad", "cu_seqlens_key", "rel_pos_{order_index}"]

        # 3. Per point info (N, ..) (simple to index)
        # ["batch", "coord", "feat", "grid_coord",
        #  "serialized_code", "serialized_order", "serialized_inverse"]

        # 4. Time related (time_depth=1, batch_time_coord can be directly indexed)
        # ["time_depth", "batch_time_coord"]
        # 5. Merger related (should also be indexed)
        # ["merger_parent"]
        # 6. Pooling related (HMM, should also be indexed)
        # ["pooling_inverse", "pooling_parent"] , .. "unpooling_parent"
        # 7. Stays same
        # ["serialized_depth"]

    def serialization(self, order="z", depth=None, shuffle_orders=False):
        """
        Point Cloud Serialization

        relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]
        """
        assert "batch" in self.keys()
        self.prepare_grid_coord_ifnotexist()

        if depth is None:
            # Adaptive measure the depth of serialization cube (length = 2 ^ depth)
            depth = int(self.grid_coord.max()).bit_length() + 1
        # Maximum bit length for serialization code is 63 (int64)
        assert depth * 3 + len(self.offset).bit_length() <= 63
        # Here we follow OCNN and set the depth limitation to 16 (48bit) for the point position.
        # Although depth is limited to less than 16, we can encode a 655.36^3 (2^16 * 0.01) meter^3
        # cube with a grid size of 0.01 meter. We consider it is enough for the current stage.
        # We can unlock the limitation by optimizing the z-order encoding function if necessary.
        assert (
            depth <= 16
        ), f"Max grid_coord, coord: {self.grid_coord.max(0).values, self.coord.max(0).values}, depth: {depth}"

        # The serialization codes are arranged as following structures:
        # [Order1 ([n]),
        #  Order2 ([n]),
        #   ...
        #  OrderN ([n])] (k, n)
        code = [encode(self.grid_coord, self.batch, depth, order=order_) for order_ in order]
        code = torch.stack(code)
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(code.shape[0], 1),
        )

        if shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        self["serialized_depth"] = depth
        self["serialized_code"] = code
        self["serialized_order"] = order
        self["serialized_inverse"] = inverse

    def sparsify(self, pad=96):
        """
        Point Cloud Serialization

        Point cloud is sparse, here we use "sparsify" to specifically refer to
        preparing "spconv.SparseConvTensor" for SpConv.

        relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]

        pad: padding sparse for sparse shape.
        """
        assert {"feat", "batch"}.issubset(self.keys())
        self.prepare_grid_coord_ifnotexist()
        if "sparse_shape" in self.keys():
            sparse_shape = self.sparse_shape
        else:
            sparse_shape = torch.add(torch.max(self.grid_coord, dim=0).values, pad).tolist()
        sparse_conv_feat = spconv.SparseConvTensor(
            features=self.feat,
            indices=torch.cat([self.batch.unsqueeze(-1).int(), self.grid_coord.int()], dim=1).contiguous(),
            spatial_shape=sparse_shape,
            batch_size=self.batch[-1].tolist() + 1,
        )
        self["sparse_shape"] = sparse_shape
        self["sparse_conv_feat"] = sparse_conv_feat

    def time_ixs(self):
        assert "batch_time_coord" in self.keys()
        return self.batch_time_coord & ((1 << self.time_depth) - 1)

    def batch_ixs(self):
        batch = self.batch
        if "batch_time_coord" in self.keys():
            batch = self.batch_time_coord >> self.time_depth
        return batch


class PointModule(nn.Module):
    r"""PointModule
    placeholder, all module subclass from this will take Point in PointSequential.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PointSequential(PointModule):
    r"""A sequential container.
    Modules will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of modules can also be passed in.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        for k, module in self._modules.items():
            # Point module
            if isinstance(module, PointModule):
                input = module(input)
            # Spconv module
            elif spconv.modules.is_spconv_module(module):
                if isinstance(input, Point):
                    input.sparse_conv_feat = module(input.sparse_conv_feat)
                    input.feat = input.sparse_conv_feat.features
                else:
                    input = module(input)
            # PyTorch module
            else:
                if isinstance(input, Point):
                    input.feat = module(input.feat)
                    if "sparse_conv_feat" in input.keys():
                        input.sparse_conv_feat = input.sparse_conv_feat.replace_feature(input.feat)
                elif isinstance(input, spconv.SparseConvTensor):
                    if input.indices.shape[0] != 0:
                        input = input.replace_feature(module(input.features))
                else:
                    input = module(input)
        return input


class PDNorm(PointModule):
    def __init__(
        self,
        num_features,
        norm_layer,
        context_channels=256,
        conditions=("ScanNet", "S3DIS", "Structured3D"),
        decouple=True,
        adaptive=False,
    ):
        super().__init__()
        self.conditions = conditions
        self.decouple = decouple
        self.adaptive = adaptive
        if self.decouple:
            self.norm = nn.ModuleList([norm_layer(num_features) for _ in conditions])
        else:
            self.norm = norm_layer
        if self.adaptive:
            self.modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(context_channels, 2 * num_features, bias=True)
            )

    def forward(self, point):
        assert {"feat", "condition"}.issubset(point.keys())
        if isinstance(point.condition, str):
            condition = point.condition
        else:
            condition = point.condition[0]
        if self.decouple:
            assert condition in self.conditions
            norm = self.norm[self.conditions.index(condition)]
        else:
            norm = self.norm
        point.feat = norm(point.feat)
        if self.adaptive:
            assert "context" in point.keys()
            shift, scale = self.modulation(point.context).chunk(2, dim=1)
            point.feat = point.feat * (1.0 + scale) + shift
        return point


class RPE(torch.nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)  # clamp into bnd
            + self.pos_bnd  # relative position to positive index
            + torch.arange(3, device=coord.device) * self.rpe_num  # x, y, z stride
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out


class SerializedAttention(PointModule):
    """
    adds following keys so that we don't need to recompute for decoder
    - "pad", "unpad", "cu_seqlens_key", "rel_pos_{order_index}" (if rpe is enabled)
    """

    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash
        if enable_flash:
            assert enable_rpe is False, "Set enable_rpe to False when enable Flash Attention"
            assert upcast_attention is False, "Set upcast_attention to False when enable Flash Attention"
            assert upcast_softmax is False, "Set upcast_softmax to False when enable Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            # when disable flash attention, we still don't want to use mask
            # consequently, patch size will auto set to the
            # min number of patch_size_max and number of points
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = torch.nn.Dropout(attn_drop)

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        # order: (N', K) index to get ordered-padded points from the unordered-unpadded points
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]  # (N', K, K, 3)

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        if pad_key not in point.keys() or unpad_key not in point.keys() or cu_seqlens_key not in point.keys():
            offset = point.offset
            bincount = offset2bincount(offset)
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            # only pad point when num of points larger than patch_size
            # not sure but looks unnecessary since we pad to min(patch_size, nr of points in smallest batch)
            bincount_pad = torch.where(bincount > self.patch_size, bincount_pad, bincount)

            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            for i in range(len(offset)):
                total_pads_added_before_batch_i = _offset_pad[i] - _offset[i]

                # if we have to pad: borrow from previous patch
                if bincount[i] != bincount_pad[i]:
                    # == self.patch_size -  (bincount[i] % self.patch_size)
                    pad_added_at_batch_i = bincount_pad[i] - bincount[i]

                    # start and end of added pad at the end of the last patch
                    end_added_pad = _offset_pad[i + 1]  # end of padded batch_i
                    start_added_pad = end_added_pad - pad_added_at_batch_i

                    # start and end of borrowed pad at the previous patch
                    end_borrowed_pad = end_added_pad - self.patch_size  # which is start of last patch
                    start_borrowed_pad = end_borrowed_pad - pad_added_at_batch_i

                    pad[start_added_pad:end_added_pad] = pad[start_borrowed_pad:end_borrowed_pad]

                unpad[_offset[i] : _offset[i + 1]] += total_pads_added_before_batch_i
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= total_pads_added_before_batch_i

                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )

            point[pad_key] = pad  # (N') index to get the padded points from the unpadded points
            point[unpad_key] = unpad  # (N) index to get the unpadded points from the padded points
            # (batch_size + 1) index to get the cumulative patch lengths of all the patches
            point[cu_seqlens_key] = nn.functional.pad(torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1])

        return point[pad_key], point[unpad_key], point[cu_seqlens_key]
        # return point.pop(pad_key), point.pop(unpad_key), point.pop(cu_seqlens_key)

    def forward(self, point):
        # if not enable_flash, set patch_size to min of patch_size_max and nr of points in the smallest batch
        if not self.enable_flash:
            self.patch_size = min(offset2bincount(point.offset).min().tolist(), self.patch_size_max)

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        # should happen before because it can change self.patch_size if not enable_flash
        pad, unpad, cu_seqlens = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]  # (N'* K *) ix of ordered-padded pts
        inverse = unpad[point.serialized_inverse[self.order_index]]  # (N) ix of unordered-unpaded pts

        # padding and reshape feat and batch for serialized point patch
        qkv = self.qkv(point.feat)[order]
        # print(
        #     f"num_heads: {H}, patch_size: {K}, channels: {C}, qkv.shape: {qkv.reshape(-1, 3, H, C // H).shape}"
        # )

        if not self.enable_flash:
            # N': number of patches, K: patch size, H: num_heads, C': channels per head
            # encode and reshape qkv: (N', K, 3, H, C') => (3, N', H, K, C')
            q, k, v = qkv.reshape(-1, K, 3, H, C // H).permute(2, 0, 3, 1, 4).unbind(dim=0)
            # attn
            if self.upcast_attention:
                q = q.float()
                k = k.float()
            attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
            if self.enable_rpe:
                attn = attn + self.rpe(self.get_rel_pos(point, order))
            if self.upcast_softmax:
                attn = attn.float()
            attn = self.softmax(attn)
            attn = self.attn_drop(attn).to(qkv.dtype)
            feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        else:
            # cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
            # of the sequences in the batch, used to index into qkv.
            # max_seqlen: int. Maximum sequence length in the batch.
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv.half().reshape(-1, 3, H, C // H),
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, C)
            feat = feat.to(qkv.dtype)
        feat = feat[inverse]

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
        prepend_layer_norm_act: tuple[Any, Any] = None,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

        self.prepend = prepend_layer_norm_act is not None
        if self.prepend:
            norm_pre, act_pre = prepend_layer_norm_act
            self.norm_pre, self.act_pre = norm_pre(in_channels), act_pre()

    def forward(self, x):
        if self.prepend:
            x = self.norm_pre(x)
            x = self.act_pre(x)

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm
        self.gradient_checkpointing = get_cfg().train.gradient_checkpointing

        self.cpe = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(DropPath(drop_path) if drop_path > 0.0 else nn.Identity())

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)

        if self.gradient_checkpointing:
            point = self.drop_path(torch.utils.checkpoint.checkpoint(self.attn, point))
        else:
            point = self.drop_path(self.attn(point))

        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        if self.gradient_checkpointing:
            point = self.drop_path(torch.utils.checkpoint.checkpoint(self.mlp, point))
        else:
            point = self.drop_path(self.mlp(point))

        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        # update sparse_conv_feat, because in encoder we use spconv for cpe
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class SerializedPooling(PointModule):
    """
    adds following keys so that we don't need to recompute for decoder
    -"pooling_inverse", "pooling_parent"
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,  # record parent and cluster
        grid_mode=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.grid_mode = grid_mode
        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)

        self.norm = None
        self.act = None
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def serialized_forward(self, point: Point):
        # this way of pooling saves 1 call to point.serialization()
        pooling_depth = (math.ceil(self.stride) - 1).bit_length()
        if pooling_depth > point.serialized_depth:
            pooling_depth = 0
        assert {
            "serialized_code",
            "serialized_order",
            "serialized_inverse",
            "serialized_depth",
        }.issubset(point.keys()), "Run point.serialization() point cloud before SerializedPooling"

        code = point.serialized_code >> pooling_depth * 3
        # cluster -> cluster_id for each point
        code_, cluster, counts = torch.unique(
            code[0],  # only use the first serialization order
            sorted=True,
            return_inverse=True,
            return_counts=True,
        )
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]
        # generate down code, order, inverse
        code = code[:, head_indices]
        order = torch.argsort(code)
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(code.shape[0], 1),
        )

        if self.shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        # collect information
        point_dict = Dict(
            feat=torch_scatter.segment_csr(self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce),
            coord=torch_scatter.segment_csr(point.coord[indices], idx_ptr, reduce="mean"),
            grid_coord=point.grid_coord[head_indices] >> pooling_depth,
            serialized_code=code,
            serialized_order=order,
            serialized_inverse=inverse,
            serialized_depth=point.serialized_depth - pooling_depth,
            batch=point.batch[head_indices],
        )

        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context

        if {"time_depth", "batch_time_coord"}.issubset(point.keys()):
            point_dict["time_depth"] = point.time_depth
            point_dict["batch_time_coord"] = point.batch_time_coord[head_indices]

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.sparsify()
        return point

    def grid_forward(self, point: Point):
        point.prepare_grid_coord_ifnotexist()
        grid_coord = torch.div(point.grid_coord, self.stride, rounding_mode="trunc")

        # cluster==inverse: indices of where the elements from the input ended up in the unique list
        # cluster: part of which cluster the element belongs to
        grid_coord = grid_coord | point.batch.view(-1, 1) << 16
        grid_coord, cluster, counts = torch.unique(
            grid_coord,
            sorted=True,
            return_inverse=True,
            return_counts=True,
            dim=0,
        )
        grid_coord = grid_coord & 0xFFFF
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]
        point_dict = Dict(
            feat=torch_scatter.segment_csr(self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce),
            coord=torch_scatter.segment_csr(point.coord[indices], idx_ptr, reduce="mean"),
            grid_coord=grid_coord,
            batch=point.batch[head_indices],
        )
        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context

        if {"time_depth", "batch_time_coord"}.issubset(point.keys()):
            point_dict["time_depth"] = point.time_depth
            point_dict["batch_time_coord"] = point.batch_time_coord[head_indices]

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
        order = point.serialized_order
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.serialization(
            order=("z", "z-trans", "hilbert", "hilbert-trans"), shuffle_orders=self.shuffle_orders
        )
        point.sparsify()
        return point

    def forward(self, point: Point):
        if self.stride == 2 ** (math.ceil(self.stride) - 1).bit_length() and not self.grid_mode:
            return self.serialized_forward(point)
        else:
            return self.grid_forward(point)
        return self.grid_forward(point)


class SerializedUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,  # record parent and cluster
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pop("pooling_inverse")
        point = self.proj(point)
        parent = self.proj_skip(parent)
        parent.feat = parent.feat + point.feat[inverse]
        # no spconv in decoder so no need to update sparse_conv_feat
        # parent.sparse_conv_feat = parent.sparse_conv_feat.replace_feature(parent.feat)

        if self.traceable:
            parent["unpooling_parent"] = point
        return parent


class Merger(PointModule):
    """
    adds following keys so that we don't need to recompute for decoder
    - "merger_parent"
    """

    def __init__(
        self,
        window_size=2,
        time_emb=None,
        traceable=True,  # record parent point
    ):
        super().__init__()
        self.pos_emb = None
        if time_emb is not None and window_size > 1:
            self.pos_emb = nn.Parameter(torch.zeros(window_size, time_emb))
            torch.nn.init.trunc_normal_(self.pos_emb, std=0.02)

        self.shift = int(math.log2(window_size)) if window_size >= 1 else -1
        self.traceable = traceable

    def forward(self, point):
        # cases where merger is not needed
        if self.shift == 0 or point.time_depth == 0:
            point2 = Point(**point)
            point2["merger_parent"] = point
            return point2

        # shift can be up to time_depth
        shift = min(self.shift, point.time_depth)
        new_time_depth = point.time_depth - shift
        new_batch_time_coord = point.batch_time_coord >> shift

        # update batch and offset (we see different time as different batch)
        unique, cluster = torch.unique(new_batch_time_coord, return_inverse=True)
        new_batch = torch.arange(len(unique), device=point.coord.device)[cluster]
        new_offset = batch2offset(new_batch)

        # relative time encooding
        new_feat = None
        if self.pos_emb is not None:
            time_coords_before = point.batch_time_coord & ((1 << point.time_depth) - 1)
            time_coords_after = (unique << shift & ((1 << point.time_depth) - 1))[cluster]
            relative_time_coords = time_coords_before - time_coords_after  # max is window_size
            new_feat = point.feat + self.pos_emb[relative_time_coords]

        # serialization: stays the same, only change batch bits
        depth = point.serialized_depth * 3  # doesnt change
        # zero_out batch bits and update with new_batch bits
        # code_bits & clear_batch_bits | new_batch_bits
        new_scode = point.serialized_code & ((1 << depth) - 1) | (new_batch << depth)
        new_sorder = torch.argsort(new_scode)
        new_sinverse = torch.zeros_like(new_sorder).scatter_(
            dim=1,
            index=new_sorder,
            src=torch.arange(0, new_scode.shape[1], device=new_sorder.device).repeat(new_scode.shape[0], 1),
        )

        point_dict = Dict(
            feat=point.feat if new_feat is None else new_feat,
            coord=point.coord,
            grid_coord=point.grid_coord,
            # changing
            batch=new_batch,
            offset=new_offset,
            batch_time_coord=new_batch_time_coord,
            time_depth=new_time_depth,
            # serialization (serialized_depth stays the same)
            serialized_depth=point.serialized_depth,
            serialized_code=new_scode,
            serialized_order=new_sorder,
            serialized_inverse=new_sinverse,
        )

        if self.traceable:
            point_dict["merger_parent"] = point

        new_sorder = point.serialized_order
        new_point = Point(point_dict)
        new_point.sparsify()
        return new_point


class Unmerger(PointModule):
    """We should possible to recover only the last time step of the serialized point cloud from parent."""

    def __init__(
        self,
        last_time_step_only=True,
    ):
        super().__init__()

        self.last_time_step_only = last_time_step_only

    def forward(self, point):
        # FIXME
        assert "merger_parent" in point.keys()
        parent = point.pop("merger_parent")
        if parent is None:
            parent = point
        else:
            parent.feat = point.feat

        return parent


class Embedding(PointModule):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        # TODO: check remove spconv
        self.stem = PointSequential(
            conv=spconv.SubMConv3d(
                in_channels,
                embed_channels,
                kernel_size=5,
                padding=1,
                bias=False,
                indice_key="stem",
            )
        )
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

    def forward(self, point: Point):
        point = self.stem(point)
        return point


orderT = Literal["z", "z-trans", "hilbert", "hilbert-trans"]


@dataclass
class PointTrainsformerV3Cfg:
    in_channels: int = 6
    out_channels: int = 3
    grid_size: float = 0.01
    order: List[orderT] = field(default_factory=lambda: ("z", "z-trans", "hilbert", "hilbert-trans"))
    mlp_ratio: int = 4
    cls_mode: bool = False
    # temporal merger
    temporal_merger: bool = True
    with_time_emb: bool = True
    merge_window_size: list[int] = field(default_factory=lambda: (2, 2, 2, 2))
    # encoder
    grid_mode: bool = False
    stride: list[int] = field(default_factory=lambda: (2, 2, 2, 2))
    enc_depths: list[int] = field(default_factory=lambda: (2, 2, 2, 6, 2))
    enc_channels: list[int] = field(default_factory=lambda: (32, 64, 128, 256, 512))
    enc_num_head: list[int] = field(default_factory=lambda: (2, 4, 8, 16, 32))
    enc_patch_size: list[int] = field(default_factory=lambda: (1024, 1024, 1024, 1024, 1024))
    # decoder
    dec_depths: list[int] = field(default_factory=lambda: (2, 2, 2, 2))
    dec_channels: list[int] = field(default_factory=lambda: (64, 64, 128, 256))
    dec_num_head: list[int] = field(default_factory=lambda: (4, 4, 8, 16))
    dec_patch_size: list[int] = field(default_factory=lambda: (1024, 1024, 1024, 1024))
    # self attention
    qkv_bias: bool = True
    qk_scale = None
    attn_drop: float = 0.0
    proj_drop: float = 0.0
    drop_path: float = 0.3
    pre_norm: bool = True
    shuffle_orders: bool = True
    enable_rpe: bool = False
    enable_flash: bool = True
    upcast_attention: bool = False
    upcast_softmax: bool = False
    # normalization
    pdnorm_bn: bool = False
    pdnorm_ln: bool = False
    pdnorm_decouple: bool = True
    pdnorm_adaptive: bool = False
    pdnorm_affine: bool = True
    pdnorm_conditions: List[str] = field(default_factory=lambda: ("ScanNet", "S3DIS", "Structured3D"))


class PointTransformerV3(PointModule):
    def __init__(
        self,
        cfg: PointTrainsformerV3Cfg,
    ):
        super().__init__()
        pprint(cfg)
        print()
        self.cfg = cfg
        self.grid_size = cfg.grid_size
        self.max_distance = 2**14 * cfg.grid_size - 0.01
        self.num_stages = len(cfg.enc_depths)
        self.order = [cfg.order] if isinstance(cfg.order, str) else cfg.order
        self.cls_mode = cfg.cls_mode
        self.shuffle_orders = cfg.shuffle_orders
        self.temporal_merger = cfg.temporal_merger

        assert self.num_stages == len(cfg.stride) + 1
        assert self.num_stages == len(cfg.enc_depths)
        assert self.num_stages == len(cfg.enc_channels)
        assert self.num_stages == len(cfg.enc_num_head)
        assert self.num_stages == len(cfg.enc_patch_size)
        assert self.cls_mode or self.num_stages == len(cfg.dec_depths) + 1
        assert self.cls_mode or self.num_stages == len(cfg.dec_channels) + 1
        assert self.cls_mode or self.num_stages == len(cfg.dec_num_head) + 1
        assert self.cls_mode or self.num_stages == len(cfg.dec_patch_size) + 1

        # norm layers
        if cfg.pdnorm_bn:
            bn_layer = partial(
                PDNorm,
                norm_layer=partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01, affine=cfg.pdnorm_affine),
                conditions=cfg.pdnorm_conditions,
                decouple=cfg.pdnorm_decouple,
                adaptive=cfg.pdnorm_adaptive,
            )
        else:
            bn_layer = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        if cfg.pdnorm_ln:
            ln_layer = partial(
                PDNorm,
                norm_layer=partial(nn.LayerNorm, elementwise_affine=cfg.pdnorm_affine),
                conditions=cfg.pdnorm_conditions,
                decouple=cfg.pdnorm_decouple,
                adaptive=cfg.pdnorm_adaptive,
            )
        else:
            ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=cfg.in_channels,
            embed_channels=cfg.enc_channels[0],
            norm_layer=bn_layer,
            act_layer=act_layer,
        )

        # encoder
        enc_drop_path = [x.item() for x in torch.linspace(0, cfg.drop_path, sum(cfg.enc_depths))]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[sum(cfg.enc_depths[:s]) : sum(cfg.enc_depths[: s + 1])]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    SerializedPooling(
                        in_channels=cfg.enc_channels[s - 1],
                        out_channels=cfg.enc_channels[s],
                        stride=cfg.stride[s - 1],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                        grid_mode=cfg.grid_mode,
                    ),
                    name="down",
                )
                if cfg.temporal_merger:
                    enc.add(
                        Merger(
                            window_size=cfg.merge_window_size[s - 1],
                            time_emb=cfg.enc_channels[s] if cfg.with_time_emb else None,
                        ),
                        name="merger",
                    )

            for i in range(cfg.enc_depths[s]):
                enc.add(
                    Block(
                        channels=cfg.enc_channels[s],
                        num_heads=cfg.enc_num_head[s],
                        patch_size=cfg.enc_patch_size[s],
                        mlp_ratio=cfg.mlp_ratio,
                        qkv_bias=cfg.qkv_bias,
                        qk_scale=cfg.qk_scale,
                        attn_drop=cfg.attn_drop,
                        proj_drop=cfg.proj_drop,
                        drop_path=enc_drop_path_[i],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=cfg.pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=cfg.enable_rpe,
                        enable_flash=cfg.enable_flash,
                        upcast_attention=cfg.upcast_attention,
                        upcast_softmax=cfg.upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        if not self.cls_mode:
            dec_drop_path = [x.item() for x in torch.linspace(0, cfg.drop_path, sum(cfg.dec_depths))]
            self.dec = PointSequential()
            dec_channels = list(cfg.dec_channels) + [cfg.enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[sum(cfg.dec_depths[:s]) : sum(cfg.dec_depths[: s + 1])]
                dec_drop_path_.reverse()
                dec = PointSequential()

                if cfg.temporal_merger:
                    dec.add(Unmerger(), name="unmerger")
                dec.add(
                    SerializedUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=cfg.enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=bn_layer,
                        act_layer=act_layer,
                    ),
                    name="up",
                )

                for i in range(cfg.dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=cfg.dec_num_head[s],
                            patch_size=cfg.dec_patch_size[s],
                            mlp_ratio=cfg.mlp_ratio,
                            qkv_bias=cfg.qkv_bias,
                            qk_scale=cfg.qk_scale,
                            attn_drop=cfg.attn_drop,
                            proj_drop=cfg.proj_drop,
                            drop_path=dec_drop_path_[i],
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=cfg.pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=cfg.enable_rpe,
                            enable_flash=cfg.enable_flash,
                            upcast_attention=cfg.upcast_attention,
                            upcast_softmax=cfg.upcast_softmax,
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")

            self.dec.add(
                MLP(
                    in_channels=dec_channels[0],
                    hidden_channels=dec_channels[0],
                    out_channels=cfg.out_channels,
                    act_layer=nn.Identity,
                    prepend_layer_norm_act=(ln_layer, act_layer),
                ),
            )

    def forward(self, data_dict):
        """
        A data_dict is a dictionary containing properties of a batched point cloud.
        It should contain the following properties for PTv3:
        1. "feat": feature of point cloud
        2. "grid_coord": discrete coordinate after grid sampling (voxelization) or "coord" + "grid_size"
        3. "offset" or "batch": https://github.com/Pointcept/Pointcept?tab=readme-ov-file#offset
        """
        point = Point(data_dict)
        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.embedding(point)
        # print("ENCODER")
        point = self.enc(point)
        if not self.cls_mode:
            # print("DECODER")
            point = self.dec(point)
        return point

    def step(
        self,
        xyz: Float[torch.Tensor, "batch time gaussian 3"],
        features: Float[torch.Tensor, "batch time gaussian feat"],
        state_mask: Optional[torch.Tensor] = None,
    ) -> Float[torch.Tensor, "batch time_pred gaussian out_feat"]:
        # prepare data_dict

        # clip xyz to the maximum distane
        # xyz = torch.clamp(xyz, -self.max_distance, self.max_distance)

        data_dict, state_mask = batch_to_pointdict(
            xyz, features, state_mask, self.grid_size, self.temporal_merger
        )
        point = self(data_dict)
        # return the feature of the point cloud
        out = pointdict_to_batch(point, xyz, state_mask, keep_time_dim=self.temporal_merger)
        return out


def pointdict_to_batch(
    point: Point,
    xyz: Float[torch.Tensor, "batch time gaussian 3"],
    state_mask: Optional[Bool[torch.Tensor, "btg 1"]] = None,
    keep_time_dim=False,
) -> Float[torch.Tensor, "batch time_pred gaussian feat"]:
    B, T, N = xyz.shape[:3]

    if state_mask is not None:

        # v1
        # features = torch.zeros_like(state_mask, dtype=point.feat.dtype).repeat(1, point.feat.shape[-1])
        # features[state_mask[..., 0]] = point.feat

        features = torch.zeros_like(state_mask, dtype=point.feat.dtype).repeat(1, point.feat.shape[-1])
        features = features.masked_scatter(state_mask, point.feat)

    else:
        features = point.feat

    if keep_time_dim:
        features = eo.rearrange(features, "(b t n) c -> b t n c", b=B, t=T, n=N)
    else:
        features = eo.rearrange(features, "(b n) c -> b 1 n c", b=B, n=N)

    return features


def batch_to_pointdict(
    xyz: Float[torch.Tensor, "batch time gaussian 3"],
    features: Float[torch.Tensor, "batch time gaussian feat"],
    state_mask: Optional[Bool[torch.Tensor, "batch time gaussian _"]] = None,
    grid_size=0.01,
    keep_time_dim=False,
):
    B, T, N = xyz.shape[:3]

    if keep_time_dim:
        point_dict = Dict(
            coord=eo.rearrange(xyz, "b t n c -> (b t n) c"),
            feat=eo.rearrange(features, "b t n c -> (b t n) c"),
            batch=eo.repeat(torch.arange(B).cuda(), "b -> (b t n)", t=T, n=N),
            time_coord=eo.repeat(torch.arange(T).cuda(), "t -> (b t n)", b=B, n=N),
        )
        state_mask = eo.rearrange(state_mask, "b t n c -> (b t n) c") if state_mask is not None else None
    else:
        point_dict = Dict(
            coord=eo.rearrange(xyz[:, -1], "b n c -> (b n) c"),
            feat=eo.rearrange(features, "b t n c -> (b n) (t c)"),
            batch=eo.repeat(torch.arange(B).cuda(), "b -> (b n)", n=N),
        )
        state_mask = eo.rearrange(state_mask[:, -1], "b n c -> (b n) c") if state_mask is not None else None

    if state_mask is not None:
        point_dict = {k: v[state_mask.squeeze(-1)].contiguous() for k, v in point_dict.items()}
    point_dict["grid_size"] = grid_size

    return point_dict, state_mask


@torch.inference_mode()
def time_func(func, x, iters=10, with_backward=False):
    import time

    try:
        func(x)  # first doesnt count

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t = time.perf_counter()

        if not with_backward:
            with torch.no_grad():
                for _ in range(iters):
                    func(x)
        else:
            x = x.requires_grad_()
            for _ in range(iters):
                out = func(x)
                out = out[0] if isinstance(out, tuple) else out
                torch.autograd.grad(out, x, out, only_inputs=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return (time.perf_counter() - t) / iters

    except RuntimeError as e:
        if "out of memory" not in str(e):
            raise RuntimeError(e)
        else:
            print(e)
        torch.cuda.empty_cache()
        return float("inf")


def test_temporal_merger():
    torch.set_printoptions(linewidth=250, edgeitems=65)

    def dec2bin(x, bits):
        # mask = 2 ** torch.arange(bits).to(x.device, x.dtype)
        mask = 2 ** torch.arange(bits - 1, -1, -1).to(x.device, x.dtype)
        return x.unsqueeze(-1).bitwise_and(mask).ne(0).float()

    # test tempo-spatial encoding
    B = 2
    T = 4
    N = 7500

    xyz = torch.rand(B, T, N, 3).cuda()
    feat = torch.rand(B, T, N, 6).cuda()

    point_dict = batch_to_pointdict(xyz, feat, grid_size=0.01, keep_time_dim=True)
    point = Point(point_dict)
    point.serialization()
    point.sparsify(pad=1)

    merger1 = Merger(window_size=2).cuda()
    merger2 = Merger(window_size=2).cuda()
    merger3 = Merger(window_size=2).cuda()

    unmerger3 = Unmerger().cuda()
    unmerger2 = Unmerger().cuda()
    unmerger1 = Unmerger().cuda()

    point1 = merger1(point)
    point2 = merger2(point1)
    point3 = merger3(point2)

    point32 = unmerger3(point3)
    point21 = unmerger2(point32)
    point10 = unmerger1(point21)

    merger_embed = Merger(window_size=2, time_emb=6).cuda()
    _ = merger_embed(point)

    for key in point10.keys():
        if not isinstance(point10[key], spconv.SparseConvTensor):
            assert torch.allclose(torch.tensor(point10[key]), torch.tensor(point[key]), atol=1e-6)
        else:
            dense = (point10[key] + point[key].minus()).dense()
            assert torch.allclose(dense, torch.zeros_like(dense), atol=1e-6)

    point_dict = batch_to_pointdict(xyz, feat, grid_size=0.01, keep_time_dim=True)

    # Setup Model
    cfg = PointTrainsformerV3Cfg(temporal_merger=True, in_channels=6, enable_flash=True)
    model = PointTransformerV3(cfg=cfg).cuda()
    model.eval()
    print("input batch, time, num", B, T, N)
    out: Point = model(point_dict)
    print(
        "output batch, time, num",
        int(out.batch_ixs()[-1]) + 1,
        int(out.time_ixs()[-1]) + 1,
        len(out.feat) // B // T,
    )

    print(time_func(model, point_dict, iters=10, with_backward=False))


def test_no_temporal():
    N = 30000 * 2
    coord = torch.rand(N, 3).cuda()
    feat = torch.rand(N, 6).cuda()
    offset = torch.tensor([N]).cuda()

    point_dict = Dict(
        feat=feat,
        coord=coord,
        offset=offset,
        grid_size=0.04,
    )
    point_dict = Point(point_dict)
    point_dict.serialization(order=["z", "hilbert"], shuffle_orders=True)
    point_dict.sparsify()

    pool_lay = SerializedPooling(in_channels=6, out_channels=6, stride=2).cuda()

    print(pool_lay(point_dict).feat.shape)

    cfg = PointTrainsformerV3Cfg(temporal_merger=False, in_channels=6, enable_flash=True)
    model = PointTransformerV3(cfg=cfg).cuda()
    model.eval()
    print(time_func(model, point_dict, iters=10, with_backward=False))


if __name__ == "__main__":
    test_no_temporal()
    test_temporal_merger()
    test_temporal_merger()
