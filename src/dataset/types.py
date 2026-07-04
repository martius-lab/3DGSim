from typing import Callable, Literal, Optional, TypedDict

from einops.packing import Shape, pack, unpack
from jaxtyping import Bool, Float, Int64
from torch import Tensor

Stage = Literal["train", "val", "test"]


# The following types mainly exist to make type-hinted keys show up in VS Code. Some
# dimensions are annotated as "_" because either:
# 1. They're expected to change as part of a function call (e.g., resizing the dataset).
# 2. They're expected to vary within the same function call (e.g., the number of views,
#    which differs between context and target BatchedViews).


class BatchedTempViews(TypedDict, total=False):
    extrinsics: Float[Tensor, "batch time _ 4 4"]  # batch time view 4 4
    intrinsics: Float[Tensor, "batch time _ 3 3"]  # batch time view 3 3
    image: Float[Tensor, "batch time _ _ _ _"]  # batch time view channel height width
    near: Float[Tensor, "batch time _"]  # batch time view
    far: Float[Tensor, "batch time _"]  # batch time view
    index: Int64[Tensor, "batch time _"]  # batch time view
    state_mask: Optional[Bool[Tensor, "batch time _ 1 height width"]] = None
    static_float: Optional[Float[Tensor, "batch time _ 1 height width"]] = None


class BatchedTempExample(TypedDict, total=False):
    target: BatchedTempViews
    context: BatchedTempViews
    scene: list[str]  # one scene per batch


class BatchedViews(TypedDict, total=False):
    extrinsics: Float[Tensor, "batch _ 4 4"]  # batch view 4 4
    intrinsics: Float[Tensor, "batch _ 3 3"]  # batch view 3 3
    image: Float[Tensor, "batch _ _ _ _"]  # batch view channel height width
    near: Float[Tensor, "batch _"]  # batch view
    far: Float[Tensor, "batch _"]  # batch view
    index: Int64[Tensor, "batch _"]  # batch view
    state_mask: Optional[Bool[Tensor, "batch _ 1 height width"]] = None
    static_float: Optional[Float[Tensor, "batch _ 1 height width"]] = None


class BatchedExample(TypedDict, total=False):
    target: BatchedViews
    context: BatchedViews
    scene: list[str]


class UnbatchedViews(TypedDict, total=False):
    extrinsics: Float[Tensor, "_ 4 4"]
    intrinsics: Float[Tensor, "_ 3 3"]
    image: Float[Tensor, "_ _ height width"]
    near: Float[Tensor, " _"]
    far: Float[Tensor, " _"]
    index: Int64[Tensor, " _"]
    state_mask: Optional[Bool[Tensor, "_ 1 height width"]] = None
    static_float: Optional[Float[Tensor, "_ 1 height width"]] = None


class UnbatchedExample(TypedDict, total=False):
    target: UnbatchedViews
    context: UnbatchedViews
    scene: str


AnyExample = BatchedExample | UnbatchedExample | BatchedTempExample
AnyViews = BatchedViews | UnbatchedViews | BatchedTempViews

# A data shim modifies the example after it's been returned from the data loader.
DataShim = Callable[[AnyExample], AnyExample]


pattern_dict = {
    "extrinsics": "{batch} view four1 four2",
    "intrinsics": "{batch} view three1 three2",
    "image": "{batch} view channel height width",
    "state_mask": "{batch} view channel height width",
    "static_float": "{batch} view channel height width",
    "near": "{batch} view",
    "far": "{batch} view",
    "index": "{batch} view",
}

import torch


def stack_views(list_any_views: list[AnyViews], dim=0) -> AnyViews:
    return {k: torch.stack([v[k] for v in list_any_views], dim=dim) for k in list_any_views[0].keys()}


def to_batched_views(any_views: AnyViews) -> BatchedViews:
    return BatchedViews(
        **{
            k: pack([any_views[k]], v.format(batch="*"))[0]
            for k, v in pattern_dict.items()
            if any_views.get(k, None) is not None
        }
    )


def to_batched_example(any_example: AnyExample) -> BatchedExample:
    return BatchedExample(
        target=to_batched_views(any_example["target"]),
        context=to_batched_views(any_example["context"]),
        scene=any_example["scene"],
    )


def to_batched_temp_views(any_views: AnyViews, ps: list[Shape]) -> BatchedTempViews:
    return BatchedTempViews(
        **{
            k: (unpack(any_views[k], ps, v.format(batch="*"))[0])
            for k, v in pattern_dict.items()
            if any_views.get(k, None) is not None
        }
    )


def to_batched_temp_example(
    any_example: AnyExample,
    ps_c: list[Shape],
    ps_t: list[Shape],
) -> BatchedTempExample:
    if len(ps_c[0]) == 1:
        ps_c = [ps_c[0] + [1]]
    if len(ps_t[0]) == 1:
        ps_t = [ps_t[0] + [1]]

    return BatchedTempExample(
        context=to_batched_temp_views(any_example["context"], ps_c),
        target=to_batched_temp_views(any_example["target"], ps_t),
        scene=any_example["scene"],  # one scene per batch
    )


def to_batched_temp_shim(batch: AnyExample) -> BatchedTempExample:
    temporal = len(batch["context"]["image"].shape) == 3 + 1 + 1 + 1
    if temporal:
        return batch
    else:
        b = batch["context"]["image"].shape[0]
        return to_batched_temp_example(batch, [[b, 1]], [[b, 1]])


def test_to_batched_example():
    import torch

    B = 4
    T = 4
    V = 5

    def random_temp_view() -> BatchedTempViews:
        return BatchedTempViews(
            extrinsics=torch.randn(B, T, V, 4, 4),
            intrinsics=torch.randn(B, T, V, 3, 3),
            image=torch.randn(B, T, V, 3, 256, 256),
            near=torch.randn(B, T, V),
            far=torch.randn(B, T, V),
            index=torch.randint(0, 10, (B, T, V)),
        )

    batched_temp_example = BatchedExample(
        target=random_temp_view(),
        context=random_temp_view(),
        scene=["scene1"] * B,
    )

    *trg_bt, _, _, h, w = batched_temp_example["target"]["image"].shape
    *ctx_bt, _, _, h, w = batched_temp_example["context"]["image"].shape

    batched_example = to_batched_example(batched_temp_example)

    batched_temp_2 = to_batched_temp_example(batched_example, [ctx_bt], [trg_bt])

    for key in ["context", "target"]:
        for kkey in batched_temp_example[key]:
            assert torch.allclose(
                batched_temp_example[key][kkey],
                batched_temp_2[key][kkey],
                atol=1e-6,
                rtol=1e-6,
            ), f"TEST ERROR: {(key, kkey)} don't match"

    batched_temp_example2 = BatchedExample(
        target=random_temp_view(),
        context=random_temp_view(),
        scene=["scene1"] * B,
    )
    stacked_example = stack_views([batched_temp_example["target"], batched_temp_example2["target"]])

    for key in ["context", "target"]:
        for kkey in batched_temp_example[key]:
            assert torch.allclose(
                batched_temp_example[key][kkey],
                stacked_example[key][kkey][0],
                atol=1e-6,
                rtol=1e-6,
            ), f"TEST ERROR: {(key, kkey)} don't match"


def test_to_batched_temp_shim():
    import torch

    B = 4
    V = 5

    def random_batched_view() -> BatchedViews:
        return BatchedViews(
            extrinsics=torch.randn(B, V, 4, 4),
            intrinsics=torch.randn(B, V, 3, 3),
            image=torch.randn(B, V, 3, 256, 256),
            near=torch.randn(B, V),
            far=torch.randn(B, V),
            index=torch.randint(0, 10, (B, V)),
        )

    batched_example = BatchedExample(
        target=random_batched_view(),
        context=random_batched_view(),
        scene=["scene1"] * B,
    )
    temp_batched_example = to_batched_temp_shim(batched_example)

    batched_example_2 = to_batched_example(temp_batched_example)
    for key in ["context", "target"]:
        for kkey in batched_example[key]:
            assert torch.allclose(
                batched_example[key][kkey],
                batched_example_2[key][kkey],
                atol=1e-6,
                rtol=1e-6,
            ), f"TEST ERROR: {(key, kkey)} don't match"
    assert (
        batched_example["scene"] == batched_example_2["scene"]
    ), f"TEST ERROR: scene don't match, {batched_example['scene']} != {batched_example_2['scene']}"


if __name__ == "__main__":
    # test_to_batched_example()
    test_to_batched_temp_shim()
